import os

import numpy as np
import pytest
import ray
import torch
from gymnasium import spaces as gym_spaces

from rlightning.types import EnvMeta
from rlightning.utils.builders import build_policy_group
from rlightning.utils.config import Config, PolicyConfig, TrainConfig, WeightBufferConfig, ClusterConfig
from rlightning.weights.weight_buffer import WeightBuffer

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

POLICY_AS_REMOTE = True


@pytest.fixture(scope="module", autouse=True)
def _auto_ray_cluster():
    # Ensure a Ray cluster is available for this module's tests
    from tests.test_utils import setup_ray_cluster, teardown_ray_cluster

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    os.environ["RLIGHTNING_REMOTE_TRAIN"] = "1" if POLICY_AS_REMOTE else "0"
    os.environ["RLIGHTNING_REMOTE_EVAL"] = "1" if POLICY_AS_REMOTE else "0"

    setup_ray_cluster(num_cpus=4, num_gpus=4, local_mode=False)
    yield
    teardown_ray_cluster()


def mock_weight_buffer_config(buffer_strategy: str = "Double"):
    if buffer_strategy == "Double":
        return WeightBufferConfig(
            type="WeightBuffer",
            buffer_strategy=buffer_strategy,
        )
    elif buffer_strategy == "Shared":
        return WeightBufferConfig(
            type="CPUWeightBuffer",
            buffer_strategy=buffer_strategy,
        )


def mock_env_meta():
    """Build a minimal env_meta compatible with SimplePPOPolicy."""
    action_space = gym_spaces.Discrete(5)
    return EnvMeta(
        env_id=None,
        action_space=action_space,
    )


def mock_policy_config(buffer_strategy: str = "Double"):
    return PolicyConfig(
        type="SimplePPOPolicy",
        rollout_mode="sync",
        weight_buffer=mock_weight_buffer_config(buffer_strategy),
        optim_cfg=Config(lr=0.001),
        train_config=TrainConfig(max_epochs=1),
    )


def mock_policy_group(policy_cfg):
    return build_policy_group(
        policy_cls=policy_cfg.type,
        policy_cfg=policy_cfg,
        cluster_cfg=ClusterConfig(),
    )


def _get_policy_state_dict(policy, policy_as_remote):
    model_parameters = (
        ray.get(policy.get_trainable_parameters.remote())
        if policy_as_remote
        else policy.get_trainable_parameters()
    )
    return model_parameters


def _resolve_weight_buffer(eval_policy):
    """Resolve the actual weight buffer object/actor from a policy."""
    if isinstance(eval_policy, ray.actor.ActorHandle):
        return ray.get(eval_policy.get_weight_buffer.remote())
    return eval_policy.get_weight_buffer()


def _get_weight_buffer_state_dict(weight_buffer):
    """Helper to get state dict from weight buffer."""
    if isinstance(weight_buffer, ray.actor.ActorHandle):
        return ray.get(weight_buffer.get_state_dict.remote())
    return weight_buffer.get_state_dict()


def _add_state_dict_to_buffer(weight_buffer, state_dict):
    """Helper to add a state dict to weight buffer."""
    if isinstance(weight_buffer, ray.actor.ActorHandle):
        ray.get(weight_buffer.add_state_dict.remote(state_dict))
    else:
        weight_buffer.add_state_dict(state_dict)


@pytest.mark.parametrize("policy_as_remote", [POLICY_AS_REMOTE])
@pytest.mark.parametrize("buffer_strategy", ["Double", "Shared"])
def test_init_weight_buffer_with_simple_ppo(
    policy_as_remote: bool,
    buffer_strategy: str,
):
    """Test weight buffer initialization with SimplePPO policy"""
    policy_cfg = mock_policy_config(buffer_strategy=buffer_strategy)

    # Initialize weight buffer for eval policy
    policy_group = mock_policy_group(policy_cfg)

    eval_policy = policy_group.eval_list[0]
    weight_buffer = _resolve_weight_buffer(eval_policy)

    # Verify weight buffer is initialized
    assert weight_buffer is not None
    if buffer_strategy == "Shared":
        assert isinstance(weight_buffer, ray.actor.ActorHandle)
    elif buffer_strategy == "Double":
        assert isinstance(weight_buffer, WeightBuffer)


@pytest.mark.parametrize("policy_as_remote", [POLICY_AS_REMOTE])
@pytest.mark.parametrize("buffer_strategy", ["Double", "Shared"])
def test_send_weights_with_simple_ppo(
    policy_as_remote: bool,
    buffer_strategy: str,
):
    """Test sending weights with SimplePPO policy"""
    policy_cfg = mock_policy_config(buffer_strategy=buffer_strategy)

    # Initialize weight buffer for eval policy
    policy_group = mock_policy_group(policy_cfg)

    # Initialize eval and train policies
    env_meta = mock_env_meta()
    policy_group.init_eval(env_meta=env_meta)
    policy_group.init_train(policy_cfg.train_config, env_meta=env_meta)

    train_policy = policy_group.train_list[0]
    eval_policy = policy_group.eval_list[0]

    # send weights to eval policy
    if policy_as_remote:
        if buffer_strategy == "Double":
            ray.get(train_policy.send_weights.remote([eval_policy]))
            weight_buffer = _resolve_weight_buffer(eval_policy)
        elif buffer_strategy == "Shared":
            weight_buffer = next(iter(policy_group.shared_weight_buffer_map.values()))[
                "shared_weight_buffer"
            ]
            ray.get(train_policy.send_weights.remote([], shared_weight_buffer=weight_buffer))

    # Verify weights were added
    retrieved_dict = _get_weight_buffer_state_dict(weight_buffer)
    assert retrieved_dict != {}, "Initial state dict is empty"

    # Create a state dict with model parameters
    initial_state_dict = _get_policy_state_dict(train_policy, policy_as_remote)

    # Verify the state dict contains expected keys
    for module_name, module_state_dict in retrieved_dict.items():
        print(module_name)
        for name in module_state_dict.keys():
            assert name in initial_state_dict[module_name]
            assert module_state_dict[name].shape == initial_state_dict[module_name][name].shape


@pytest.mark.parametrize("policy_as_remote", [POLICY_AS_REMOTE])
@pytest.mark.parametrize("buffer_strategy", ["Double", "Shared"])
def test_update_weights_with_simple_ppo(
    policy_as_remote: bool,
    buffer_strategy: str,
):
    """Test updating weights with SimplePPO policy"""
    policy_cfg = mock_policy_config(buffer_strategy=buffer_strategy)

    # Initialize weight buffer for eval policy
    policy_group = mock_policy_group(policy_cfg)

    # Initialize eval and train policies
    env_meta = mock_env_meta()
    policy_group.init_eval(env_meta=env_meta)
    policy_group.init_train(policy_cfg.train_config, env_meta=env_meta)

    train_policy = policy_group.train_list[0]
    eval_policy = policy_group.eval_list[0]

    # send weights to eval policy
    if policy_as_remote:
        if buffer_strategy == "Double":
            ray.get(train_policy.send_weights.remote([eval_policy]))
            weight_buffer = _resolve_weight_buffer(eval_policy)
        elif buffer_strategy == "Shared":
            weight_buffer = next(iter(policy_group.shared_weight_buffer_map.values()))[
                "shared_weight_buffer"
            ]
            ray.get(train_policy.send_weights.remote([], shared_weight_buffer=weight_buffer))

    # Get initial weights from train policy
    initial_state_dict = _get_policy_state_dict(train_policy, policy_as_remote)

    ray.get(eval_policy.update_weights_from_buffer.remote())
    eval_state_dict = _get_policy_state_dict(eval_policy, policy_as_remote)

    # Verify eval policy weights were updated from buffer
    for module_name, module_state_dict in eval_state_dict.items():
        print(module_name)
        for name in module_state_dict.keys():
            assert name in initial_state_dict[module_name]
            assert torch.allclose(
                eval_state_dict[module_name][name], initial_state_dict[module_name][name]
            )


@pytest.mark.parametrize("buffer_strategy", ["Double", "Shared"])
def test_weight_buffer_clear_and_reinit(buffer_strategy: str):
    """Test clearing and reinitializing weight buffer"""
    from rlightning.weights.utils import build_weight_buffer

    weight_buffer_cfg = mock_weight_buffer_config(buffer_strategy)
    node_id = ray.get_runtime_context().get_node_id()
    weight_buffer = build_weight_buffer(weight_buffer_cfg.type, weight_buffer_cfg, node_id=node_id)

    # First state dict
    state_dict1 = {
        "layer1": {
            "weight": (
                torch.randn(10, 10) if buffer_strategy == "Double" else np.random.randn(10, 10)
            )
        }
    }
    _add_state_dict_to_buffer(weight_buffer, state_dict1)

    # Check readiness
    if isinstance(weight_buffer, ray.actor.ActorHandle):
        is_ready = ray.get(weight_buffer.is_ready.remote())
    else:
        is_ready = weight_buffer.is_ready
    assert is_ready

    # Clear buffer
    if isinstance(weight_buffer, ray.actor.ActorHandle):
        ray.get(weight_buffer.clear.remote())
        is_ready = ray.get(weight_buffer.is_ready.remote())
    else:
        weight_buffer.clear()
        is_ready = weight_buffer.is_ready()
    assert not is_ready

    # Add new state dict
    state_dict2 = {
        "layer2": {
            "weight": torch.randn(5, 5) if buffer_strategy == "Double" else np.random.randn(5, 5)
        }
    }
    _add_state_dict_to_buffer(weight_buffer, state_dict2)

    # Verify new state dict is present
    retrieved = _get_weight_buffer_state_dict(weight_buffer)
    assert "layer2" in retrieved
    assert "layer1" not in retrieved
