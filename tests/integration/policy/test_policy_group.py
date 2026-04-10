import os
from typing import Tuple

import pytest
from gymnasium import spaces as gym_spaces
from tqdm import tqdm

from rlightning.types import EnvMeta
from rlightning.utils.builders import (
    build_data_buffer,
    build_env_group,
    build_policy_group,
)
from rlightning.utils.config import (
    BufferConfig,
    PolicyConfig,
    TrainConfig,
    WeightBufferConfig,
    ClusterConfig,
)
from rlightning.utils.ray import resolve_object
from rlightning.utils.utils import InternalFlag
from tests.test_utils import _make_env_config, require_env_backend

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _auto_ray_cluster():
    # Ensure a Ray cluster is available for this module's tests
    from tests.test_utils import setup_ray_cluster, teardown_ray_cluster

    setup_ray_cluster(num_cpus=4, num_gpus=4, local_mode=False)
    yield
    teardown_ray_cluster()


def _make_env_group(env_backend, task_name, as_remote):
    require_env_backend(env_backend)
    num_workers = 1
    num_envs = 1
    env_cfgs = [_make_env_config(env_backend, task_name, num_workers, num_envs)]

    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_ENV == as_remote
    return build_env_group(env_cfgs=env_cfgs)


def _make_policy_config():
    return PolicyConfig(
        type="SimplePPOPolicy",
        weight_buffer=WeightBufferConfig(
            type="WeightBuffer",
            buffer_strategy="Double",
        ),
        rollout_mode="sync",
        optim_cfg=WeightBufferConfig(
            lr=1e-4,
        ),
    )


def _make_train_config():
    return TrainConfig(
        max_epochs=10,
        lr=1e-3,
        max_rollout_steps=10,
        gamma=0.99,
        ppo_epochs=1,
        batch_size=10,
        minibatch_size=2,
        clip_ratio=0.2,
        entropy_coef=1e-3,
        value_coef=0.1,
        max_grad_norm=0.5,
        epochs=2,
    )


def _make_buffer_config(capacity: int = 100) -> BufferConfig:
    return BufferConfig.from_dict(
        {
            "type": "RolloutBuffer",
            "capacity": capacity,
            "storage": {
                "type": "unified",
                "device": "cpu",
            },
            "sampler": {
                "type": "uniform",
            },
        }
    )


def _make_buffer(buffer_cfg, env_meta_list=None, as_remote=False):
    from tests.test_utils import _episode_postprocess_fn

    os.environ["RLIGHTNING_REMOTE_STORAGE"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_STORAGE == as_remote

    buffer = build_data_buffer(
        buffer_cls=buffer_cfg.type,
        buffer_cfg=buffer_cfg,
        # obs_preprocessor=obs_preprocessor,
        postprocess_fn=_episode_postprocess_fn,
    )
    buffer.init(env_meta_list)

    return buffer


def _make_policy_group(policy_cfg, train_as_remote, eval_as_remote):
    os.environ["RLIGHTNING_REMOTE_TRAIN"] = "1" if train_as_remote else "0"
    os.environ["RLIGHTNING_REMOTE_EVAL"] = "1" if eval_as_remote else "0"

    return build_policy_group(
        policy_cls=policy_cfg.type,
        policy_cfg=policy_cfg,
        cluster_cfg=ClusterConfig(),
    )


@pytest.mark.parametrize(
    "env_backend,task_name",
    [
        ["ale", "ALE/Alien-v5"],
    ],
)
@pytest.mark.parametrize("eval_mode", [False, True])
@pytest.mark.parametrize("env_as_remote", [True, False])
@pytest.mark.parametrize("policy_as_remote", [True, False])
def test_policy_group_with_simple_ppo(
    env_backend,
    task_name: str,
    eval_mode: bool,
    env_as_remote: bool,
    policy_as_remote: bool,
):
    env_group = _make_env_group(env_backend, task_name, env_as_remote)

    policy_cfg = _make_policy_config()
    train_config = _make_train_config()

    policy_group = _make_policy_group(
        policy_cfg, train_as_remote=policy_as_remote, eval_as_remote=policy_as_remote
    )

    env_meta = resolve_object(env_group.init()[0])

    assert isinstance(env_meta, EnvMeta), type(env_meta)

    assert isinstance(env_meta.observation_space, gym_spaces.Space), type(
        env_meta.observation_space
    )

    assert isinstance(env_meta.action_space, gym_spaces.Space), type(env_meta.action_space)

    if eval_mode:
        episode_meta = policy_group.init_eval(None, env_meta)
    else:
        policy_group.init_train(train_config, env_meta)


@pytest.mark.parametrize(
    "env_backend,task_name",
    [
        ["ale", "ALE/Adventure-v5"],
    ],
)
@pytest.mark.parametrize("env_as_remote", [True, False])
@pytest.mark.parametrize("policy_as_remote", [True, False])
def test_simple_ppo_evaluation(
    env_backend: str,
    task_name: str,
    env_as_remote: bool,
    policy_as_remote: bool,
):
    """Test the functionality of rollouts

    The coverage indluces:

        1. Support RGB input environments only
        2. The correctness of V-trace has not been validated yet

    Args:
        env_backend (str): Registered environment backend
        task_name (str): Registered environment name
    """

    max_rollout_steps: int = 50

    env_group = _make_env_group(env_backend, task_name, env_as_remote)

    policy_cfg = _make_policy_config()
    # train_config = _make_train_config()

    policy_group = _make_policy_group(
        policy_cfg, train_as_remote=policy_as_remote, eval_as_remote=policy_as_remote
    )

    env_meta = resolve_object(env_group.init()[0])

    assert isinstance(env_meta.observation_space, gym_spaces.Space), type(
        env_meta.observation_space
    )

    assert isinstance(env_meta.action_space, gym_spaces.Space), type(env_meta.action_space)

    policy_group.init_eval(env_meta=env_meta)

    batched_env_ret, _ = env_group.reset(seed=0)

    for _ in tqdm(range(max_rollout_steps), disable=True):
        batched_policy_resp = policy_group.rollout_batch(batched_env_ret)
        batched_env_ret, _ = env_group.step(batched_policy_resp)


@pytest.mark.parametrize(
    "env_backend,task_name",
    [
        ["ale", "ALE/Adventure-v5"],
    ],
)
@pytest.mark.parametrize("env_as_remote", [True, False])
@pytest.mark.parametrize("buffer_as_remote", [True, False])
@pytest.mark.parametrize("policy_as_remote", [True, False])
def test_simple_ppo_training(
    env_backend: str,
    task_name: str,
    env_as_remote: bool,
    buffer_as_remote: bool,
    policy_as_remote: bool,
):
    """Test the functionality of SimplePPOPolicy

    The coverage includes:

        1. Support RGB input environments only
        2. The correctness of V-trace has not been validated yet

    Args:
        env_backend (str): Registered environment backend
        task_name (str): Task name, or registered environment name
    """

    batch_size: int = 10

    env_group = _make_env_group(env_backend, task_name, env_as_remote)
    env_meta_list = resolve_object(env_group.init())
    env_meta = env_meta_list[0]

    policy_cfg = _make_policy_config()
    train_config = _make_train_config()
    policy_cfg.train_config = train_config

    # create buffer here
    buffer_cfg = _make_buffer_config(capacity=100)
    buffer = _make_buffer(buffer_cfg, env_meta_list, as_remote=buffer_as_remote)

    policy_group = _make_policy_group(
        policy_cfg, train_as_remote=policy_as_remote, eval_as_remote=policy_as_remote
    )

    assert isinstance(env_meta.observation_space, gym_spaces.Space), type(
        env_meta.observation_space
    )
    assert isinstance(env_meta.action_space, gym_spaces.Space), type(env_meta.action_space)

    policy_group.init_train(train_config, env_meta)
    policy_group.init_eval(None, env_meta)

    batched_env_ret, _ = env_group.reset(seed=0)

    for _ in range(train_config.epochs):
        for _ in tqdm(range(train_config.max_rollout_steps), disable=True):
            batched_policy_resp = policy_group.rollout_batch(batched_env_ret)
            buffer.add_batched_transition(batched_env_ret, batched_policy_resp)
            batched_env_ret, _ = env_group.step(batched_policy_resp)

        batched_policy_resp = policy_group.rollout_batch(batched_env_ret)
        buffer.add_batched_transition(batched_env_ret, batched_policy_resp)
        buffer.truncate_episodes(batched_policy_resp.ids())

        data = buffer.sample(batch_size=batch_size)
        policy_group.update_dataset(data)
        info = policy_group.train()
        print(info)
