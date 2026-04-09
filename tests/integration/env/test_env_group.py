import os
from pathlib import Path

import hydra
import pytest
import torch

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from rlightning.types import BatchedData, PolicyResponse
from rlightning.utils.builders import build_env_group
from rlightning.utils.ray import resolve_object

# set parent dir for examples
# Resolve repo root even after moving tests into tests/integration/...
EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"
from rlightning.utils.utils import InternalFlag
from tests.test_utils import _make_env_config, require_env_backend

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _auto_ray_cluster():
    # Ensure a Ray cluster is available for this module's tests
    from tests.test_utils import setup_ray_cluster, teardown_ray_cluster

    setup_ray_cluster(num_cpus=4, num_gpus=2)
    yield
    teardown_ray_cluster()


@pytest.fixture
def isaac_cleanup_registry(scope="module"):
    """
    Fixture that provides a list of instances containing isaaclab for cleanup.
    The creation happens in the test, but cleanup happens here.
    Used to avoid clear failure caused by isaaclab in testing.
    """
    registry = []
    yield registry

    # Teardown: runs after the test finishes
    print(f"Fixture teardown: Closing {len(registry)} registered...")
    for instance in registry:
        try:
            if hasattr(instance, "close"):
                instance.close()
            else:
                print("Warning: instance does not have a close() method.")
        except Exception as e:
            print(f"Warning: Error closing instance: {e}")
    print("Fixture teardown: All registered instance closed.")


def _mock_batched_policy_ret(action_spaces, env_ids):
    action_list = [resolve_object(action_space).sample() for action_space in action_spaces]
    policy_resp_list = [
        PolicyResponse(env_id=env_id, action=action, log_prob=0.0, entropy=0.0)
        for action, env_id in zip(action_list, env_ids)
    ]
    batched_policy_resp = BatchedData(env_ids, policy_resp_list)
    return batched_policy_resp


def _mock_policy_ret(action_space, env_id):
    action = action_space.sample()
    policy_resp = PolicyResponse(env_id=env_id, action=action, log_prob=0.0, entropy=0.0)
    return policy_resp


# # to be fixed: it can pass individually but fails in `make test-unit`
# @pytest.mark.parametrize(
#     "config_path,config_name",
#     [
#         ("rslrl_isaaclab/conf/env", "isaaclab_mimic"),
#     ],
# )
# def test_hydra_entrypoint(config_path: str, config_name: str, isaac_cleanup_registry):
#     """Test hydra entrypoint for building env group from config."""
#     # set parent dir for examples
#     EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"

#     @hydra.main()
#     def entrypoint(cfg: DictConfig):
#         print(f"--- Complete Config ---\n{cfg}")
#         env_cfg = EnvConfig.from_omegaconf(cfg)
#         env_group = build_env_group([env_cfg])
#         isaac_cleanup_registry.append(env_group)

#     with initialize_config_dir(config_dir=str(EXAMPLES_DIR / config_path)):
#         cfg = compose(config_name=config_name)
#         entrypoint(cfg)


@pytest.mark.parametrize("as_remote", [True, False])
@pytest.mark.parametrize("task", ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5"])
def test_mujoco_make_env_group(task, as_remote):
    require_env_backend("mujoco")
    env_cfgs = [
        _make_env_config(env_backend="mujoco", task_name=task, num_workers=2),
    ]
    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_ENV == as_remote
    env_group = build_env_group(env_cfgs)

    assert len(env_group) == 2


# FIXME: it can pass individually but fails in `make test-unit`
# @pytest.mark.parametrize("task", ["Tracking-Flat-G1-v0"])
# @pytest.mark.parametrize("num_envs", [1])
# def test_isaaclab_manager_based_env_group(task: str, num_envs: int, isaac_cleanup_registry):
#     task = "Tracking-Flat-G1-v0"

#     max_episode_steps = 10

#     config = EnvConfig(
#         name=task,
#         task=task,
#         num_envs=num_envs,
#         backend="isaac_manager_based",
#         max_episode_steps=max_episode_steps,
#         env_kwargs=dict(
#             device=f"cuda:{torch.cuda.current_device()}",
#             headless=True,
#             commands=dict(motion={"motion_file": "tests/data/fallAndGetUp1_subject1.npz"}),
#         ),
#     )

#     env_cfgs = [config]

#     # env_cfgs = [
#     #     _make_env_config(
#     #         env_backend="isaac_manager_based",
#     #         task=task,
#     #         num_envs=num_envs,
#     #     )
#     # ]

#     env_group = build_env_group(env_cfgs)
#     assert len(env_group) == 1

#     isaac_cleanup_registry.append(env_group)


@pytest.mark.parametrize("as_remote", [True, False])
def test_env_group_shape_check_fail_case(as_remote):
    """
    Test that the environment group raises a ValueError when environments have mismatched
    observation or action space shapes.
    """
    require_env_backend("mujoco")
    env_cfgs = [
        _make_env_config(env_backend="mujoco", task_name="Ant-v5", num_workers=1),
        _make_env_config(env_backend="mujoco", task_name="Hopper-v5", num_workers=1),
    ]
    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_ENV == as_remote

    env_group = build_env_group(env_cfgs)
    with pytest.raises(ValueError):
        env_group.init()


@pytest.mark.parametrize("as_remote", [True, False])
def test_env_group_shape_check_success_case(as_remote):
    require_env_backend("mujoco")
    env_cfgs = [
        _make_env_config(env_backend="mujoco", task_name="Ant-v5", num_workers=1),
        _make_env_config(env_backend="mujoco", task_name="Ant-v5", num_workers=1),
    ]

    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_ENV == as_remote

    env_group = build_env_group(env_cfgs)
    env_group.init()


@pytest.mark.parametrize("as_remote", [True, False])
@pytest.mark.parametrize("num_env_workers", [1, 2, 4])
def test_env_sync_step(as_remote, num_env_workers):
    require_env_backend("mujoco")
    env_cfgs = [
        _make_env_config(env_backend="mujoco", task_name="Ant-v5", num_workers=num_env_workers),
    ]

    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_ENV == as_remote

    env_group = build_env_group(env_cfgs)
    env_meta_list = env_group.init()

    batched_env_ret, _ = env_group.reset()
    assert isinstance(batched_env_ret, BatchedData)
    assert len(batched_env_ret.ids()) == num_env_workers
    for i, (env_id, env_ret) in enumerate(batched_env_ret.items()):
        assert (
            resolve_object(env_ret).observation.shape
            == resolve_object(env_meta_list[i]).observation_space.shape
        )

    for _ in range(5):
        policy_resp_list = [
            _mock_policy_ret(resolve_object(env_meta).action_space, env_id)
            for env_meta, env_id in zip(env_meta_list, batched_env_ret.ids())
        ]
        batched_policy_resp = BatchedData(batched_env_ret.ids(), policy_resp_list)
        batched_env_ret, _ = env_group.step(batched_policy_resp)
        assert isinstance(batched_env_ret, BatchedData)
        assert len(batched_env_ret.ids()) == num_env_workers


@pytest.mark.parametrize("as_remote", [True, False])
@pytest.mark.parametrize("num_env_workers", [1, 2, 4])
def test_env_step_counter(as_remote, num_env_workers):
    require_env_backend("mujoco")
    env_cfgs = [
        _make_env_config(env_backend="mujoco", task_name="Ant-v5", num_workers=num_env_workers),
    ]

    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_ENV == as_remote

    env_group = build_env_group(env_cfgs)
    env_meta_list = env_group.init()

    max_episode_steps = 2
    with env_group.auto_reset(max_episode_steps=max_episode_steps):
        # before reset
        for env_id in env_group.env_ids:
            assert env_group.step_counter.get_steps(env_id) == -1
            assert not env_group.step_counter.is_reached_max_steps(env_id)

        batched_env_ret, truncations = env_group.reset()
        # after reset
        for env_id in env_group.env_ids:
            assert env_group.step_counter.get_steps(env_id) == 0
            assert not env_group.step_counter.is_reached_max_steps(env_id)
            for truncated in truncations:
                assert not truncated

        policy_resp_list = [
            _mock_policy_ret(resolve_object(env_meta).action_space, env_id)
            for env_meta, env_id in zip(env_meta_list, batched_env_ret.ids())
        ]
        batched_policy_resp = BatchedData(batched_env_ret.ids(), policy_resp_list)

        batched_env_ret, truncations = env_group.step(batched_policy_resp)
        # after 1st step
        for env_id in env_group.env_ids:
            assert env_group.step_counter.get_steps(env_id) == 1
            assert not env_group.step_counter.is_reached_max_steps(env_id)
            for truncated in truncations:
                assert not truncated

        batched_env_ret, truncations = env_group.step(batched_policy_resp)
        # after 2nd step
        for env_id in env_group.env_ids:
            assert env_group.step_counter.get_steps(env_id) == max_episode_steps
            assert env_group.step_counter.is_reached_max_steps(env_id)
            for truncated in truncations:
                assert truncated

        batched_env_ret, truncations = env_group.step(batched_policy_resp)
        # after 3nd step (should auto reset)
        for env_id in env_group.env_ids:
            assert env_group.step_counter.get_steps(env_id) == 0
            assert not env_group.step_counter.is_reached_max_steps(env_id)
            for truncated in truncations:
                assert not truncated


@pytest.mark.parametrize("as_remote", [True, False])
@pytest.mark.parametrize("num_env_workers", [1, 2, 4])
def test_auto_reset_sync(as_remote, num_env_workers):
    require_env_backend("mujoco")
    env_cfgs = [
        _make_env_config(env_backend="mujoco", task_name="Ant-v5", num_workers=num_env_workers),
    ]

    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_ENV == as_remote

    env_group = build_env_group(env_cfgs)
    env_meta_list = env_group.init()

    max_episode_steps = 5
    with env_group.auto_reset(max_episode_steps=max_episode_steps):
        # test reset
        batched_env_ret, truncations = env_group.reset()
        for truncated in truncations:
            assert not truncated
        for env_id in env_group.env_ids:
            assert env_group.step_counter[env_id] == 0

        # test step
        for step in range(1, 20):
            policy_resp_list = [
                _mock_policy_ret(resolve_object(env_meta).action_space, env_id)
                for env_meta, env_id in zip(env_meta_list, batched_env_ret.ids())
            ]
            batched_policy_resp = BatchedData(batched_env_ret.ids(), policy_resp_list)
            batched_env_ret, truncations = env_group.step(batched_policy_resp)

            expected_step = step % (max_episode_steps + 1)

            for env_id in env_group.env_ids:
                assert env_group.step_counter[env_id] == expected_step
                for truncated in truncations:
                    if expected_step == max_episode_steps:
                        assert truncated
                    else:
                        assert not truncated
