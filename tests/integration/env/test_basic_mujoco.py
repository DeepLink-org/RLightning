"""This file implements test cases for basic mujoco tasks, which are integrated in gymnasium"""

import os

import pytest
import ray

pytest.importorskip("mujoco")

from rlightning.env.mujoco_env import MujocoEnv
from rlightning.types import BatchedData, PolicyResponse
from rlightning.utils.builders import build_env_group
from rlightning.utils.config import EnvConfig
from rlightning.utils.utils import InternalFlag

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _auto_ray_cluster():
    # Ensure a Ray cluster is available for this module's tests
    from tests.test_utils import setup_ray_cluster, teardown_ray_cluster

    setup_ray_cluster(num_gpus=1)
    yield
    teardown_ray_cluster()


@pytest.mark.parametrize("task", ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5"])
def test_env_logic(task: str):

    max_episode_steps = 10

    config = EnvConfig(name=task, task=task, backend="mujoco", max_episode_steps=max_episode_steps)

    env = MujocoEnv(config)

    _ = env.reset()

    terminated = truncated = False
    step_cnt = 0

    while not (terminated or truncated):

        action = env.action_space.sample()
        policy_resp = PolicyResponse(env_id=env.env_id, action=action)
        env_ret = env.step(policy_resp)
        observation, reward, terminated, truncated, info = (
            env_ret.observation,
            env_ret.last_reward,
            env_ret.last_terminated,
            env_ret.last_truncated,
            env_ret.info,
        )

        step_cnt += 1
        if not terminated and step_cnt >= max_episode_steps:
            assert truncated, (step_cnt, max_episode_steps, terminated, truncated)

    assert step_cnt <= max_episode_steps, (
        step_cnt,
        max_episode_steps,
        terminated,
        truncated,
    )


@pytest.mark.parametrize("task", ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5"])
@pytest.mark.parametrize("env_as_remote", [True, False])
def test_feed_into_env_client(env_as_remote: bool, task: str):

    max_episode_steps = 100

    env_cfgs = [
        EnvConfig(
            name=task,
            backend="mujoco",
            task=task,
            num_workers=1,
            num_envs=1,
            max_episode_steps=max_episode_steps,
            env_parameters=dict(),
        )
    ]
    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if env_as_remote else "0"
    assert InternalFlag.REMOTE_ENV == env_as_remote

    env_group = build_env_group(env_cfgs)

    assert len(env_group) == 1

    batched_env_ret, truncations = env_group.reset()

    assert len(batched_env_ret) == 1

    if env_as_remote:
        env_id, env_ret_future = list(batched_env_ret.items())[0]
        assert isinstance(env_ret_future, ray.ObjectRef), type(env_ret_future)
        env_ret = ray.get(env_ret_future)
    else:
        env_id, env_ret = list(batched_env_ret.items())[0]

    terminated = truncated = False
    step_cnt = 0

    while not (terminated or truncated):

        action_spaces = env_group.get_action_spaces()
        if env_as_remote:
            actions = [ray.get(e).sample() for e in action_spaces]
        else:
            actions = [e.sample() for e in action_spaces]

        policy_resp_list = [
            PolicyResponse(env_id=env_group.env_ids[i], action=action)
            for i, action in enumerate(actions)
        ]

        assert len(policy_resp_list) == 1, len(actions)
        policy_rets = BatchedData(
            ids=[env_id],
            data=policy_resp_list,
        )
        env_rets, truncations = env_group.step(policy_rets)

        if env_as_remote:
            assert len(env_rets) == 1
            env_ret_future = env_rets.values()[0]
            env_ret = ray.get(env_ret_future)
        else:
            env_ret = env_rets.values()[0]
            # observation, reward, terminated, truncated, info = env_rets[0]

        terminated, truncated = env_ret.last_terminated, env_ret.last_truncated

        step_cnt += 1
        if not terminated and step_cnt >= max_episode_steps:
            assert truncated, (step_cnt, max_episode_steps, terminated, truncated)

    assert step_cnt <= max_episode_steps, (
        step_cnt,
        max_episode_steps,
        terminated,
        truncated,
    )
