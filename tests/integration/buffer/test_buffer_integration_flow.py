"""
Integration tests for buffer components working together.
"""

import gymnasium as gym
import pytest

from rlightning.buffer.replay_buffer import ReplayBuffer
from rlightning.types import BatchedData, EnvRet, PolicyResponse
from rlightning.types.metadata import EnvMeta
from rlightning.utils.config import BufferConfig

pytestmark = pytest.mark.integration


def _make_buffer_config(auto_truncate=False):
    return BufferConfig.from_dict(
        {
            "type": "ReplayBuffer",
            "capacity": 8,
            "auto_truncate_episode": auto_truncate,
            "sampler": {"type": "uniform"},
            "storage": {"type": "unified", "device": "cpu", "unit": "transition"},
        }
    )


def _make_env_meta(env_id: str) -> EnvMeta:
    return EnvMeta(
        env_id=env_id,
        action_space=gym.spaces.Discrete(2),
        observation_space=gym.spaces.Box(low=0, high=1, shape=(1,), dtype=float),
        num_envs=1,
    )


def test_replay_buffer_transition_flow():
    env_id = "env-1"
    buffer = ReplayBuffer(config=_make_buffer_config())
    buffer.init([_make_env_meta(env_id)], [env_id])

    for step in range(2):
        env_ret = EnvRet(
            env_id=env_id,
            observation=float(step),
            last_reward=float(step),
            last_terminated=step == 1,
            last_truncated=False,
            info={},
        )
        policy_resp = PolicyResponse(env_id=env_id, action=float(step))
        buffer.add_batched_transition(
            BatchedData([env_id], [env_ret]),
            BatchedData([env_id], [policy_resp]),
        )

    buffer.truncate_episodes([env_id])

    # Precise length check: 2 steps -> 1 transition after postprocess
    assert len(buffer) == 1

    sample_data = buffer.sample(batch_size=1)
    assert len(sample_data) == 1
    assert isinstance(sample_data[0], dict)

    # Keys validation
    keys = set(sample_data[0].keys())
    assert "observation" in keys
    assert "next_observation" in keys
    assert "action" in keys
    assert "reward" in keys

    # Content validation: verify sampled data matches written values
    sample = sample_data[0]
    # Step 0: obs=0.0, action=0.0 -> Step 1: obs=1.0, reward=1.0
    assert sample["observation"].item() == 0.0, "observation should be from step 0"
    assert sample["next_observation"].item() == 1.0, "next_observation should be from step 1"
    assert sample["action"].item() == 0.0, "action should be from step 0"
    assert sample["reward"].item() == 1.0, "reward should be from step 1"


def test_replay_buffer_async_data_flow():
    env_id = "env-async"
    buffer = ReplayBuffer(config=_make_buffer_config(auto_truncate=True))
    buffer.init([_make_env_meta(env_id)], [env_id])

    env_ret_0 = EnvRet(
        env_id=env_id,
        observation=0.0,
        last_reward=1.0,
        last_terminated=False,
        last_truncated=False,
        info={},
    )
    policy_resp_0 = PolicyResponse(env_id=env_id, action=0.0)

    env_ret_1 = EnvRet(
        env_id=env_id,
        observation=1.0,
        last_reward=2.0,
        last_terminated=True,
        last_truncated=False,
        info={},
    )
    policy_resp_1 = PolicyResponse(env_id=env_id, action=1.0)

    buffer.add_batched_data_async(BatchedData([env_id], [env_ret_0]))
    buffer.add_batched_data_async(BatchedData([env_id], [policy_resp_0]))
    buffer.add_batched_data_async(BatchedData([env_id], [env_ret_1]))
    buffer.add_batched_data_async(BatchedData([env_id], [policy_resp_1]))

    # Precise length check: 2 steps -> 1 transition after postprocess
    assert len(buffer) == 1

    sample_data = buffer.sample(batch_size=1)
    assert isinstance(sample_data[0], dict)

    # Keys validation (consistent with test_replay_buffer_transition_flow)
    keys = set(sample_data[0].keys())
    assert "observation" in keys
    assert "next_observation" in keys
    assert "action" in keys
    assert "reward" in keys

    # Content validation: verify sampled data matches written values
    sample = sample_data[0]
    # env_ret_0: obs=0.0, reward=1.0, action=0.0
    # env_ret_1: obs=1.0, reward=2.0, action=1.0, terminated=True
    assert sample["observation"].item() == 0.0, "observation should be from env_ret_0"
    assert sample["next_observation"].item() == 1.0, "next_observation should be from env_ret_1"
    assert sample["action"].item() == 0.0, "action should be from policy_resp_0"
    assert sample["reward"].item() == 2.0, "reward should be from env_ret_1"
