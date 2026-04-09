"""
Integration tests for episode-unit storage in ReplayBuffer.
under episode-unit mode, a full processing flow of storing
and sampling of replaybuffer.
"""

import gymnasium as gym
import pytest
import torch

from rlightning.buffer.replay_buffer import ReplayBuffer
from rlightning.types import BatchedData, EnvRet, PolicyResponse
from rlightning.types.metadata import EnvMeta
from rlightning.utils.config import BufferConfig

from tensordict import TensorDict

pytestmark = pytest.mark.integration


def _make_env_meta(env_id: str) -> EnvMeta:
    return EnvMeta(
        env_id=env_id,
        action_space=gym.spaces.Discrete(2),
        observation_space=gym.spaces.Box(low=0, high=1, shape=(1,), dtype=float),
        num_envs=1,
    )


def _make_buffer_config():
    return BufferConfig.from_dict(
        {
            "type": "ReplayBuffer",
            "capacity": 10,
            "sampler": {"type": "uniform"},
            "storage": {"type": "unified", "device": "cpu", "unit": "episode"},
        }
    )


def test_episode_unit_storage_flow():
    tensordict = pytest.importorskip("tensordict")

    env_id = "env-episode"
    buffer = ReplayBuffer(config=_make_buffer_config())
    buffer.init([_make_env_meta(env_id)], [env_id])

    # Episode 1: 3 steps -> stored episode length 2 after postprocess
    for step in range(3):
        env_ret = EnvRet(
            env_id=env_id,
            observation=float(step),
            last_reward=float(step),
            last_terminated=step == 2,
            last_truncated=False,
            info={},
        )
        policy_resp = PolicyResponse(env_id=env_id, action=float(step))
        buffer.add_batched_transition(
            BatchedData([env_id], [env_ret]),
            BatchedData([env_id], [policy_resp]),
        )
    buffer.truncate_episodes([env_id])

    # Episode 2: same length
    for step in range(3):
        env_ret = EnvRet(
            env_id=env_id,
            observation=float(step + 10),
            last_reward=float(step),
            last_terminated=step == 2,
            last_truncated=False,
            info={},
        )
        policy_resp = PolicyResponse(env_id=env_id, action=float(step))
        buffer.add_batched_transition(
            BatchedData([env_id], [env_ret]),
            BatchedData([env_id], [policy_resp]),
        )
    buffer.truncate_episodes([env_id])

    assert len(buffer) == 2

    sample_data = buffer.sample(batch_size=2)
    assert isinstance(sample_data[0], dict)
    sample_data = [TensorDict.from_dict(data, auto_batch_size=True) for data in sample_data]
    assert sample_data[0].batch_size == torch.Size([2, 2])

    # Content validation: verify sampled data matches written episodes
    sampled_obs = sample_data[0]["observation"]  # shape: [2, 2] (batch=2, time=2)
    sampled_action = sample_data[0]["action"]  # shape: [2, 2]

    # Episode 1: obs=[0.0, 1.0], action=[0.0, 1.0] (after postprocess, length 3->2)
    # Episode 2: obs=[10.0, 11.0], action=[0.0, 1.0]
    # Each sampled episode should match one of these patterns
    valid_first_obs = {0.0, 10.0}
    for i in range(2):
        first_obs = sampled_obs[i, 0].item()
        assert first_obs in valid_first_obs, f"Unexpected first obs: {first_obs}"

        # Verify temporal consistency within episode
        if first_obs == 0.0:
            assert sampled_obs[i, 1].item() == 1.0, "Episode 1 obs sequence mismatch"
            assert sampled_action[i, 0].item() == 0.0, "Episode 1 action mismatch"
            assert sampled_action[i, 1].item() == 1.0, "Episode 1 action mismatch"
        else:
            assert sampled_obs[i, 1].item() == 11.0, "Episode 2 obs sequence mismatch"
            assert sampled_action[i, 0].item() == 0.0, "Episode 2 action mismatch"
            assert sampled_action[i, 1].item() == 1.0, "Episode 2 action mismatch"
