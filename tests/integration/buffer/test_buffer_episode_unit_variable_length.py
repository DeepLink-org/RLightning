"""
Integration test for episode-unit storage with variable episode lengths.
"""

import gymnasium as gym
import pytest

from rlightning.buffer.replay_buffer import ReplayBuffer
from rlightning.types import BatchedData, EnvRet, PolicyResponse
from rlightning.types.metadata import EnvMeta
from rlightning.utils.config import BufferConfig

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
            "capacity": 20,
            "sampler": {"type": "all"},
            "storage": {"type": "unified", "device": "cpu", "unit": "episode"},
        }
    )


def _add_episode(buffer, env_id: str, steps: int):
    for step in range(steps):
        env_ret = EnvRet(
            env_id=env_id,
            observation=float(step),
            last_reward=float(step),
            last_terminated=step == (steps - 1),
            last_truncated=False,
            info={},
        )
        policy_resp = PolicyResponse(env_id=env_id, action=float(step))
        buffer.add_batched_transition(
            BatchedData([env_id], [env_ret]),
            BatchedData([env_id], [policy_resp]),
        )
    buffer.truncate_episodes([env_id])


def test_episode_unit_variable_length_returns_list():
    tensordict = pytest.importorskip("tensordict")

    env_id = "env-episode-varlen"
    buffer = ReplayBuffer(config=_make_buffer_config())
    buffer.init([_make_env_meta(env_id)], [env_id])

    # Episode lengths 3 and 4 steps -> stored transition lengths 2 and 3
    _add_episode(buffer, env_id, steps=3)
    _add_episode(buffer, env_id, steps=4)

    assert len(buffer) == 2

    sample_data = buffer.sample(batch_size=None)
    assert len(sample_data) == 1

    # Variable lengths should produce a list of TensorDicts
    assert isinstance(sample_data[0], list)
    assert len(sample_data[0]) == 2
    assert all(isinstance(item, tensordict.TensorDict) for item in sample_data[0])

    lengths = [len(item) for item in sample_data[0]]
    assert sorted(lengths) == [2, 3]
