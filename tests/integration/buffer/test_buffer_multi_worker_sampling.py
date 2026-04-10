"""
Integration test for buffer sampling across multiple train workers.
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
            "capacity": 16,
            "sampler": {"type": "uniform"},
            "storage": {"type": "unified", "device": "cpu", "unit": "transition"},
        }
    )


def test_multi_worker_sampling_split():
    tensordict = pytest.importorskip("tensordict")

    env_id = "env-multi-worker"
    buffer = ReplayBuffer(config=_make_buffer_config())
    buffer.init([_make_env_meta(env_id)], [env_id])

    # Two episodes with 4 transitions each -> total 8 transitions
    for episode_idx in range(2):
        for step in range(5):
            env_ret = EnvRet(
                env_id=env_id,
                observation=float(step + episode_idx * 10),
                last_reward=float(step),
                last_terminated=step == 4,
                last_truncated=False,
                info={},
            )
            policy_resp = PolicyResponse(env_id=env_id, action=float(step))
            buffer.add_batched_transition(
                BatchedData([env_id], [env_ret]),
                BatchedData([env_id], [policy_resp]),
            )
        buffer.truncate_episodes([env_id])

    assert len(buffer) == 8

    # Simulate two train workers bound to the same storage.
    buffer.table._storage_to_train_workers = {0: [0, 1]}

    sample_data = buffer.sample(batch_size=6, shuffle=False, drop_last=True)

    assert len(sample_data) == 2
    assert isinstance(sample_data[0], dict)
    assert isinstance(sample_data[1], dict)

    # Each worker should receive half the samples (6 total -> 3 each).
    assert len(next(iter(sample_data[0].values()))) == 3
    assert len(next(iter(sample_data[1].values()))) == 3

    # Ensure workers do not receive identical data sets.
    obs_worker0 = sample_data[0]["observation"].tolist()
    obs_worker1 = sample_data[1]["observation"].tolist()
    assert obs_worker0 != obs_worker1

    # Content validation: all sampled obs should come from written data
    # Episode 0: obs in [0,1,2,3], Episode 1: obs in [10,11,12,13]
    valid_obs = {0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0}
    all_sampled_obs = obs_worker0 + obs_worker1
    for obs in all_sampled_obs:
        assert obs in valid_obs, f"Unexpected observation {obs}"

    # Verify next_observation corresponds to obs + 1 (within same episode)
    for worker_data in sample_data:
        obs_list = worker_data["observation"].tolist()
        next_obs_list = worker_data["next_observation"].tolist()
        for obs, next_obs in zip(obs_list, next_obs_list):
            # obs and next_obs should differ by 1 (same episode continuity)
            assert next_obs == obs + 1.0, f"next_obs {next_obs} should be obs {obs} + 1"
