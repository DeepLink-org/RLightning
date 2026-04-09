"""
Deeper integration tests for buffer sampling and stats.
"""

import gymnasium as gym
import pytest
from tensordict import TensorDict

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


def _make_buffer_config(sampler_type: str = "uniform"):
    return BufferConfig.from_dict(
        {
            "type": "ReplayBuffer",
            "capacity": 16,
            "sampler": {"type": sampler_type},
            "storage": {"type": "unified", "device": "cpu", "unit": "transition"},
        }
    )


def test_replay_buffer_multi_env_sampling():
    env_ids = ["env-1", "env-2"]
    buffer = ReplayBuffer(config=_make_buffer_config())
    buffer.init([_make_env_meta(env_id) for env_id in env_ids], env_ids)

    # Two steps for each env (episode length=2 -> 1 transition per episode)
    for step in range(2):
        env_rets = []
        policy_resps = []
        for env_id in env_ids:
            env_rets.append(
                EnvRet(
                    env_id=env_id,
                    observation=float(step),
                    last_reward=float(step),
                    last_terminated=step == 1,
                    last_truncated=False,
                    info={},
                )
            )
            policy_resps.append(PolicyResponse(env_id=env_id, action=float(step)))
        buffer.add_batched_transition(
            BatchedData(env_ids, env_rets),
            BatchedData(env_ids, policy_resps),
        )

    buffer.truncate_episodes(env_ids)

    # 2 envs × 1 transition each = 2 transitions total
    assert len(buffer) == 2
    sample_data = buffer.sample(batch_size=2)

    assert len(sample_data) == 1
    assert isinstance(sample_data[0], dict)
    assert len(TensorDict.from_dict(sample_data[0], auto_batch_size=True)) == 2

    # Content validation: verify sampled data matches written values
    # Both envs wrote obs=0.0 at step 0, next_obs=1.0 at step 1
    for i in range(2):
        assert sample_data[0]["observation"][i].item() == 0.0, "observation should be 0.0"
        assert sample_data[0]["next_observation"][i].item() == 1.0, "next_observation should be 1.0"
        assert sample_data[0]["action"][i].item() == 0.0, "action should be 0.0"
        assert sample_data[0]["reward"][i].item() == 1.0, "reward should be 1.0"


def test_replay_buffer_all_sampler_returns_all_data():
    env_id = "env-all"
    buffer = ReplayBuffer(config=_make_buffer_config(sampler_type="all"))
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

    # all sampler ignores batch_size and returns everything
    sample_data = buffer.sample(batch_size=None)

    assert len(sample_data) == 1
    assert isinstance(sample_data[0], dict)
    assert len(TensorDict.from_dict(sample_data[0], auto_batch_size=True)) == len(buffer)

    # Content validation: verify sampled data matches written values
    # 2 steps -> 1 transition: obs=0.0, next_obs=1.0, action=0.0, reward=1.0
    assert sample_data[0]["observation"].item() == 0.0, "observation should be 0.0"
    assert sample_data[0]["next_observation"].item() == 1.0, "next_observation should be 1.0"
    assert sample_data[0]["action"].item() == 0.0, "action should be 0.0"
    assert sample_data[0]["reward"].item() == 1.0, "reward should be 1.0"


def test_replay_buffer_batch_sampler_no_replacement():
    tensordict = pytest.importorskip("tensordict")

    env_id = "env-batch"
    buffer = ReplayBuffer(config=_make_buffer_config(sampler_type="batch"))
    buffer.init([_make_env_meta(env_id)], [env_id])

    # Episode length=5 -> 4 transitions stored after truncate.
    for step in range(5):
        env_ret = EnvRet(
            env_id=env_id,
            observation=float(step),
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

    # 5 steps -> 4 transitions after postprocess
    assert len(buffer) == 4

    sample_data = buffer.sample(batch_size=3)
    assert isinstance(sample_data[0], dict)

    # Batch sampler should return unique samples (no replacement)
    sample_obs = sample_data[0]["observation"]
    assert len(set(sample_obs)) == len(sample_obs), "batch sampler should not have duplicates"

    # Content validation: sampled obs should be from written values [0.0, 1.0, 2.0, 3.0]
    valid_obs = {0.0, 1.0, 2.0, 3.0}
    for obs in sample_obs:
        assert obs.item() in valid_obs, f"unexpected observation {obs}"

    # Requesting more samples than buffer size should raise ValueError
    with pytest.raises(ValueError):
        buffer.sample(batch_size=5)
