import torch
import pytest

from rlightning.buffer.utils.utils import (
    default_compute_gae,
    default_env_ret_preprocess_fn,
    default_postprocess_fn,
    default_policy_resp_preprocess_fn,
    default_preprocess_fn,
)
from rlightning.types import EnvRet, PolicyResponse


def test_default_env_ret_preprocess_fn_applies_obs_and_reward_preprocessors():
    env_ret = EnvRet(
        env_id="env-1",
        observation=[1.0, 2.0],
        last_reward=3.0,
        last_terminated=False,
        last_truncated=False,
        info={"k": 1},
    )

    def obs_preprocessor(x):
        return torch.tensor(x) + 1

    def reward_preprocessor(x):
        return torch.tensor(x) * 2

    transition = default_env_ret_preprocess_fn({}, env_ret, obs_preprocessor, reward_preprocessor)

    expected_obs = torch.tensor([2.0, 3.0], dtype=transition["observation"].dtype)
    expected_reward = torch.tensor(6.0, dtype=transition["last_reward"].dtype)

    assert torch.allclose(transition["observation"], expected_obs)
    assert torch.allclose(transition["last_reward"], expected_reward)
    assert transition["last_terminated"] is False
    assert transition["last_truncated"] is False
    assert transition["info"] == {"k": 1}


def test_default_env_ret_preprocess_fn_rejects_non_env_return_input():
    with pytest.raises(TypeError):
        default_env_ret_preprocess_fn({}, "not-env-ret", lambda x: x, lambda x: x)


def test_default_policy_resp_preprocess_fn_merges_policy_fields():
    transition = default_policy_resp_preprocess_fn({}, PolicyResponse(env_id="env-1", action=torch.tensor(1), log_prob=0.5))

    assert torch.equal(transition["action"], torch.tensor(1))
    assert transition["log_prob"] == 0.5


def test_default_policy_resp_preprocess_fn_rejects_non_policy_response_input():
    with pytest.raises(TypeError):
        default_policy_resp_preprocess_fn({}, "not-policy")


def test_default_preprocess_fn_requires_at_least_one_input():
    with pytest.raises(ValueError, match="At least one of env_ret or policy_resp"):
        default_preprocess_fn({})


def test_default_preprocess_fn_rejects_mismatched_env_ids():
    env_ret = EnvRet(env_id="env-0", observation=[1, 2], last_reward=1.0)
    policy_resp = PolicyResponse(env_id="env-1", action=0)

    with pytest.raises(ValueError, match="Mismatched env_id"):
        default_preprocess_fn({}, env_ret=env_ret, policy_resp=policy_resp)


def test_default_preprocess_fn_merges_env_return_and_policy_response():
    env_ret = EnvRet(
        env_id="env-0",
        observation={"state": [1.0, 2.0]},
        last_reward=1.5,
        last_terminated=False,
        last_truncated=False,
        info={"episode": 3},
    )
    policy_resp = PolicyResponse(env_id="env-0", action=2, log_prob=0.3)

    transition = default_preprocess_fn({}, env_ret=env_ret, policy_resp=policy_resp)

    assert transition["observation"] == {"state": [1.0, 2.0]}
    assert transition["last_reward"] == 1.5
    assert transition["info"] == {"episode": 3}
    assert transition["action"] == 2
    assert transition["log_prob"] == 0.3


def test_default_postprocess_fn_builds_training_batch_from_episode_buffer():
    episode_buffer = {
        "observation": [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([5.0, 6.0]),
        ],
        "last_reward": [0.0, 1.0, 2.0],
        "last_terminated": [False, False, True],
        "last_truncated": [False, False, False],
        "info": [{"step": 0}, {"step": 1}, {"step": 2}],
        "action": [
            torch.tensor([0.1]),
            torch.tensor([0.2]),
            torch.tensor([0.3]),
        ],
        "log_prob": [torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3)],
    }

    data = default_postprocess_fn(episode_buffer)

    assert "info" not in data
    assert torch.equal(data["observation"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(data["next_observation"], torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    assert torch.equal(data["reward"], torch.tensor([1.0, 2.0]))
    assert torch.equal(data["terminated"], torch.tensor([False, True]))
    assert torch.equal(data["truncated"], torch.tensor([False, False]))
    assert torch.equal(data["action"], torch.tensor([[0.1], [0.2]]))
    assert torch.allclose(data["log_prob"], torch.tensor([0.1, 0.2]))


def test_default_compute_gae_respects_terminal_boundaries():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    next_values = torch.tensor([[0.5], [0.5], [0.0]])
    dones = torch.tensor([[0.0], [0.0], [1.0]])

    advantages, returns = default_compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        dones=dones,
        gamma=0.9,
        lam=0.95,
        normalize_adv=False,
    )

    expected_advantages = torch.tensor([[4.4448], [4.0875], [2.5000]])
    expected_returns = torch.tensor([[4.9448], [4.5875], [3.0000]])

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)
    assert torch.allclose(returns, expected_returns, atol=1e-4)


def test_default_compute_gae_normalizes_advantages_when_requested():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    next_values = torch.tensor([[0.5], [0.5], [0.5]])
    dones = torch.tensor([[0.0], [0.0], [0.0]])

    advantages, _ = default_compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        dones=dones,
        gamma=0.9,
        lam=0.95,
        normalize_adv=True,
    )

    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages.std(), torch.tensor(1.0), atol=1e-6)
