from types import SimpleNamespace
import sys
import types

import torch
import torch.nn as nn

from rlightning.policy.base_policy import PolicyRole
from rlightning.policy.rsl_rl_policy import RSLRLPolicy, RSLRLVecEnvMeta
from rlightning.utils.config import Config, PolicyConfig, WeightBufferConfig


def _make_env_ret(observation):
    return SimpleNamespace(env_id="env-0", observation=observation, info={})


def _sync_and_compare(train_policy, eval_policy, env_ret):
    train_action = train_policy.rollout_step(env_ret).action
    eval_action = eval_policy.rollout_step(env_ret).action
    assert not torch.allclose(train_action, eval_action)

    eval_policy.load_state_dict(train_policy.get_trainable_parameters())
    synced_action = eval_policy.rollout_step(env_ret).action
    torch.testing.assert_close(synced_action, train_action)


def test_param_sync_rsl_rl_policy(monkeypatch):
    class FakeAgent:
        def __init__(self, env_meta, **kwargs):
            self.actor = nn.Linear(3, 2, bias=False)
            self.initialized = True

        def to(self, device):
            self.actor.to(device)
            return self

        def draw_actions(self, obs, info):
            return self.actor(obs), {"data": torch.zeros(obs.shape[0], device=obs.device)}

        def draw_random_actions(self, obs, info):
            return torch.zeros((obs.shape[0], 2), device=obs.device), {
                "data": torch.zeros(obs.shape[0], device=obs.device)
            }

    fake_algorithms = types.SimpleNamespace(PPO=FakeAgent)
    fake_rsl_rl = types.SimpleNamespace(algorithms=fake_algorithms)
    monkeypatch.setitem(sys.modules, "rsl_rl", fake_rsl_rl)

    policy_cfg = PolicyConfig(
        type="RSLRLPolicy",
        weight_buffer=WeightBufferConfig(type="WeightBuffer", buffer_strategy="Double"),
        policy_kwargs=Config.from_dict({"algorithm": "PPO"}),
    )
    train_policy = RSLRLPolicy(policy_cfg, PolicyRole.TRAIN)
    eval_policy = RSLRLPolicy(policy_cfg, PolicyRole.EVAL)

    env_meta = RSLRLVecEnvMeta(num_envs=1, num_actions=2, get_observations=None)
    train_policy.construct_network(env_meta=env_meta)
    eval_policy.construct_network(env_meta=env_meta)

    with torch.no_grad():
        train_policy.algo.actor.weight.fill_(1.0)
        eval_policy.algo.actor.weight.zero_()

    observation = torch.ones(1, 3, device=train_policy.device)
    _sync_and_compare(train_policy, eval_policy, _make_env_ret(observation))
