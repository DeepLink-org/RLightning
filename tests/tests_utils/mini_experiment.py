"""Test-only components for running tiny end-to-end training experiments."""

from __future__ import annotations

import datetime
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn
from torch import distributed as dist
from types import SimpleNamespace

from rlightning.env.base_env import BaseEnv
from rlightning.policy.base_policy import BasePolicy
from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.distributed.comm_context import CommContext
from rlightning.utils.distributed.group_initializer import CommMode
from rlightning.utils.registry import ENVS, POLICIES
from rlightning.utils.ray.remote_class import RayActorMixin
from rlightning.utils.utils import to_device


if not torch.cuda.is_available():
    # BasePolicy currently calls `.cuda()` in rollout/postprocess hooks even on CPU-only
    # machines. Patch the data containers in this test-only module so subprocess smoke
    # tests can exercise the full engine stack without requiring GPUs.
    EnvRet.cuda = lambda self, device="cuda": self  # type: ignore[method-assign]
    PolicyResponse.cuda = lambda self, device="cuda": self  # type: ignore[method-assign]
    RayActorMixin._get_gpu_ids = lambda self: [0]  # type: ignore[method-assign]

    def _cpu_safe_init_distributed_env(
        self,
        world_size: int | None = None,
        rank: int | None = None,
        backend: str = "gloo",
        dist_url: str = "env://",
        timeout: int = 1800,
    ) -> None:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(0, timeout),
        )
        dist.barrier()

        ranks = list(range(world_size))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._register_group(local_rank, world_size, dist.GroupMember.WORLD, ranks, CommMode.GLOBAL)
        self._global_ranks[CommMode.GLOBAL] = rank

    CommContext.init_distributed_env = _cpu_safe_init_distributed_env  # type: ignore[method-assign]


@ENVS.register("mini_counter")
class MiniCounterEnv(BaseEnv):
    """A tiny deterministic env whose target action is encoded in the observation."""

    def __init__(self, config, worker_index=0, preprocess_fn=None):
        super().__init__(config=config, worker_index=worker_index, preprocess_fn=preprocess_fn)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.env = SimpleNamespace(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        self.horizon = int(getattr(config, "max_episode_steps", None) or 4)
        self._step_count = 0
        self._episode_return = 0.0
        self._obj_set = "train"

    def _target(self) -> np.ndarray:
        if self._obj_set == "test":
            return np.array([-0.5, 0.25], dtype=np.float32)
        return np.array([0.25, -0.25], dtype=np.float32)

    def _observation(self) -> np.ndarray:
        target = self._target()
        return np.array(
            [
                self._step_count / max(self.horizon, 1),
                float(target[0]),
                float(target[1]),
                1.0,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        del seed
        self._step_count = 0
        self._episode_return = 0.0
        self._obj_set = (options or {}).get("obj_set", "train")
        return EnvRet(env_id=self.env_id, observation=self._observation())

    def step(self, policy_resp):
        action = self._preprocess_fn(policy_resp) if self._preprocess_fn is not None else policy_resp.action
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        target = self._target()
        reward = 1.0 - float(np.square(action[:2] - target).mean())

        self._step_count += 1
        self._episode_return += reward
        terminated = self._step_count >= self.horizon

        info = {}
        if terminated:
            info["episode_info"] = {
                "episode_return": self._episode_return,
                "success": float(reward > 0.8),
            }

        return EnvRet(
            env_id=self.env_id,
            observation=self._observation(),
            last_reward=reward,
            last_terminated=terminated,
            last_truncated=False,
            info=info,
        )


class _MiniPolicyBase(BasePolicy):
    """Shared training logic for tiny sync/async smoke-test policies."""

    def construct_network(self, env_meta, *args, **kwargs):
        del args, kwargs
        obs_shape = env_meta.observation_space.shape
        act_shape = env_meta.action_space.shape
        obs_dim = int(np.prod(obs_shape))
        action_dim = int(np.prod(act_shape))

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(32, action_dim)
        self.critic_head = nn.Linear(32, 1)
        self.to(self.device)

    def setup_optimizer(self, optim_cfg):
        lr = float(getattr(optim_cfg, "lr", 5e-2))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _tensor_obs(self, observation: Any) -> torch.Tensor:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return obs.reshape(obs.shape[0], -1)

    def _policy_forward(self, observation: Any) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self._tensor_obs(observation)
        hidden = self.encoder(obs)
        action = torch.tanh(self.actor_head(hidden))
        value = self.critic_head(hidden).squeeze(-1)
        return action, value

    def _rollout_step_impl(self, env_ret: EnvRet) -> PolicyResponse:
        action, value = self._policy_forward(env_ret.observation)
        log_prob = torch.zeros_like(value)
        return PolicyResponse(
            env_id=env_ret.env_id,
            action=action.squeeze(0),
            log_prob=log_prob.squeeze(0),
            value=value.squeeze(0),
        )

    def postprocess(self, env_ret=None, policy_resp=None):
        return {
            "env_id": getattr(env_ret, "env_id", getattr(policy_resp, "env_id", None)),
        }

    def update_dataset(self, data) -> None:
        self.dataset = to_device(data, self.device)

    def train(self):
        obs = self.dataset["observation"].float()
        obs = obs.reshape(obs.shape[0], -1)
        reward = self.dataset["reward"].float().reshape(-1)
        target_action = obs[:, 1:3].detach()

        inner_updates = int(getattr(self.config.train_config, "inner_updates", 2))
        metrics: Dict[str, float] = {}
        for _ in range(inner_updates):
            hidden = self.encoder(obs)
            predicted_action = torch.tanh(self.actor_head(hidden))
            predicted_value = self.critic_head(hidden).squeeze(-1)

            actor_loss = F.mse_loss(predicted_action, target_action)
            value_loss = F.mse_loss(predicted_value, reward)
            loss = actor_loss + 0.1 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metrics = {
                "loss": float(loss.detach().cpu().item()),
                "actor_loss": float(actor_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
            }

        return metrics

    def get_trainable_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        params: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, model in self.model_list:
            module = model.module if hasattr(model, "module") else model
            params[name] = module.state_dict()
        return params

    def load_state_dict(self, state_dict, strict=True, assign=False):
        del strict, assign
        for name, model in self.model_list:
            module = model.module if hasattr(model, "module") else model
            module.load_state_dict(state_dict[name])


@POLICIES.register("MiniSyncPolicy")
class MiniSyncPolicy(_MiniPolicyBase):
    """Sync rollout variant for tiny smoke tests."""

    def rollout_step(self, env_ret: EnvRet, **kwargs):
        del kwargs
        return self._rollout_step_impl(env_ret)


@POLICIES.register("MiniAsyncPolicy")
class MiniAsyncPolicy(_MiniPolicyBase):
    """Async rollout variant for tiny smoke tests."""

    async def rollout_step(self, env_ret: EnvRet, **kwargs):
        del kwargs
        return self._rollout_step_impl(env_ret)
