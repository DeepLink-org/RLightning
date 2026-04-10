"""Shared helper utilities for tests."""

from __future__ import annotations

import os
import random
import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlightning.utils.config import EnvConfig

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - fallback only used in degraded envs.
    from types import SimpleNamespace

    class _DummyEnv:
        def reset(self, seed=None, options=None):
            return None

    class _DummySpace:
        pass

    class _DummyBox(_DummySpace):
        def __init__(self, low, high, shape, dtype=None):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _DummyDict(dict):
        pass

    gym = SimpleNamespace(Env=_DummyEnv, spaces=SimpleNamespace(Box=_DummyBox, Dict=_DummyDict))

try:
    import ray

    RAY_AVAILABLE = True
except Exception:
    RAY_AVAILABLE = False

try:
    import vllm  # noqa: F401

    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

try:
    import mani_skill  # noqa: F401

    MANISKILL_AVAILABLE = True
except Exception:
    MANISKILL_AVAILABLE = False


class MockEnv(gym.Env):
    def __init__(self, config=None, env_id="mock_env", **kwargs):
        super().__init__()
        self.config = config or {}
        self.env_id = env_id
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
                "state": gym.spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.step_count = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        obs = {
            "rgb": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            "state": np.random.randn(10).astype(np.float32),
        }
        return obs, {}

    def step(self, action):
        self.step_count += 1
        obs = {
            "rgb": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            "state": np.random.randn(10).astype(np.float32),
        }
        reward = float(np.random.randn())
        terminated = self.step_count >= self.max_steps
        truncated = False
        info = {"truncated": truncated}
        return obs, reward, terminated, truncated, info

    def init(self, *args, **kwargs):
        return self.observation_space, self.action_space


class MockEnvGroup:
    def __init__(self, env):
        self.env = env
        self._observation_space = env.observation_space
        self._action_space = env.action_space

    def init(self):
        return self._observation_space, self._action_space

    def reset(self, seed=0):
        obs, info = self.env.reset(seed=seed)
        return MagicMock(obs=[obs], info=[info]), []

    def step(self, responses):
        actions = [r.action for r in responses] if hasattr(responses[0], "action") else responses
        obs, reward, terminated, truncated, info = self.env.step(actions[0])
        return (
            MagicMock(obs=[obs], reward=[reward], done=[terminated], truncated=[truncated], info=[info]),
            [],
        )

    def observation_space(self):
        return self._observation_space

    def action_space(self):
        return self._action_space


class MockPolicyGroup:
    def __init__(self):
        self.task = None
        self.train_config = {}

    def init_eval(self, eval_config, env_meta):
        return MagicMock()

    def init_train(self, train_config, env_meta):
        pass

    def rollout_batch(self, requests):
        actions = torch.randn(len(requests.obs), 4)
        return MagicMock(
            action=actions,
            value=torch.randn(len(requests.obs)),
            log_prob=torch.randn(len(requests.obs)),
            ids=lambda: [str(uuid.uuid4()) for _ in range(len(requests.obs))],
        )

    def update_dataset(self, buffer, batch_size):
        pass

    def train(self):
        pass

    def send_weights_to_eval(self):
        pass

    def notify_update_weights(self):
        pass


class MockBuffer:
    def __init__(self, config):
        self.config = config
        self.data = []

    def add(self, obs, act):
        self.data.append((obs, act))

    def sample(self, batch_size, device="cpu"):
        if not self.data:
            return None, None
        indices = np.random.choice(len(self.data), min(batch_size, len(self.data)), replace=False)
        obs_batch = [self.data[i][0] for i in indices]
        act_batch = [self.data[i][1] for i in indices]
        return obs_batch, act_batch

    def init(self, episode_meta):
        pass

    def add_transitions_async(self, transitions, truncations=None):
        if hasattr(transitions, "obs"):
            for i in range(len(transitions.obs)):
                self.add(transitions.obs[i], getattr(transitions, "action", [0])[i])

    def size(self):
        return len(self.data)

    def clear(self):
        self.data.clear()

    def truncate_episodes(self, episode_ids):
        pass


def create_tiny_config():
    return OmegaConf.create(
        {
            "env": {
                "envs": [{"name": "mock_env", "num_envs": 1}],
                "num_envs": 1,
            },
            "policy": {
                "type": "mock_policy",
                "train_config": {},
            },
            "buffer": {
                "type": "mock_buffer",
            },
            "train": {
                "max_epochs": 2,
                "max_rollout_steps": 10,
                "batch_size": 4,
            },
            "mode": "sync",
            "train_worker_num": 1,
            "eval_worker_num": 1,
            "logging": {
                "type": "console",
            },
        }
    )


def create_sample_obs():
    return {
        "rgb": torch.randint(0, 255, (1, 64, 64, 3), dtype=torch.uint8),
        "state": torch.randn(1, 10),
    }


def create_sample_episode():
    t = 10
    return {
        "obs": [torch.randn(64, 64, 3) for _ in range(t)],
        "action": [torch.randn(4) for _ in range(t)],
        "reward": [torch.tensor(float(np.random.randn())) for _ in range(t)],
        "value": [torch.tensor(float(np.random.randn())) for _ in range(t)],
        "next/value": [torch.tensor(float(np.random.randn())) for _ in range(t)],
        "done": [torch.tensor(False) for _ in range(t - 1)] + [torch.tensor(True)],
        "truncated": [torch.tensor(False) for _ in range(t)],
    }


_ORIGINAL_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)
_ORIG_CUDA = None


def get_test_ray_address(default="local"):
    return os.environ.get("RLIGHTNING_TEST_RAY_ADDRESS", default)


def init_test_ray(
    num_cpus=1,
    num_gpus=0,
    local_mode=True,
    log_to_driver=False,
    runtime_env=None,
    default_address="local",
):
    global _ORIG_CUDA

    if not RAY_AVAILABLE:
        raise RuntimeError("Ray not available")

    if _ORIG_CUDA is None:
        _ORIG_CUDA = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if ray.is_initialized():
        ray.shutdown()
        time.sleep(0.2)

    ray_address = get_test_ray_address(default=default_address)
    if ray_address == "local":
        ray.init(
            address="local",
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=log_to_driver,
            local_mode=local_mode,
            runtime_env=runtime_env,
        )
    else:
        ray.init(
            address=ray_address,
            ignore_reinit_error=True,
            log_to_driver=log_to_driver,
            runtime_env=runtime_env,
        )

    return ray_address


def setup_ray_cluster(num_cpus=1, num_gpus=0, local_mode=True, log_to_driver=False):
    return init_test_ray(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        local_mode=local_mode,
        log_to_driver=log_to_driver,
        default_address="local",
    )


def teardown_ray_cluster():
    global _ORIG_CUDA

    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()
        time.sleep(0.2)

    if _ORIG_CUDA is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ORIG_CUDA


def check_dependency_availability():
    return {
        "ray": RAY_AVAILABLE,
        "vllm": VLLM_AVAILABLE,
        "maniskill": MANISKILL_AVAILABLE,
        "gpu": torch.cuda.is_available(),
    }


def create_tmp_artifacts_dir(tmp_path):
    artifacts_dir = tmp_path / "outputs" / "test_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def set_deterministic_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _compute_gae(rewards, values, dones, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])

    for t in reversed(range(rewards.shape[0])):
        next_not_done = 1.0 - dones[t].float()
        current_value = values[t]
        next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_not_done - current_value
        lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values[:-1]
    return advantages, returns


def _episode_postprocess_fn(raw_episode):
    episode = {}
    for key, value in raw_episode.items():
        if "info" in key:
            continue
        if isinstance(value[0], torch.Tensor):
            episode[key] = torch.stack(value).squeeze()
        else:
            episode[key] = torch.tensor(value)

    episode["last_reward"] = episode["last_reward"][1:]
    dones = torch.logical_or(episode["last_terminated"], episode["last_truncated"])
    episode["done"] = dones[1:]
    episode["action"] = episode["action"][:-1]
    episode["last_terminated"] = episode["last_terminated"][1:]
    episode["last_truncated"] = episode["last_truncated"][1:]
    episode["log_prob"] = episode["log_prob"][:-1]
    episode["entropy"] = episode["entropy"][:-1]
    episode["observation"] = episode["observation"][:-1]

    rewards = episode["last_reward"]
    values = episode["value"]
    episode["value"] = episode["value"][:-1]
    dones = episode["done"]

    advantages, returns = _compute_gae(rewards, values, dones, 0.8, 0.9)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    episode["advantages"] = advantages
    episode["returns"] = returns
    return episode


def _make_env_config(env_backend, task_name, num_workers, num_envs=1):
    return EnvConfig(
        backend=env_backend,
        name=f"{env_backend}/{task_name}",
        task=task_name,
        max_episode_steps=20,
        num_workers=num_workers,
        num_envs=num_envs,
    )


class PerformanceTimer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        return self.elapsed

    @property
    def elapsed(self):
        if self.start_time is None:
            return 0
        end = self.end_time or time.time()
        return end - self.start_time


def should_skip_gpu_test():
    return not torch.cuda.is_available()


def should_skip_ray_test():
    return not RAY_AVAILABLE


def should_skip_vllm_test():
    return not VLLM_AVAILABLE


def should_skip_maniskill_test():
    return not MANISKILL_AVAILABLE


def require_env_backend(backend: str) -> None:
    if backend == "mujoco":
        pytest.importorskip("mujoco")
    elif backend == "ale":
        pytest.importorskip("ale_py")
    elif backend == "isaac_manager_based":
        pytest.importorskip("isaaclab")


@pytest.fixture
def performance_timer():
    return PerformanceTimer()
