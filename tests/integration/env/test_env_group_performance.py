import pytest

try:
    import gymnasium as gym
except Exception:
    gym = None

import numpy as np

from rlightning.env.base_env import BaseEnv
from rlightning.types import EnvRet
from rlightning.utils.builders import build_env_group, build_policy_group
from rlightning.utils.config import EnvConfig, PolicyConfig, ClusterConfig
from rlightning.utils.registry import ENVS, POLICIES


class _DummyInnerEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *args, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), {}

    def step(self, action):
        observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info


class DummyPerfEnv(BaseEnv):
    def __init__(self, config, preprocess_fn, **kwargs):
        super().__init__(config, preprocess_fn)
        self.env = _DummyInnerEnv()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, *args, **kwargs):
        observation, _info = self.env.reset(*args, **kwargs)
        return EnvRet(env_id=self.env_id, observation=observation)

    def step(self, policy_resp):
        action = self._preprocess_fn(policy_resp) if self._preprocess_fn else policy_resp
        observation, reward, terminated, truncated, info = self.env.step(action)
        return EnvRet(
            env_id=self.env_id,
            observation=observation,
            last_reward=reward,
            last_terminated=terminated,
            last_truncated=truncated,
            info=info,
        )


class DummyPolicy:
    def __init__(self, config, role_type=None):
        self.config = config
        self.role_type = role_type

    def init_weight_buffer(self):
        return None


@pytest.mark.integration
@pytest.mark.slow
def test_env_group_worker_perf(performance_timer, monkeypatch):
    if gym is None:
        pytest.skip("gymnasium not available")

    env_key = "ale"
    policy_key = "mock_perf_policy"

    monkeypatch.setitem(ENVS.module_dict, env_key, DummyPerfEnv)
    monkeypatch.setitem(POLICIES.module_dict, policy_key, DummyPolicy)

    env_cfg = EnvConfig.load_yaml("examples/wbc_tracking/conf/env/single_task.yaml")
    env_cfg.backend = env_key
    env_cfg.task = "DummyTask-v0"
    env_cfg.num_envs = 1

    policy_cfg = PolicyConfig(type=policy_key)

    worker_counts = [1, 2, 4]
    results = []

    for num_workers in worker_counts:
        env_cfg.num_workers = num_workers

        performance_timer.start()
        env_group = build_env_group(env_cfg)
        env_time_s = performance_timer.stop()

        performance_timer.start()
        policy_group = build_policy_group(
            policy_key,
            policy_cfg,
            cluster_cfg=ClusterConfig(),
        )
        policy_time_s = performance_timer.stop()

        env_group.close()
        _ = policy_group

        env_time_s = max(env_time_s, 1e-9)
        policy_time_s = max(policy_time_s, 1e-9)

        results.append(
            {
                "num_workers": num_workers,
                "env_time_s": env_time_s,
                "env_throughput": num_workers / env_time_s,
                "policy_time_s": policy_time_s,
                "policy_throughput": 2 / policy_time_s,
            }
        )
        print(
            f"workers={num_workers} "
            f"env_time_s={env_time_s:.6f} env_throughput={num_workers / env_time_s:.2f} "
            f"policy_time_s={policy_time_s:.6f} policy_throughput={2 / policy_time_s:.2f}"
        )

    for metrics in results:
        assert metrics["env_time_s"] >= 0.0
        assert metrics["policy_time_s"] >= 0.0
        assert metrics["env_throughput"] > 0.0
        assert metrics["policy_throughput"] > 0.0
