import random
import time
from typing import Dict, List

import numpy as np
import ray
import torch

from rlightning.env import BaseEnv
from rlightning.env.env_server import RemoteEnvServer
from rlightning.env.remote_env.env_client import RemoteEnvClient
from rlightning.env.utils.utils import default_env_preprocess_fn
from rlightning.types import EnvRet, PolicyResponse, Processed_EnvRet_fields
from rlightning.utils.config import EnvConfig


class MockPiperEnv(BaseEnv):
    def __init__(self, config, worker_index=0, preprocess_fn=default_env_preprocess_fn) -> None:
        super().__init__(config, worker_index, preprocess_fn)
        self.step_cnt = 0
        self.episode_cnt = 0
        self.episode_reward = 0
        self.total_steps = config.total_steps

    def reset(self):
        obs = {
            "state": np.zeros((7,)),
            "camera_rgb_front": np.zeros((480, 640, 3), dtype=np.uint8),
            "camera_rgb_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        self.step_cnt = 0
        self.episode_cnt = 0
        self.episode_reward = 0
        return EnvRet(env_id=self.env_id, observation=obs, ts_env_sent_ns=time.time_ns())

    def step(self, policy_resp: PolicyResponse) -> EnvRet:
        _ = self._preprocess_fn(policy_resp)
        obs = {
            "state": np.zeros((7,)),
            "camera_rgb_front": np.zeros((480, 640, 3), dtype=np.uint8),
            "camera_rgb_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        done = self.step_cnt == random.choice([2, 5, 8])
        truncated = self.step_cnt == random.choice([2, 5, 8])
        self.step_cnt += 1
        time.sleep(random.random() * 0.2)
        return EnvRet(
            env_id=policy_resp.env_id,
            observation=obs,
            last_reward=0.0,
            last_terminated=done,
            last_truncated=truncated,
            info={},
            ts_env_sent_ns=time.time_ns(),
        )

    @classmethod
    def episode_postprocess_fn(cls, episode_buffer: Dict) -> Dict:
        data = {}
        for k, v in episode_buffer.items():
            if "info" in k:
                continue
            if k == "observation":
                state = np.array([elem["state"] for elem in v])
                data["state"] = state[:-1]
                data["next_state"] = state[1:]
                camera_rgb_front = np.array([elem["camera_rgb_front"] for elem in v])
                data["camera_rgb_front"] = camera_rgb_front[:-1]
                data["next_camera_rgb_front"] = camera_rgb_front[1:]
                camera_rgb_wrist = np.array([elem["camera_rgb_wrist"] for elem in v])
                data["camera_rgb_wrist"] = camera_rgb_wrist[:-1]
                data["next_camera_rgb_wrist"] = camera_rgb_wrist[1:]
                continue

            if isinstance(v[0], torch.Tensor):
                v = torch.stack(v, dim=0)
            else:
                v = torch.tensor(v)
            if k.startswith("last_"):
                data[k[5:]] = v[1:]
            else:
                data[k] = v

            env_fields = set(EnvRet.fields() + Processed_EnvRet_fields)
            policy_fields = [field for field in episode_buffer.keys() if field not in env_fields]
            if k in policy_fields:
                data[k] = v[:-1]

        return data

    def get_action_space(self):
        return

    def get_observation_space(self):
        return

    def is_finish(self):
        return self.step_cnt >= self.total_steps


@ray.remote
class EnvClientWorker:
    def __init__(self, address, port, total_steps=10):
        config = EnvConfig(
            name="MockPiper-v0",
            task="MockPiper-v0",
            backend="piper",
            max_episode_steps=1000,
            total_steps=total_steps,
        )
        self.env = MockPiperEnv(config)
        self.client = RemoteEnvClient(self.env, address, port)

    def run(self):
        self.client.connect()
        self.client.run()
        time.sleep(random.choice([1, 4, 7]))
        self.client.env.reset()
        self.client.connect()
        self.client.run()

    def is_alive(self):
        return True


@ray.remote
class EnvServerWorker:
    def __init__(self):
        config = EnvConfig(name="test_env_server", backend="env_server", task="real_world", zmq_port="6366")
        self.server = RemoteEnvServer(config)
        self.server.init()

    def _mock_policy_response_list(self, env_ret_list: List[EnvRet]) -> List[PolicyResponse]:
        policy_response_list = []
        for env_ret in env_ret_list:
            action = np.random.uniform(-1, 1, size=(7,))
            policy_response_list.append(PolicyResponse(env_id=env_ret.env_id, action=action))
        return policy_response_list

    def get_address_port(self):
        return self.server.get_address_port()

    def run(self, num_envs, num_steps, timeout=100):
        server = self.server
        env_ret_list = server.reset()
        cnt = len(env_ret_list)
        expect_cnts = (1 + num_steps) * num_envs * 2

        while True:
            policy_resp_list = self._mock_policy_response_list(env_ret_list)
            server.step_async(policy_resp_list)
            env_ret_list = server.collect_async()
            cnt += len(env_ret_list)

            if cnt == expect_cnts:
                policy_resp_list = self._mock_policy_response_list(env_ret_list)
                server.step_async(policy_resp_list)
                env_ret_list = server.collect_async(timeout=10)
                return len(env_ret_list) == 0

    def is_alive(self):
        return True

    def close(self):
        self.server.close()
