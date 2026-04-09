import asyncio
import copy
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from rlightning.utils.placement.placement_manager import GlobalResourceManager
from tests.test_utils import PerformanceTimer, set_deterministic_seeds, should_skip_gpu_test, should_skip_ray_test

_ROOT = Path(__file__).resolve().parents[1]


def _deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


@pytest.fixture(autouse=True)
def reset_global_resource_manager() -> None:
    GlobalResourceManager.reset()
    yield
    GlobalResourceManager.reset()


@pytest.fixture(autouse=True)
def set_random_seeds():
    set_deterministic_seeds(42)
    yield
    np.random.seed()
    torch.manual_seed(torch.initial_seed())
    random.seed()


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def performance_timer():
    return PerformanceTimer()


@pytest.fixture(scope="session")
def ray_multi_node_cluster():
    if should_skip_ray_test() or should_skip_gpu_test():
        yield
        return

    env = os.environ.copy()
    subprocess.run(["ray", "stop", "--force"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        head_env = env.copy()
        head_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        subprocess.run(
            ["ray", "start", "--head", "--disable-usage-stats", "--num-gpus", "2"],
            cwd=str(_ROOT),
            env=head_env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        worker1_env = env.copy()
        worker1_env["CUDA_VISIBLE_DEVICES"] = "2,3,4"
        subprocess.run(
            ["ray", "start", "--address", "auto", "--num-gpus", "3"],
            cwd=str(_ROOT),
            env=worker1_env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        worker2_env = env.copy()
        worker2_env["CUDA_VISIBLE_DEVICES"] = "5,6,7"
        subprocess.run(
            ["ray", "start", "--address", "auto", "--num-gpus", "3"],
            cwd=str(_ROOT),
            env=worker2_env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        yield
    finally:
        subprocess.run(["ray", "stop", "--force"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@pytest.fixture
def make_main_config_dict():
    def factory(**overrides):
        config = {
            "engine": "syncrl",
            "env": [
                {
                    "name": "test-env",
                    "backend": "mujoco",
                    "task": "cartpole",
                    "num_workers": 2,
                    "num_envs": 1,
                    "num_cpus": 2,
                    "num_gpus": 0.0,
                }
            ],
            "buffer": {
                "type": "RolloutBuffer",
                "capacity": 32,
                "storage": {
                    "type": "unified",
                    "device": "cpu",
                },
            },
            "policy": {
                "type": "SimplePPOPolicy",
                "train_num_gpus": 1.0,
                "eval_num_gpus": 0.5,
                "weight_buffer": {
                    "type": "WeightBuffer",
                    "buffer_strategy": "Double",
                },
            },
            "train": {
                "max_epochs": 1,
                "batch_size": 8,
            },
            "cluster": {
                "train_worker_num": 2,
                "eval_worker_num": 3,
                "buffer_worker_num": 1,
                "remote_train": False,
                "remote_eval": False,
                "remote_storage": False,
                "remote_env": False,
            },
        }
        return _deep_update(copy.deepcopy(config), overrides)

    return factory
