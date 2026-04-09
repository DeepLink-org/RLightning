import os
import time
from typing import List

import numpy as np
import pytest
import ray

from rlightning.buffer.utils.preprocessors import get_preprocessor_cls
from rlightning.types import BatchedData, PolicyResponse
from rlightning.utils.builders import build_data_buffer, build_env_group
from rlightning.utils.config import BufferConfig, EnvConfig
from rlightning.utils.ray import TaskSubmitter, resolve_object
from rlightning.utils.utils import InternalFlag
from tests.tests_utils.envs.utils_remote_env import (
    EnvClientWorker,
    EnvServerWorker,
    MockPiperEnv,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _auto_ray_cluster():
    # Ensure a Ray cluster is available for this module's tests
    from tests.test_utils import setup_ray_cluster, teardown_ray_cluster

    setup_ray_cluster(num_cpus=4, num_gpus=2, local_mode=False, log_to_driver=True)
    yield
    teardown_ray_cluster()


def _mock_batched_policy_response(batched_env_ret: BatchedData, policy_as_remote=False) -> List[PolicyResponse]:
    policy_response_list = []
    for env_ret in batched_env_ret.values():
        env_ret = resolve_object(env_ret)
        action = np.random.uniform(-1, 1, size=(7,))
        policy_response = PolicyResponse(env_id=env_ret.env_id, action=action)
        if policy_as_remote:
            policy_response = ray.put(policy_response)
        policy_response_list.append(policy_response)

    return BatchedData(batched_env_ret.ids(), policy_response_list)


@pytest.mark.parametrize("num_envs", [4])
@pytest.mark.parametrize("num_steps", [10])
def test_env_server_client_interaction(num_envs, num_steps):
    server = EnvServerWorker.remote()
    address, port = ray.get(server.get_address_port.remote())

    success = server.run.remote(num_envs, num_steps, timeout=100)
    workers = [EnvClientWorker.remote(address, port, num_steps) for _ in range(num_envs)]
    _ = [worker.run.remote() for worker in workers]

    success = ray.get(success)

    assert success, "Env server-client interaction test failed."

    server.close.remote()


@pytest.mark.parametrize("as_remote", [False, True])
@pytest.mark.parametrize("num_steps", [10])
@pytest.mark.parametrize("num_envs", [4])
def test_env_group_with_env_server_to_interaction(as_remote, num_steps, num_envs):
    task_submitter = TaskSubmitter()

    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_ENV == as_remote

    env_server_cfg = EnvConfig(
        name="test_PiperEnv_Server",
        backend="env_server",
        task="real_world",
        zmq_port="6366",
    )
    env_cfgs = [env_server_cfg]
    env_group = build_env_group(env_cfgs)
    _ = env_group.init()

    address, port = task_submitter.submit(env_group.env_servers[0].get_address_port, _block=True)

    workers = [EnvClientWorker.remote(address, port, num_steps) for _ in range(num_envs)]
    _ = [worker.run.remote() for worker in workers]

    batched_env_ret, _ = env_group.reset()
    # two times reset called defined in RemoteEnvClient.run()
    expect_cnt = (1 + num_steps) * num_envs * 2
    cnt = len(batched_env_ret)

    step = 0
    while True:
        step += 1
        batched_policy_resp = _mock_batched_policy_response(batched_env_ret, as_remote)
        env_group.step_async(batched_policy_resp)
        batched_env_ret, _ = env_group.collect_async()
        cnt += len(batched_env_ret)

        print(f"Step {step}: received {len(batched_env_ret)}, total {cnt}.", flush=True)
        # FIXME cannot detect the wrong case that clients send more returns than expected
        # and if not receive enough, the test will hang here forever.
        if cnt == expect_cnt:
            break

    env_group.close()


def _make_buffer_config(capacity: int = 100) -> BufferConfig:
    return BufferConfig.from_dict(
        {
            "type": "ReplayBuffer",
            "capacity": capacity,
            "auto_truncate_episode": True,
            "storage": {
                "type": "unified",
                "device": "cpu",
            },
            "sampler": {
                "type": "uniform",
            },
        }
    )


def _make_buffer(buffer_cfg, env_meta_list=None, as_remote: bool = False):
    os.environ["RLIGHTNING_REMOTE_STORAGE"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_STORAGE == as_remote

    env_meta = env_meta_list[0] if env_meta_list is not None else None
    if env_meta is not None and "observation_space" in env_meta:
        observation_space = env_meta.observation_space
        obs_preprocessor = get_preprocessor_cls(observation_space)(observation_space)
        buffer = build_data_buffer(
            buffer_cls=buffer_cfg.type,
            buffer_cfg=buffer_cfg,
            obs_preprocessor=obs_preprocessor,
        )
    else:
        buffer = build_data_buffer(
            buffer_cls=buffer_cfg.type,
            buffer_cfg=buffer_cfg,
            postprocess_fn=MockPiperEnv.episode_postprocess_fn,
        )

    buffer.init(env_meta_list)

    return buffer


@pytest.mark.parametrize("as_remote", [True, False])
@pytest.mark.parametrize("num_steps", [10])
@pytest.mark.parametrize("num_envs", [4])
def test_rollout_and_collect_replay_buffer_with_remote_env(as_remote, num_steps, num_envs):
    task_submitter = TaskSubmitter()

    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_ENV == as_remote

    buffer_cfg = _make_buffer_config(capacity=100)
    buffer = _make_buffer(buffer_cfg, env_meta_list=None, as_remote=as_remote)

    env_server_cfg = EnvConfig(
        name="test_PiperEnv_Server",
        backend="env_server",
        task="real_world",
        zmq_port="6366",
    )
    env_cfgs = [env_server_cfg]
    env_group = build_env_group(env_cfgs)
    _ = env_group.init()
    address, port = task_submitter.submit(env_group.env_servers[0].get_address_port, _block=True)

    workers = [EnvClientWorker.remote(address, port, num_steps) for _ in range(num_envs)]
    _ = [worker.run.remote() for worker in workers]

    # two times reset called defined in RemoteEnvClient.run()
    expect_cnt = (1 + num_steps) * num_envs * 2

    batched_env_ret, _ = env_group.reset()
    cnt = len(batched_env_ret)
    step = 0
    buffer.add_batched_data_async(batched_env_ret)
    while True:
        batched_policy_resp = _mock_batched_policy_response(batched_env_ret, as_remote)
        buffer.add_batched_data_async(batched_policy_resp)

        env_group.step_async(batched_policy_resp)
        batched_env_ret, _ = env_group.collect_async()
        buffer.add_batched_data_async(batched_env_ret)

        step += 1
        cnt += len(batched_env_ret)

        print(f"Step {step}: received {len(batched_env_ret)}, total {cnt}.", flush=True)
        # FIXME cannot detect the wrong case that clients send more returns than expected
        # and if not receive enough, the test will hang here forever.
        if cnt == expect_cnt:
            break

    assert buffer.size() > 0

    env_group.close()
