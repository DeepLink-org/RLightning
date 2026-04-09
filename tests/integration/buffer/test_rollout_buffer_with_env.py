import os

import pytest

from rlightning.buffer.utils.preprocessors import get_preprocessor_cls
from rlightning.env import EnvGroup
from rlightning.types import PolicyResponse
from rlightning.utils.builders import build_data_buffer
from rlightning.utils.config import BufferConfig
from rlightning.utils.ray import TaskSubmitter, resolve_object
from rlightning.utils.utils import InternalFlag
from tests.test_utils import _make_env_config, require_env_backend
from tensordict import TensorDict

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _auto_ray_cluster():
    # Ensure a Ray cluster is available for this module's tests
    from tests.test_utils import setup_ray_cluster, teardown_ray_cluster

    setup_ray_cluster()
    yield
    teardown_ray_cluster()


def _make_buffer_config(capacity: int = 100) -> BufferConfig:
    return BufferConfig.from_dict(
        {
            "type": "RolloutBuffer",
            "capacity": capacity,
            "storage": {
                "type": "unified",
                "device": "cpu",
            },
            "sampler": {
                "type": "all",
            },
        }
    )


def _make_env(env_cfg, as_remote):
    require_env_backend(env_cfg.backend)
    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_ENV == as_remote
    env, _ = EnvGroup.build_env(env_cfg)
    if as_remote:
        env_meta = env.init.remote()
    else:
        env_meta = env.init()
    return env, env_meta


def _make_buffer(buffer_cfg, env_meta_list=None, as_remote: bool = False):
    os.environ["RLIGHTNING_REMOTE_STORAGE"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_STORAGE == as_remote

    env_meta = env_meta_list[0] if env_meta_list is not None else None
    if env_meta is not None and "observation_space" in env_meta:
        observation_space = env_meta.observation_space
        obs_preprocessor = get_preprocessor_cls(observation_space)(observation_space)
        buffer = build_data_buffer(buffer_cls=buffer_cfg.type, buffer_cfg=buffer_cfg, obs_preprocessor=obs_preprocessor)
    else:
        buffer = build_data_buffer(buffer_cls=buffer_cfg.type, buffer_cfg=buffer_cfg)

    buffer.init(env_meta_list)

    return buffer


@pytest.mark.parametrize("env_as_remote", [False, True])
@pytest.mark.parametrize("buffer_as_remote", [False, True])
def test_sample(env_as_remote, buffer_as_remote):
    require_env_backend("mujoco")
    env_backend = "mujoco"
    task_name = "Ant-v5"
    num_env_workers = 1
    env_cfg = _make_env_config(env_backend, task_name, num_env_workers)

    env, _ = _make_env(env_cfg, env_as_remote)
    buffer_cfg = _make_buffer_config(capacity=10)
    buffer = _make_buffer(buffer_cfg, as_remote=buffer_as_remote)

    task_submitter = TaskSubmitter()

    env_ret = task_submitter.submit(env.reset, _block=True)
    for _ in range(5):
        action = task_submitter.submit(env.get_action_space, _block=True).sample()
        policy_resp = PolicyResponse(env_id=env_ret.env_id, action=action, log_prob=0.0, entropy=0.0)
        buffer.add_transition(env_ret.env_id, env_ret, policy_resp)
        env_ret = task_submitter.submit(env.step, policy_resp, _block=True)

    action = task_submitter.submit(env.get_action_space, _block=True).sample()
    policy_resp = PolicyResponse(env_id=env_ret.env_id, action=action, log_prob=0.0, entropy=0.0)
    buffer.add_transition(env_ret.env_id, env_ret, policy_resp)

    buffer.truncate_one_episode(policy_resp.env_id)
    assert len(buffer) == 5
    sample_data = buffer.sample(batch_size=5)
    data = resolve_object(sample_data[0])
    data = TensorDict.from_dict(data, auto_batch_size=True)
    assert len(data) == 5
    # after RolloutBuffer.sample, the buffer will clear automatically
    assert len(buffer) == 0
