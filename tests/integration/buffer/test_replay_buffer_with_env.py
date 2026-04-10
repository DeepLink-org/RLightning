import os
import random

import pytest
import torch

from rlightning.buffer.utils.preprocessors import get_preprocessor_cls
from rlightning.env import EnvGroup
from rlightning.types import BatchedData, EnvMeta, PolicyResponse
from rlightning.utils.builders import build_data_buffer, build_env_group
from rlightning.utils.config import BufferConfig
from rlightning.utils.ray import TaskSubmitter, resolve_object
from rlightning.utils.utils import InternalFlag
from tests.test_utils import _make_env_config, require_env_backend

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
            "type": "ReplayBuffer",
            "capacity": capacity,
            "storage": {
                "type": "unified",
                "device": "cpu",
            },
            "sampler": {
                "type": "uniform",
            },
        }
    )


def _make_env(env_cfg, as_remote: bool):
    require_env_backend(env_cfg.backend)
    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_ENV == as_remote
    env, _ = EnvGroup.build_env(env_cfg)
    if as_remote:
        env_meta = env.init.remote()
    else:
        env_meta = env.init()
    return env, env_meta


def _make_env_group(num_env_workers, as_remote):
    os.environ["RLIGHTNING_REMOTE_ENV"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_ENV == as_remote
    require_env_backend("mujoco")
    env_cfgs = [
        _make_env_config(env_backend="mujoco", task_name="Ant-v5", num_workers=num_env_workers),
    ]

    env_group = build_env_group(env_cfgs)

    return env_group


def _mock_batched_policy_ret(action_spaces, env_ids):
    action_list = [resolve_object(action_space).sample() for action_space in action_spaces]
    policy_resp_list = [
        PolicyResponse(env_id=env_id, action=action, log_prob=0.0, entropy=0.0)
        for action, env_id in zip(action_list, env_ids)
    ]
    batched_policy_resp = BatchedData(env_ids, policy_resp_list)
    return batched_policy_resp


def _mock_policy_ret(action_space, env_id):
    action = action_space.sample()
    policy_resp = PolicyResponse(env_id=env_id, action=action, log_prob=0.0, entropy=0.0)
    return policy_resp


def _make_buffer(buffer_cfg, env_meta_list=None, as_remote: bool = False):
    os.environ["RLIGHTNING_REMOTE_STORAGE"] = "1" if as_remote else "0"
    assert InternalFlag.REMOTE_STORAGE == as_remote

    env_meta = env_meta_list[0] if env_meta_list is not None else None
    if env_meta is not None and env_meta.observation_space is not None:
        observation_space = env_meta.observation_space
        obs_preprocessor = get_preprocessor_cls(observation_space)(observation_space)
        buffer = build_data_buffer(buffer_cls=buffer_cfg.type, buffer_cfg=buffer_cfg, obs_preprocessor=obs_preprocessor)
    else:
        buffer = build_data_buffer(buffer_cls=buffer_cfg.type, buffer_cfg=buffer_cfg)

    buffer.init(env_meta_list)

    return buffer


def _make_episode(episode_length: int = 8, n_envs: int = 1, obs_dim: int = 3, act_dim: int = 2):
    if n_envs == 1:
        return {
            "obs": torch.randn(episode_length, obs_dim),
            "action": torch.randn(episode_length, act_dim),
        }

    return {
        "obs": torch.randn(episode_length, n_envs, obs_dim),
        "action": torch.randn(episode_length, n_envs, act_dim),
    }


@pytest.mark.parametrize(
    "env_backend,task_name",
    [
        ["mujoco", "Ant-v5"],
        ["ale", "ALE/Adventure-v5"],
    ],
)
@pytest.mark.parametrize("num_env_workers", [1, 2])
@pytest.mark.parametrize("env_as_remote", [False])
@pytest.mark.parametrize("buffer_as_remote", [False])
def test_replay_buffer(
    env_backend: str,
    task_name: str,
    num_env_workers: int,
    env_as_remote: bool,
    buffer_as_remote: bool,
):
    """Test the functionality of replay buffer.

    The coverage includes:

        1. Making timesteps holder (now only test for single-environment yet)
        2. Functional preprocessing for each timestep
        3. Functional postprocessing for a dict of batched-timesteps, in the order of [N_timesteps, N_envs, *inner_dims]
        4. Pushing a dict of batch, but not test for the case of overflow
        5. Functional and Correctness checking for the buffer sampling, cases including: a) predefined batch size; b) random batch size; c) given indices with random batch_size

    Args:
        env_backend (str): Environmnt backend
        task_name (str): Task name
        num_env_workers (int): Environment Instance number
    """

    env_cfg = _make_env_config(env_backend, task_name, num_env_workers)
    buffer_cfg = _make_buffer_config(capacity=100)
    # TODO(ming): apply environment vectorization for future work
    env, env_meta = _make_env(env_cfg, env_as_remote)
    env_meta_list = None if env_meta is None else [env_meta]
    buffer = _make_buffer(buffer_cfg, env_meta_list, buffer_as_remote)

    batch_size: int = 10

    env_ret = env.reset()
    buffer.add_data_async(env_ret.env_id, env_ret)
    step_cnt = 0
    while not (env_ret.last_terminated or env_ret.last_truncated or step_cnt >= env_cfg.max_episode_steps):
        action = env.action_space.sample()
        policy_resp = PolicyResponse(env_id=env_ret.env_id, action=action, log_prob=0.0, entropy=0.0)
        buffer.add_data_async(env_ret.env_id, policy_resp)

        env_ret = env.step(policy_resp)
        buffer.add_data_async(env_ret.env_id, env_ret)

        step_cnt += 1

    action = env.action_space.sample()
    policy_resp = PolicyResponse(env_id=env_ret.env_id, action=action, log_prob=0.0, entropy=0.0)
    buffer.add_data_async(env_ret.env_id, policy_resp)
    buffer.truncate_episodes([policy_resp.env_id])

    assert env_ret.last_terminated or env_ret.last_truncated, (
        env_ret.last_terminated,
        env_ret.last_truncated,
    )

    assert len(buffer) == step_cnt, (step_cnt, len(buffer))

    custom_batch_size = random.choice(range(1, batch_size))
    sample_data = buffer.sample(batch_size=custom_batch_size)
    assert len(sample_data) == 1
    data = resolve_object(sample_data[0])
    for k, v in data.items():
        assert len(v) == custom_batch_size, (k, v.shape)


@pytest.mark.parametrize("buffer_as_remote", [True, False])
@pytest.mark.parametrize("env_as_remote", [False])
def test_init_creates_storage(buffer_as_remote: bool, env_as_remote: bool):
    env_backend = "mujoco"
    task_name = "Ant-v5"
    num_env_workers = 1
    env_cfg = _make_env_config(env_backend, task_name, num_env_workers)
    buffer_cfg = _make_buffer_config(capacity=100)

    _, env_meta = _make_env(env_cfg, env_as_remote)
    env_meta_list = None if env_meta is None else [env_meta]
    buffer = _make_buffer(buffer_cfg, env_meta_list, buffer_as_remote)
    assert len(buffer.storages) > 0
    # assert buf.table is not None
    assert buffer.size() == 0


@pytest.mark.parametrize("num_envs", [1, 2, 4])
@pytest.mark.parametrize("buffer_as_remote", [True, False])
def test_add_increases_size(num_envs, buffer_as_remote):
    env_meta = EnvMeta(
        env_id=None,
        action_space=None,
        observation_space=None,
        num_envs=num_envs,
    )
    buffer_cfg = _make_buffer_config(capacity=100)
    buf = _make_buffer(buffer_cfg, [env_meta], as_remote=buffer_as_remote)
    episode_length = 8
    episode = _make_episode(episode_length=episode_length, n_envs=num_envs)
    buf.add_episode(episode, num_envs=num_envs)

    assert buf.size() == episode_length * num_envs

    sample_data = buf.sample(batch_size=8)
    assert len(sample_data) == 1
    data = resolve_object(sample_data[0])
    for k, v in data.items():
        assert len(v) == 8, (k, v.shape)


@pytest.mark.parametrize("buffer_as_remote", [True, False])
def test_clear_resets_size(buffer_as_remote):
    buffer_cfg = _make_buffer_config(capacity=100)
    buf = _make_buffer(buffer_cfg, as_remote=buffer_as_remote)
    episode_length = 8
    episode = _make_episode(episode_length=episode_length, n_envs=1)
    buf.add_episode(episode, num_envs=1)

    assert buf.size() == episode_length
    buf.clear()
    assert buf.size() == 0


@pytest.mark.parametrize("buffer_as_remote", [True, False])
def test_circular_mode_overwrite_when_exceed_capacity(buffer_as_remote):
    # capacity 10, add 14 => size should be capped at 10
    buffer_cfg = _make_buffer_config(capacity=10)
    buf = _make_buffer(buffer_cfg, as_remote=buffer_as_remote)
    buf.add_episode(_make_episode(episode_length=14, n_envs=1), num_envs=1)

    assert buf.size() == 10


@pytest.mark.parametrize("env_as_remote", [True, False])
@pytest.mark.parametrize("buffer_as_remote", [True, False])
def test_add_transition(env_as_remote, buffer_as_remote):
    env_backend = "mujoco"
    task_name = "Ant-v5"
    num_env_workers = 1
    env_cfg = _make_env_config(env_backend, task_name, num_env_workers)
    task_submitter = TaskSubmitter()

    env, _ = _make_env(env_cfg, env_as_remote)
    buffer_cfg = _make_buffer_config(capacity=10)
    buffer = _make_buffer(buffer_cfg, as_remote=buffer_as_remote)

    env_ret = task_submitter.submit(env.reset, _block=True)
    for _ in range(5):
        action = task_submitter.submit(env.get_action_space, _block=True).sample()
        policy_resp = PolicyResponse(env_id=env_ret.env_id, action=action, log_prob=0.0, entropy=0.0)
        buffer.add_transition(env_ret.env_id, env_ret, policy_resp)
        env_ret = task_submitter.submit(env.step, policy_resp, _block=True)

    action = task_submitter.submit(env.get_action_space, _block=True).sample()
    policy_resp = PolicyResponse(env_id=env_ret.env_id, action=action, log_prob=0.0, entropy=0.0)
    buffer.add_transition(env_ret.env_id, env_ret, policy_resp, truncated=True)

    assert len(buffer) == 5


@pytest.mark.parametrize("env_as_remote", [True, False])
@pytest.mark.parametrize("buffer_as_remote", [True, False])
def test_add_batched_transition(env_as_remote, buffer_as_remote):
    env_group = _make_env_group(num_env_workers=2, as_remote=env_as_remote)

    buffer_cfg = _make_buffer_config(capacity=100)
    buffer = _make_buffer(buffer_cfg, as_remote=buffer_as_remote)

    batched_env_ret, _ = env_group.reset()
    for _ in range(5):
        batched_policy_resp = _mock_batched_policy_ret(env_group.get_action_spaces(), batched_env_ret.ids())
        buffer.add_batched_transition(batched_env_ret, batched_policy_resp)
        batched_env_ret, _ = env_group.step(batched_policy_resp)

    batched_policy_resp = _mock_batched_policy_ret(env_group.get_action_spaces(), batched_env_ret.ids())
    truncations = [True] * len(batched_env_ret)
    buffer.add_batched_transition(batched_env_ret, batched_policy_resp, truncations)

    assert len(buffer) == 5 * len(env_group)


@pytest.mark.parametrize("env_as_remote", [True, False])
@pytest.mark.parametrize("buffer_as_remote", [True, False])
def test_add_data_async(env_as_remote, buffer_as_remote):
    env_backend = "mujoco"
    task_name = "Ant-v5"
    num_env_workers = 1
    env_cfg = _make_env_config(env_backend, task_name, num_env_workers)

    task_submitter = TaskSubmitter()

    env, _ = _make_env(env_cfg, env_as_remote)
    buffer_cfg = _make_buffer_config(capacity=10)
    buffer = _make_buffer(buffer_cfg, as_remote=buffer_as_remote)

    env_ret = task_submitter.submit(env.reset, _block=True)
    buffer.add_data_async(env_ret.env_id, env_ret)
    for _ in range(5):
        policy_resp = _mock_policy_ret(task_submitter.submit(env.get_action_space, _block=True), env_ret.env_id)
        buffer.add_data_async(env_ret.env_id, policy_resp)
        env_ret = task_submitter.submit(env.step, policy_resp, _block=True)
        buffer.add_data_async(env_ret.env_id, env_ret)

    policy_resp = _mock_policy_ret(task_submitter.submit(env.get_action_space, _block=True), env_ret.env_id)
    buffer.add_data_async(env_ret.env_id, policy_resp, truncated=True)

    assert len(buffer) == 5


@pytest.mark.parametrize("env_as_remote", [True])
@pytest.mark.parametrize("buffer_as_remote", [True, False])
def test_add_batched_data_async(env_as_remote, buffer_as_remote):
    env_group = _make_env_group(num_env_workers=2, as_remote=env_as_remote)

    buffer_cfg = _make_buffer_config(capacity=100)
    buffer = _make_buffer(buffer_cfg, as_remote=buffer_as_remote)

    batched_env_ret, _ = env_group.reset()
    buffer.add_batched_data_async(batched_env_ret)
    for _ in range(5):
        batched_policy_resp = _mock_batched_policy_ret(env_group.get_action_spaces(), batched_env_ret.ids())
        env_group.step_async(batched_policy_resp)
        buffer.add_batched_data_async(batched_policy_resp)
        batched_env_ret, _ = env_group.collect_async()
        buffer.add_batched_data_async(batched_env_ret)

    batched_policy_resp = _mock_batched_policy_ret(env_group.get_action_spaces(), batched_env_ret.ids())
    truncations = [True] * len(batched_env_ret)
    buffer.add_batched_data_async(batched_policy_resp, truncations)

    # to avoid the case that some of the env's step has not been collected
    batche_env_ret, _ = env_group.collect_async(wait_all=True)
    if len(batche_env_ret) > 0:
        buffer.add_batched_data_async(batche_env_ret)
        batched_policy_resp = _mock_batched_policy_ret(env_group.get_action_spaces(), batched_env_ret.ids())
        buffer.add_batched_data_async(batched_policy_resp)

        truncations = [True] * len(batched_env_ret)
        buffer.add_batched_data_async(batched_policy_resp, truncations)

    assert len(buffer) == 5 * len(env_group)


@pytest.mark.parametrize("env_as_remote", [True, False])
@pytest.mark.parametrize("buffer_as_remote", [True, False])
def test_truncate_one_episode(env_as_remote, buffer_as_remote):
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

    assert len(buffer) == 0

    buffer.truncate_one_episode(policy_resp.env_id)
    assert len(buffer) == 5


@pytest.mark.parametrize("env_as_remote", [True, False])
@pytest.mark.parametrize("buffer_as_remote", [True, False])
def test_truncate_episodes(env_as_remote, buffer_as_remote):
    env_group = _make_env_group(num_env_workers=2, as_remote=env_as_remote)

    buffer_cfg = _make_buffer_config(capacity=100)
    buffer = _make_buffer(buffer_cfg, as_remote=buffer_as_remote)

    batched_env_ret, _ = env_group.reset()
    for _ in range(5):
        batched_policy_resp = _mock_batched_policy_ret(env_group.get_action_spaces(), batched_env_ret.ids())
        buffer.add_batched_transition(batched_env_ret, batched_policy_resp)
        batched_env_ret, _ = env_group.step(batched_policy_resp)

    batched_policy_resp = _mock_batched_policy_ret(env_group.get_action_spaces(), batched_env_ret.ids())
    buffer.add_batched_transition(batched_env_ret, batched_policy_resp)

    assert len(buffer) == 0

    buffer.truncate_episodes(batched_policy_resp.ids())
    assert len(buffer) == 5 * len(env_group)
    assert len(buffer) == 5 * len(env_group)
