"""
Planned unit tests for DataBuffer in base_buffer.py.
"""

from unittest import mock

import numpy as np
import pytest

from rlightning.buffer.base_buffer import DataBuffer
from rlightning.buffer.sampler import AllDataSampler, BatchSampler, UniformSampler
from rlightning.buffer.utils.storage import Storage
from rlightning.buffer.utils.table import EpisodeTable
from rlightning.types.batched_data import BatchedData
from rlightning.utils.config import BufferConfig

def _make_buffer_config(
    sampler_type: str = "uniform", storage_type: str = "unified", num_shards: int = 2
) -> BufferConfig:
    storage_cfg = {
        "type": storage_type,
        "device": "cpu",
        "unit": "transition",
    }
    if storage_type == "sharded":
        storage_cfg["num_shards"] = num_shards
    return BufferConfig.from_dict(
        {
            "type": "ReplayBuffer",
            "capacity": 10,
            "sampler": {"type": sampler_type},
            "storage": storage_cfg,
        }
    )


class _FakeSubmitter:
    def __init__(self):
        self.calls = []

    def submit(self, method, *args, _block=False, **kwargs):
        self.calls.append((method.__name__, args, kwargs, _block))
        return method(*args, **kwargs)


class _FakeStorage:
    def __init__(self, size: int = 0, label: str = "storage"):
        self.size = size
        self.label = label
        self.data = []
        self.cleared = False
        self.env_stats = {}

    def add_episode(self, episode, num_envs):
        self.data.append(("add_episode", episode, num_envs))

    def truncate_one_episode(self, item):
        self.data.append(("truncate_one_episode", item))

    def truncate_episodes(self, items):
        self.data.append(("truncate_episodes", list(items)))

    def add_transition(self, env_ret, policy_resp):
        self.data.append(("add_transition", env_ret, policy_resp))

    def add_data_async(self, data):
        self.data.append(("add_data_async", data))

    def get_size(self):
        return self.size

    def __getitem__(self, item):
        return (self.label, item)

    def get_data(self):
        return self.data

    def get_env_stats(self, reset):
        return self.env_stats

    def clear(self):
        self.cleared = True
        self.size = 0

    def print_timing_summary(self, reset):
        self.data.append(("print_timing_summary", reset))


class _FakeSampler:
    def __init__(self, indices):
        self.indices = indices

    def sample(self, batch_size, data_size, shuffle=True):
        if batch_size is None:
            return list(range(data_size))
        return list(self.indices)[:batch_size]


def test_sanity_check_preprocessor_matrix():
    """Ensure _sanity_check follows the preprocess conflict matrix.
    There are three levels of preprocessors, and _sanity_check should
    ensure the configuration and preprocess functions are legal.
    """

    cfg = _make_buffer_config()

    def _custom_preprocessor(_):
        return _

    def _custom_env_ret_preprocess_fn(_env_ret):
        return _env_ret

    def _custom_preprocess_fn(_env_ret, _policy_resp):
        return _env_ret, _policy_resp

    cases = [
        # Default (Happy Path)
        ({}, False),
        # Only atomic-level custom (Obs/Reward)
        ({"obs_preprocessor": _custom_preprocessor}, False),
        # Only component-level custom (EnvRet)
        ({"env_ret_preprocess_fn": _custom_env_ret_preprocess_fn}, False),
        # Only global-level custom (Global preprocess_fn)
        ({"preprocess_fn": _custom_preprocess_fn}, False),
        # Conflict: atomic vs component
        (
            {
                "obs_preprocessor": _custom_preprocessor,
                "env_ret_preprocess_fn": _custom_env_ret_preprocess_fn,
            },
            True,
        ),
        # Conflict: atomic vs global
        (
            {
                "obs_preprocessor": _custom_preprocessor,
                "preprocess_fn": _custom_preprocess_fn,
            },
            True,
        ),
        # Conflict: component vs global
        (
            {
                "env_ret_preprocess_fn": _custom_env_ret_preprocess_fn,
                "preprocess_fn": _custom_preprocess_fn,
            },
            True,
        ),
    ]

    for kwargs, should_raise in cases:
        if should_raise:
            with pytest.raises(ValueError):
                _ = DataBuffer(config=cfg, **kwargs)
        else:
            _ = DataBuffer(config=cfg, **kwargs)


def test_init_sampler():
    """Plan: build configs for "uniform", "all", "batch" samplers and
    verify DataBuffer._init_sampler instantiates the expected sampler class.
    Unknown sampler types should raise ValueError, but it will be privented
    by earlier config validationin from_dict when we try to make buffer config,
    so we skip that case here.
    """

    cfg_uniform = _make_buffer_config(sampler_type="uniform")
    buf_uniform = DataBuffer(config=cfg_uniform)
    buf_uniform._init_sampler()
    assert isinstance(buf_uniform.sampler, UniformSampler)

    cfg_all = _make_buffer_config(sampler_type="all")
    buf_all = DataBuffer(config=cfg_all)
    buf_all._init_sampler()
    assert isinstance(buf_all.sampler, AllDataSampler)

    cfg_batch = _make_buffer_config(sampler_type="batch")
    buf_batch = DataBuffer(config=cfg_batch)
    buf_batch._init_sampler()
    assert isinstance(buf_batch.sampler, BatchSampler)


# In base_buffer.py, _init_storage determines where to store data (locally or remotely) and how
# to organize it (unified or sharded). Need to be tested.
# Once local built, storages should be Storage instances, and table should be EpisodeTable with num_storages=1.
def test_init_storage_unified_local_builds_storage(monkeypatch):
    """locally storage, verify storages is one Storage instance, table is None."""
    # Plan: force InternalFlag.REMOTE_STORAGE=False, call init(),
    # then assert storages[0] is a Storage instance and table is EpisodeTable with num_storages=1.
    monkeypatch.setenv("RLIGHTNING_REMOTE_STORAGE", "0")
    cfg = _make_buffer_config(storage_type="unified")
    buffer = DataBuffer(config=cfg)
    buffer.init(env_meta_list=None, env_ids=None)

    assert list(buffer.storages.keys()) == [0]
    assert isinstance(buffer.storages[0], Storage)
    assert buffer.table.num_storages == 1


# Once remote built, unified storage should have one storage actor and EpisodeTable.
# And the table shouldn't be None.
def test_init_storage_sharded_remote_builds_episode_table(monkeypatch):
    """remote sharded storage, verify storages has num_shards entries, table is EpisodeTable."""
    # Plan: force InternalFlag.REMOTE_STORAGE=True, stub ray.remote to a fake,
    # then assert storages has num_shards entries and table is EpisodeTable.
    monkeypatch.setenv("RLIGHTNING_REMOTE_STORAGE", "1")
    cfg = _make_buffer_config(storage_type="sharded", num_shards=2)
    buffer = DataBuffer(config=cfg)

    _mock_grm = mock.MagicMock()
    _mock_scheduling = mock.MagicMock()
    _mock_scheduling.buffer_worker.worker_num = 2
    _mock_scheduling.train_worker.worker_num = 2
    _mock_grm.get_scheduling.return_value = _mock_scheduling

    buffer._global_resource_manager = _mock_grm

    def _fake_init_storage_actor(self, env_meta_list, gpu_requirement, scheduling_kwargs):
        return f"storage-{len(self.storages)}"

    buffer._init_storage_actor = _fake_init_storage_actor.__get__(buffer, DataBuffer)
    buffer.init(env_meta_list=None, env_ids=["env-1", "env-2"])

    assert set(buffer.storages.keys()) == {0, 1}
    assert isinstance(buffer.table, EpisodeTable)
    assert buffer.table.num_storages == 2


# in fact, still need to cover unified+remote and sharded+local and unknown type. if I had time ...

# test add_transition


def test_add_transition_truncate_calls_storage_methods():
    """truncated=true"""
    # Plan: inject fake storage + TaskSubmitter, call add_transition with truncated=True,
    # then assert add_transition and truncate_one_episode are called in order.
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage()}
    buffer.task_submitter = _FakeSubmitter()
    buffer.init_storage_table()

    buffer.add_transition("env-1", {"r": 1}, {"a": 0}, truncated=True)

    assert [call[0] for call in buffer.task_submitter.calls] == [
        "add_transition",
        "truncate_one_episode",
    ]
    assert buffer.task_submitter.calls[0][-1] is False
    assert buffer.task_submitter.calls[1][-1] is False


def test_add_transition_routes_to_sharded_storage():
    """sharded storage should route by env_id."""
    # Plan: set EpisodeTable with two envs, ensure env-2 routes to storage 1.
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage(label="s0"), 1: _FakeStorage(label="s1")}
    buffer.init_storage_table(env_ids=["env-1", "env-2"], train_worker_num=2)
    buffer.task_submitter = _FakeSubmitter()

    buffer.add_transition("env-2", {"r": 1}, {"a": 0}, truncated=False)

    assert buffer.storages[1].data[0][0] == "add_transition"
    assert buffer.storages[0].data == []


def test_add_batched_transition_validates_length_and_ids():
    """whether the original function could raise ValueErorr or TypeError when lengths or
    ids mismatch.
    """
    # Plan: create fake BatchedData with mismatched length/ids and assert ValueError.
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)

    env_ret = BatchedData(ids=["env-1"], data=[{"r": 1}])
    policy_resp = BatchedData(ids=["env-1", "env-2"], data=[{"a": 0}, {"a": 1}])
    with pytest.raises(ValueError):
        buffer.add_batched_transition(env_ret, policy_resp)

    env_ret = BatchedData(ids=["env-1", "env-2"], data=[{"r": 1}, {"r": 2}])
    policy_resp = BatchedData(ids=["env-1", "env-3"], data=[{"a": 0}, {"a": 1}])
    with pytest.raises(ValueError):
        buffer.add_batched_transition(env_ret, policy_resp)


def test_add_batched_transition_validates_truncations():
    """whether the original function could raise ValueErorr or TypeError when truncations
    is invalid.
    """
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)

    env_ret = BatchedData(ids=["env-1", "env-2"], data=[{"r": 1}, {"r": 2}])
    policy_resp = BatchedData(ids=["env-1", "env-2"], data=[{"a": 0}, {"a": 1}])

    with pytest.raises(TypeError):
        buffer.add_batched_transition(env_ret, policy_resp, truncations="not-a-list")

    with pytest.raises(ValueError):
        buffer.add_batched_transition(env_ret, policy_resp, truncations=[True])


def test_add_batched_transition_calls_add_transition_and_returns_self():
    """default truncations=None should call add_transition for each item."""
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage()}
    buffer.init_storage_table()
    buffer.task_submitter = _FakeSubmitter()

    env_ret = BatchedData(ids=["env-1", "env-2"], data=[{"r": 1}, {"r": 2}])
    policy_resp = BatchedData(ids=["env-1", "env-2"], data=[{"a": 0}, {"a": 1}])

    result = buffer.add_batched_transition(env_ret, policy_resp, truncations=None)

    assert result is buffer
    assert [call[0] for call in buffer.task_submitter.calls] == [
        "add_transition",
        "add_transition",
    ]
    assert buffer.task_submitter.calls[0][1] == ({"r": 1}, {"a": 0})
    assert buffer.task_submitter.calls[1][1] == ({"r": 2}, {"a": 1})


def test_add_batched_transition_passes_truncations_per_item():
    """truncations per item should pass to add_transition calls."""
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage()}
    buffer.init_storage_table()
    buffer.task_submitter = _FakeSubmitter()

    env_ret = BatchedData(ids=["env-1", "env-2"], data=[{"r": 1}, {"r": 2}])
    policy_resp = BatchedData(ids=["env-1", "env-2"], data=[{"a": 0}, {"a": 1}])

    buffer.add_batched_transition(env_ret, policy_resp, truncations=[True, False])

    transition_calls = [call for call in buffer.task_submitter.calls if call[0] == "add_transition"]
    assert transition_calls[0][1] == ({"r": 1}, {"a": 0})
    assert transition_calls[1][1] == ({"r": 2}, {"a": 1})
    assert buffer.task_submitter.calls[0][-1] is False
    assert buffer.task_submitter.calls[1][-1] is False
    assert buffer.storages[0].data[1][0] == "truncate_one_episode"


def test_truncate_one_episode_rejects_invalid_item():
    """test if env_id is None, raise TypeError."""
    # Plan: call truncate_one_episode with object missing env_id and assert TypeError.
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)

    class _NoEnvId:
        pass

    with pytest.raises(TypeError):
        buffer.truncate_one_episode(_NoEnvId())


def test_truncate_one_episode_routes_and_passes_item():
    """test routes by using different env_ids. and test passing object with env_id attr."""
    cfg = _make_buffer_config(storage_type="sharded", num_shards=2)
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage(label="s0"), 1: _FakeStorage(label="s1")}
    buffer.init_storage_table(env_ids=["env-1", "env-2"], train_worker_num=2)
    buffer.task_submitter = _FakeSubmitter()

    # 1. direct env_id string
    buffer.truncate_one_episode("env-1")

    class _Item:
        env_id = "env-2"

    obj = _Item()
    # 2. object with env_id attribute
    buffer.truncate_one_episode(obj)

    assert ("truncate_one_episode", "env-1") in buffer.storages[0].data
    assert ("truncate_one_episode", obj) in buffer.storages[1].data


def test_get_rejects_invalid_item_type():
    """object() will raise TypeError in get()."""
    # Plan: call get() with a non-dict, non-int, non-sequence and assert TypeError.
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)

    with pytest.raises(TypeError):
        buffer.get(object())

    # input None
    with pytest.raises(TypeError):
        buffer.get(None)

    # input float
    with pytest.raises(TypeError):
        buffer.get(3.14)

    # TODO: input dict but wrong items


def test_len_and_size_sum_storage_sizes():
    """verify len() and size() sum across storages and query each shard."""
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage(size=3), 1: _FakeStorage(size=5)}
    buffer.task_submitter = _FakeSubmitter()

    # and __len__ in base_buffer.py use _block=True and submit to task_submitter,
    # so...
    assert len(buffer) == 8
    assert len(buffer.task_submitter.calls) == 2
    assert all(call[0] == "get_size" for call in buffer.task_submitter.calls)
    assert all(call[-1] is True for call in buffer.task_submitter.calls)

    buffer.task_submitter = _FakeSubmitter()
    assert buffer.size() == 8
    assert len(buffer.task_submitter.calls) == 2
    assert all(call[0] == "get_size" for call in buffer.task_submitter.calls)
    assert all(call[-1] is True for call in buffer.task_submitter.calls)


def test_add_episode_uses_random_storage(monkeypatch):
    """test add_episode uses random.choice to select storage.
    but here we fix the choice to always return storage 1.
    """
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage(label="s0"), 1: _FakeStorage(label="s1")}
    buffer.task_submitter = _FakeSubmitter()

    monkeypatch.setattr("rlightning.buffer.base_buffer.random.choice", lambda keys: 1)
    buffer.add_episode({"ep": 1}, num_envs=1)

    assert buffer.storages[1].data[0][0] == "add_episode"
    # so far we still need to verify what we had storaged.
    call_record = buffer.storages[1].data[0]
    assert call_record[0] == "add_episode"
    assert call_record[1] == {"ep": 1}
    assert call_record[2] == 1


# the point of add_data_async is the "async". but how to test it?
# truncated is true or false, different env_ids, and keep data unchanged.


def test_add_data_async_truncate_calls_storage_methods():
    """Validate behavior of add_data_async when truncated=True.
    Specifically, ensure:
    1. tasks are submitted in the correct order, and
    2. submissions are non-blocking (i.e. _block=False).
    """
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage()}
    buffer.init_storage_table()
    buffer.task_submitter = _FakeSubmitter()

    data = {"r": 1}
    buffer.add_data_async("env-1", data, truncated=True)

    assert [call[0] for call in buffer.task_submitter.calls] == [
        "add_data_async",
        "truncate_one_episode",
    ]
    assert buffer.task_submitter.calls[0][1][0] == data
    assert buffer.task_submitter.calls[1][1][0] == data
    assert buffer.task_submitter.calls[0][-1] is False
    assert buffer.task_submitter.calls[1][-1] is False


def test_add_data_async_no_truncate_only_adds():
    """Validate behavior of add_data_async when truncated=False.
    Specifically, ensure only add_data_async is called.
    """
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage()}
    buffer.init_storage_table()
    buffer.task_submitter = _FakeSubmitter()

    data = {"r": 2}
    buffer.add_data_async("env-1", data, truncated=False)

    assert [call[0] for call in buffer.task_submitter.calls] == ["add_data_async"]
    assert buffer.task_submitter.calls[0][1][0] == data
    assert buffer.task_submitter.calls[0][-1] is False


def test_add_data_async_routes_to_sharded_storage():
    """sharded storage should route by env_id.
    very import test."""
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage(label="s0"), 1: _FakeStorage(label="s1")}
    buffer.init_storage_table(env_ids=["env-1", "env-2"], train_worker_num=2)
    buffer.task_submitter = _FakeSubmitter()

    data = {"r": 3}
    buffer.add_data_async("env-2", data, truncated=False)

    assert buffer.storages[1].data[0] == ("add_data_async", data)
    assert buffer.storages[0].data == []


def test_add_batched_data_async_validates_truncations():
    """as you can see, validation of truncations in add_batched_data_async."""
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    data = BatchedData(ids=["env-1", "env-2"], data=[{"r": 1}, {"r": 2}])

    with pytest.raises(TypeError):
        buffer.add_batched_data_async(data, truncations="not-a-list")

    with pytest.raises(ValueError):
        buffer.add_batched_data_async(data, truncations=[True])


def test_add_batched_data_async_default_truncations_calls_add_data_async():
    """test default truncations=None calls add_data_async for each item."""
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage()}
    buffer.init_storage_table()
    buffer.task_submitter = _FakeSubmitter()
    data = BatchedData(ids=["env-1", "env-2"], data=[{"r": 1}, {"r": 2}])

    buffer.add_batched_data_async(data, truncations=None)

    assert [call[0] for call in buffer.task_submitter.calls] == [
        "add_data_async",
        "add_data_async",
    ]
    assert buffer.task_submitter.calls[0][1][0] == {"r": 1}
    assert buffer.task_submitter.calls[1][1][0] == {"r": 2}
    assert buffer.task_submitter.calls[0][-1] is False
    assert buffer.task_submitter.calls[1][-1] is False


def test_add_batched_data_async_passes_truncations_per_item():
    """batch add two data, one truncated, one not.
    test if truncations per item passed to add_data_async calls."""
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage()}
    buffer.init_storage_table()
    buffer.task_submitter = _FakeSubmitter()
    data = BatchedData(ids=["env-1", "env-2"], data=[{"r": 1}, {"r": 2}])

    buffer.add_batched_data_async(data, truncations=[True, False])

    add_calls = [call for call in buffer.task_submitter.calls if call[0] == "add_data_async"]
    assert len(add_calls) == 2
    assert add_calls[0][1][0] == {"r": 1}
    assert add_calls[1][1][0] == {"r": 2}
    assert buffer.storages[0].data[1][0] == "truncate_one_episode"


def test_truncate_episodes_groups_by_storage():
    """test truncate_episodes groups by storage using env_ids.
    group route by storage index."""
    cfg = _make_buffer_config(storage_type="sharded", num_shards=2)
    buffer = DataBuffer(config=cfg)
    buffer.table = EpisodeTable(num_storages=2)
    buffer.storages = {0: _FakeStorage(label="s0"), 1: _FakeStorage(label="s1")}
    buffer.init_storage_table(env_ids=["env-1", "env-2"], train_worker_num=2)
    buffer.task_submitter = _FakeSubmitter()

    buffer.truncate_episodes(["env-1", "env-2"])

    assert ("truncate_episodes", ["env-1"]) in buffer.storages[0].data
    assert ("truncate_episodes", ["env-2"]) in buffer.storages[1].data


def test_truncate_episodes_accepts_objects_and_routes_by_env_id():
    """test truncate_episodes accepts objects and routes by env_id."""
    cfg = _make_buffer_config(storage_type="sharded", num_shards=2)
    buffer = DataBuffer(config=cfg)
    buffer.table = EpisodeTable(num_storages=2, env_ids=["env-1", "env-2"])
    buffer.storages = {0: _FakeStorage(label="s0"), 1: _FakeStorage(label="s1")}
    buffer.init_storage_table(env_ids=["env-1", "env-2"], train_worker_num=2)
    buffer.task_submitter = _FakeSubmitter()

    class _Item:
        def __init__(self, env_id):
            self.env_id = env_id

    buffer.truncate_episodes([_Item("env-1"), _Item("env-2")])

    assert ("truncate_episodes", ["env-1"]) in buffer.storages[0].data
    assert ("truncate_episodes", ["env-2"]) in buffer.storages[1].data


def test_truncate_episodes_rejects_invalid_item():
    """invalid test"""
    cfg = _make_buffer_config(storage_type="sharded", num_shards=2)
    buffer = DataBuffer(config=cfg)
    buffer.table = EpisodeTable(num_storages=2)
    buffer.storages = {0: _FakeStorage(label="s0"), 1: _FakeStorage(label="s1")}
    buffer.init_storage_table(train_worker_num=2)
    buffer.task_submitter = _FakeSubmitter()

    class _NoEnvId:
        pass

    with pytest.raises(TypeError):
        buffer.truncate_episodes(["env-1", _NoEnvId()])

    assert buffer.task_submitter.calls == []


def test_getitem_and_get_dict_route_to_storage():
    """test __getitem__ and get method route to storage correctly."""
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage(label="s0")}
    buffer.task_submitter = _FakeSubmitter()

    assert buffer[0, [1, 2]] == ("s0", [1, 2])
    assert buffer.get({"storage_idx": 0, "indices": [3, 4]}) == ("s0", [3, 4])


def test_get_all_single_and_multi_storage():
    """test get_all works for single and multi storage setups."""
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage(label="s0")}
    buffer.task_submitter = _FakeSubmitter()

    single = buffer.get_all()
    assert single == []

    buffer.storages = {0: _FakeStorage(label="s0"), 1: _FakeStorage(label="s1")}
    multi = buffer.get_all()
    assert set(multi.keys()) == {0, 1}


def test_sample_raises_on_unequal_shard_sizes():
    """test sample raises RuntimeError when shard sizes are unequal"""
    cfg = _make_buffer_config(storage_type="sharded", num_shards=2)
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage(size=3), 1: _FakeStorage(size=4)}
    buffer.task_submitter = _FakeSubmitter()
    buffer.sampler = _FakeSampler(indices=[0, 1, 2])
    buffer.init_storage_table(train_worker_num=2)
    buffer.table._storage_to_train_workers == {0: [0], 1: [1]}

    with pytest.raises(RuntimeError):
        buffer.sample(batch_size=4)


def test_sample_splits_across_workers_drop_last():
    """Covers basic split across shards with drop_last enabled and equal-sized splits."""
    cfg = _make_buffer_config(storage_type="sharded", num_shards=2)
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage(size=4, label="s0"), 1: _FakeStorage(size=4, label="s1")}
    buffer.task_submitter = _FakeSubmitter()
    buffer.sampler = _FakeSampler(indices=[0, 1, 2, 3])
    buffer.init_storage_table(train_worker_num=2)
    buffer.table._storage_to_train_workers == {0: [0], 1: [1]}

    sample_data = buffer.sample(batch_size=4, shuffle=False, drop_last=True)

    assert sample_data[0][0] == "s0"
    assert sample_data[0][1].tolist() == [0, 1]
    assert sample_data[1][0] == "s1"
    assert sample_data[1][1].tolist() == [0, 1]


def test_sample_drop_last_truncates_uneven_split():
    """Ensures drop_last truncates indices so each worker gets the same count."""
    cfg = _make_buffer_config(storage_type="unified", num_shards=1)
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage(size=6, label="s0")}
    buffer.task_submitter = _FakeSubmitter()
    buffer.sampler = _FakeSampler(indices=[0, 1, 2, 3, 4])
    buffer.init_storage_table(train_worker_num=2)
    buffer.table._storage_to_train_workers == {0: [0, 1]}

    sample_data = buffer.sample(batch_size=5, shuffle=False, drop_last=True)

    assert sample_data[0][0] == "s0"
    assert sample_data[0][1].tolist() == [0, 2]
    assert sample_data[1][0] == "s0"
    assert sample_data[1][1].tolist() == [1, 3]


def test_sample_no_drop_last_pads_uneven_split():
    """Ensures drop_last=False pads indices to make splits even across workers."""
    cfg = _make_buffer_config(storage_type="unified", num_shards=1)
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage(size=6, label="s0")}
    buffer.task_submitter = _FakeSubmitter()
    buffer.sampler = _FakeSampler(indices=[0, 1, 2, 3, 4])
    buffer.init_storage_table(train_worker_num=2)
    buffer.table._storage_to_train_workers == {0: [0, 1]}

    sample_data = buffer.sample(batch_size=5, shuffle=False, drop_last=False)

    assert sample_data[0][0] == "s0"
    assert sample_data[0][1].tolist() == [0, 2, 4]
    assert sample_data[1][0] == "s0"
    assert sample_data[1][1].tolist() == [1, 3, 0]


def test_sample_no_drop_last_converts_list_indices_to_array():
    """drop_last=False pads indices; ensure list indices are converted for storage access."""
    cfg = _make_buffer_config(storage_type="unified", num_shards=1)
    buffer = DataBuffer(config=cfg)

    class _TypeCheckingStorage(_FakeStorage):
        def __getitem__(self, item):
            if isinstance(item, list):
                raise TypeError("list indices not supported")
            return super().__getitem__(item)

    buffer.storages = {0: _TypeCheckingStorage(size=6, label="s0")}
    buffer.task_submitter = _FakeSubmitter()
    buffer.sampler = _FakeSampler(indices=[0, 1, 2, 3, 4])
    buffer.init_storage_table(train_worker_num=2)
    buffer.table._storage_to_train_workers = {0: [0, 1]}

    sample_data = buffer.sample(batch_size=5, shuffle=False, drop_last=False)

    assert isinstance(sample_data[0][1], np.ndarray)
    assert isinstance(sample_data[1][1], np.ndarray)
    assert sample_data[0][1].tolist() == [0, 2, 4]
    assert sample_data[1][1].tolist() == [1, 3, 0]


def test_init_storage_table_default_mapping():
    """test init_storage_table default mapping."""
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage()}

    buffer.init_storage_table()

    assert buffer.table._storage_to_train_workers == {0: [0]}


def test_init_storage_table_raises_on_manager_error():
    """test init_storage_table raises RuntimeError on manager error."""
    cfg = _make_buffer_config(storage_type="sharded", num_shards=2)
    buffer = DataBuffer(config=cfg)
    buffer.storages = {0: _FakeStorage(), 1: _FakeStorage()}

    class _BadManager:
        def get_component_distribution(self):
            raise RuntimeError("bad manager")

        def get_scheduling(self):
            raise RuntimeError("bad manager")

    buffer._global_resource_manager = _BadManager()

    with pytest.raises(RuntimeError):
        buffer.init_storage_table(train_worker_num=2)


def test_clear_calls_storage_clear():
    """test clear calls clear on each storage and resets buffer length."""
    cfg = _make_buffer_config()
    buffer = DataBuffer(config=cfg)
    s0 = _FakeStorage(size=2)
    s1 = _FakeStorage(size=3)
    buffer.storages = {0: s0, 1: s1}
    buffer.task_submitter = _FakeSubmitter()

    buffer.clear()

    assert s0.cleared is True
    assert s1.cleared is True
    assert len(buffer) == 0
    assert len(buffer) == 0
