"""Unit tests for storage utilities in storage.py."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from rlightning.buffer.utils.storage import (
    ActiveEpisodeBuffer,
    BufferView,
    DataContainer,
    Storage,
)
from rlightning.types import EnvMeta, EnvRet, PolicyResponse

def _stack_postprocess(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Convert list-based episode fields into tensors for storage."""
    processed: Dict[str, Any] = {}
    for key, value in episode.items():
        if isinstance(value, list):
            if len(value) == 0:
                processed[key] = torch.empty((0,))
            elif isinstance(value[0], torch.Tensor):
                processed[key] = torch.stack(value)
            else:
                processed[key] = torch.tensor(value)
        else:
            processed[key] = value
    return processed


def _simple_preprocess(
    transition_buffer: Dict[str, Any],
    env_ret: EnvRet | None,
    policy_resp: PolicyResponse | None,
    *_args,
    **_kwargs,
) -> Dict[str, Any]:
    """Minimal preprocess to assemble a transition dict for tests."""
    if env_ret is not None:
        transition_buffer["obs"] = torch.as_tensor(env_ret.observation)
        transition_buffer["last_terminated"] = bool(env_ret.last_terminated)
        transition_buffer["last_truncated"] = bool(env_ret.last_truncated)
    if policy_resp is not None:
        transition_buffer["action"] = torch.as_tensor(getattr(policy_resp, "action", 0))
    return transition_buffer


def _make_storage(auto_truncate: bool, unit: str = "episode") -> Storage:
    return Storage(
        capacity=10,
        mode="circular",
        unit=unit,
        env_meta_list=None,
        device="cpu",
        obs_preprocessor=lambda x: x,
        reward_preprocessor=lambda x: x,
        env_ret_preprocess_fn=lambda x: x,
        policy_resp_preprocess_fn=lambda x: x,
        preprocess_fn=_simple_preprocess,
        postprocess_fn=_stack_postprocess,
        auto_truncate_episode=auto_truncate,
    )


def test_buffer_view_len_and_getitem():
    """simple test for BufferView length and indexing."""
    data = TensorDict({"value": torch.tensor([1, 2, 3])}, batch_size=[3])
    view = BufferView(data)

    assert len(view) == 3
    assert view[1]["value"].item() == 2


def test_buffer_view_negative_and_slice_indexing():
    """test for BufferView negative indexing and slicing."""
    data = TensorDict({"value": torch.tensor([5, 6, 7])}, batch_size=[3])
    view = BufferView(data)

    assert view[-1]["value"].item() == 7
    sliced = view[1:]
    assert sliced["value"].tolist() == [6, 7]


def test_buffer_view_empty_and_full_slice():
    """test for BufferView with empty data and full slice indexing."""
    data = TensorDict({"value": torch.tensor([])}, batch_size=[0])
    view = BufferView(data)

    assert len(view) == 0
    sliced = view[:]
    assert sliced["value"].numel() == 0


def test_buffer_view_multidim_batch_len_and_item():
    """test for BufferView with multi-dimensional batch size."""
    data = TensorDict({"value": torch.tensor([[1, 2, 3], [4, 5, 6]])}, batch_size=[2, 3])
    view = BufferView(data)

    assert len(view) == 2
    assert view[1]["value"].tolist() == [4, 5, 6]


def test_active_episode_buffer_single_env_pop_done_episode():
    """pop done episode is very import in ActiveEpisodeBuffer.
    so we test it as folloing case. we verify only two steps first.
    """
    buffer = ActiveEpisodeBuffer(
        env_meta_list=None,
        auto_truncate_episode=True,
        postprocess_fn=_stack_postprocess,
        device="cpu",
    )

    buffer.add_transition(
        "env-1",
        {
            "obs": torch.tensor(0),
            "last_terminated": False,
            "last_truncated": False,
        },
    )
    buffer.add_transition(
        "env-1",
        {
            "obs": torch.tensor(1),
            "last_terminated": True,
            "last_truncated": False,
        },
    )

    episodes = buffer.pop_done_episode("env-1")

    assert len(episodes) == 1
    assert episodes[0]["obs"].tolist() == [0, 1]
    assert episodes[0]["last_terminated"].tolist() == [False, True]
    assert buffer.pop_done_episode("env-1") == []


def test_pop_done_episode_with_env_id_none():
    """Test env_id=None pops all done episodes from all environments."""
    buffer = ActiveEpisodeBuffer(
        env_meta_list=None,
        auto_truncate_episode=True,
        postprocess_fn=_stack_postprocess,
        device="cpu",
    )

    # Add done episode for env-1
    buffer.add_transition(
        "env-1",
        {"obs": torch.tensor(1), "last_terminated": True, "last_truncated": False},
    )
    # Add done episode for env-2
    buffer.add_transition(
        "env-2",
        {"obs": torch.tensor(2), "last_terminated": True, "last_truncated": False},
    )
    # Add non-done episode for env-3
    buffer.add_transition(
        "env-3",
        {"obs": torch.tensor(3), "last_terminated": False, "last_truncated": False},
    )

    # Pop all done episodes (env_id=None)
    episodes = buffer.pop_done_episode(env_id=None)

    assert len(episodes) == 2
    obs_values = sorted([ep["obs"].tolist()[0] for ep in episodes])
    assert obs_values == [1, 2]

    # env-3 should still have data (not done)
    assert "env-3" in buffer._buffer


def test_pop_done_episode_with_last_truncated():
    """Test last_truncated=True also triggers episode done."""
    buffer = ActiveEpisodeBuffer(
        env_meta_list=None,
        auto_truncate_episode=True,
        postprocess_fn=_stack_postprocess,
        device="cpu",
    )

    buffer.add_transition(
        "env-1",
        {"obs": torch.tensor(0), "last_terminated": False, "last_truncated": False},
    )
    # Use last_truncated instead of last_terminated
    buffer.add_transition(
        "env-1",
        {"obs": torch.tensor(1), "last_terminated": False, "last_truncated": True},
    )

    episodes = buffer.pop_done_episode("env-1")

    assert len(episodes) == 1
    assert episodes[0]["obs"].tolist() == [0, 1]
    assert episodes[0]["last_truncated"].tolist() == [False, True]


def test_pop_done_episode_with_sub_env_ids():
    """Test sub-env done flag tracking in multi-env scenarios.

    TODO: Note: When auto_truncate_episode=True and num_envs>1, transitions are split
    into sub-env IDs (env-vec/0, env-vec/1). The done flags are tracked per sub-env.
    However, pop_done_episode has a limitation: it finds sub-env IDs but pop()
    expects parent env_id. This test verifies the done flag tracking behavior.
    @yangzhenyu I don't konw whether it will be fixed in future develop branch, so just
    check it when you merging.
    """
    env_meta_list = [EnvMeta(env_id="env-vec", num_envs=2)]
    buffer = ActiveEpisodeBuffer(
        env_meta_list=env_meta_list,
        auto_truncate_episode=True,
        postprocess_fn=_stack_postprocess,
        device="cpu",
    )

    # Add transition where only sub-env 0 is done
    buffer.add_transition(
        "env-vec",
        {
            "obs": torch.tensor([10, 20]),
            "last_terminated": torch.tensor([True, False]),
            "last_truncated": torch.tensor([False, False]),
        },
    )

    # Verify sub-envs are stored separately
    assert "env-vec/0" in buffer._buffer
    assert "env-vec/1" in buffer._buffer

    # Verify done flags are tracked per sub-env
    assert buffer._episodes_done_flag["env-vec/0"] is True
    assert buffer._episodes_done_flag.get("env-vec/1", False) is False

    # Use pop() with parent env_id to get all sub-env episodes
    # (This is the correct way to pop multi-env episodes)
    episodes = buffer.pop("env-vec")

    assert len(episodes) == 2
    obs_values = sorted([ep["obs"].tolist()[0] for ep in episodes])
    assert obs_values == [10, 20]


def test_pop_done_episode_when_no_done_episodes():
    """Test returns empty list when no episodes are done."""
    buffer = ActiveEpisodeBuffer(
        env_meta_list=None,
        auto_truncate_episode=True,
        postprocess_fn=_stack_postprocess,
        device="cpu",
    )

    # Add non-done transitions
    buffer.add_transition(
        "env-1",
        {"obs": torch.tensor(0), "last_terminated": False, "last_truncated": False},
    )
    buffer.add_transition(
        "env-1",
        {"obs": torch.tensor(1), "last_terminated": False, "last_truncated": False},
    )

    # No done episodes
    episodes = buffer.pop_done_episode("env-1")
    assert episodes == []

    # Also test with env_id=None
    episodes = buffer.pop_done_episode(env_id=None)
    assert episodes == []

    # Data should still be in buffer
    assert "env-1" in buffer._buffer
    assert len(buffer._buffer["env-1"]) == 2


def test_pop_done_episode_multiple_envs_different_states():
    """Test multiple environments with different done states."""
    buffer = ActiveEpisodeBuffer(
        env_meta_list=None,
        auto_truncate_episode=True,
        postprocess_fn=_stack_postprocess,
        device="cpu",
    )

    # env-1: 2 transitions, done
    buffer.add_transition(
        "env-1",
        {"obs": torch.tensor(1), "last_terminated": False, "last_truncated": False},
    )
    buffer.add_transition(
        "env-1",
        {"obs": torch.tensor(2), "last_terminated": True, "last_truncated": False},
    )

    # env-2: 1 transition, not done
    buffer.add_transition(
        "env-2",
        {"obs": torch.tensor(10), "last_terminated": False, "last_truncated": False},
    )

    # env-3: 1 transition, done via truncated
    buffer.add_transition(
        "env-3",
        {"obs": torch.tensor(100), "last_terminated": False, "last_truncated": True},
    )

    # Pop only env-1
    episodes_1 = buffer.pop_done_episode("env-1")
    assert len(episodes_1) == 1
    assert episodes_1[0]["obs"].tolist() == [1, 2]

    # Pop env-2 (not done, should be empty)
    episodes_2 = buffer.pop_done_episode("env-2")
    assert episodes_2 == []

    # Pop remaining done episodes (env-3)
    episodes_remaining = buffer.pop_done_episode(env_id=None)
    assert len(episodes_remaining) == 1
    assert episodes_remaining[0]["obs"].tolist() == [100]


def test_active_episode_buffer_multi_env_pop_auto_truncate():
    """Test multi-env buffer correctly splits and reassembles transitions when auto_truncate=True."""
    env_meta_list = [EnvMeta(env_id="env-vec", num_envs=2)]
    buffer = ActiveEpisodeBuffer(
        env_meta_list=env_meta_list,
        auto_truncate_episode=True,
        postprocess_fn=_stack_postprocess,
        device="cpu",
    )

    transition_1 = {
        "obs": torch.tensor([1, 2]),
        "last_terminated": torch.tensor([False, True]),
        "last_truncated": torch.tensor([False, False]),
    }
    transition_2 = {
        "obs": torch.tensor([3, 4]),
        "last_terminated": torch.tensor([True, False]),
        "last_truncated": torch.tensor([False, False]),
    }

    buffer.add_transition("env-vec", transition_1)
    buffer.add_transition("env-vec", transition_2)

    episodes = buffer.pop("env-vec")

    # Verify correct number of episodes
    assert len(episodes) == 2

    # Verify obs data correctly split by sub-env
    episode_values = [episode["obs"].tolist() for episode in episodes]
    assert sorted(episode_values) == sorted([[1, 3], [2, 4]])

    # Verify other fields also correctly split
    term_values = [episode["last_terminated"].tolist() for episode in episodes]
    assert sorted(term_values) == sorted([[False, True], [True, False]])

    # Verify buffer is empty after pop
    assert f"env-vec/0" not in buffer._buffer
    assert f"env-vec/1" not in buffer._buffer


def test_active_episode_buffer_multi_env_pop_no_auto_truncate():
    """Test multi-env pop behavior when auto_truncate is False."""
    env_meta_list = [EnvMeta(env_id="env-vec", num_envs=2)]
    buffer = ActiveEpisodeBuffer(
        env_meta_list=env_meta_list,
        auto_truncate_episode=False,
        postprocess_fn=_stack_postprocess,
        device="cpu",
    )

    buffer.add_transition(
        "env-vec",
        {
            "obs": torch.tensor([10, 20]),
            "last_terminated": torch.tensor([False, False]),
            "last_truncated": torch.tensor([False, False]),
        },
    )
    buffer.add_transition(
        "env-vec",
        {
            "obs": torch.tensor([30, 40]),
            "last_terminated": torch.tensor([False, False]),
            "last_truncated": torch.tensor([False, False]),
        },
    )

    episodes = buffer.pop("env-vec")

    assert len(episodes) == 1
    assert episodes[0]["obs"].tolist() == [10, 30, 20, 40]
    assert episodes[0]._metadata == {"_num_episodes": 2, "_episode_length": 2}


def test_active_episode_buffer_get_num_envs_sub_env_fallback():
    """Test sub-env id resolves to single env."""
    env_meta_list = [EnvMeta(env_id="env-vec", num_envs=2)]
    buffer = ActiveEpisodeBuffer(
        env_meta_list=env_meta_list,
        auto_truncate_episode=True,
        postprocess_fn=_stack_postprocess,
        device="cpu",
    )

    assert buffer._get_num_envs("env-vec/0") == 1


def test_active_episode_buffer_pop_done_episode_sub_env_ids_returns_only_done():
    """pop_done_episode should handle sub-env ids when auto_truncate=True."""
    env_meta_list = [EnvMeta(env_id="env-vec", num_envs=2)]
    buffer = ActiveEpisodeBuffer(
        env_meta_list=env_meta_list,
        auto_truncate_episode=True,
        postprocess_fn=_stack_postprocess,
        device="cpu",
    )

    buffer.add_transition(
        "env-vec",
        {
            "obs": torch.tensor([10, 20]),
            "last_terminated": torch.tensor([True, False]),
            "last_truncated": torch.tensor([False, False]),
        },
    )

    episodes = buffer.pop_done_episode("env-vec")

    assert len(episodes) == 1
    assert episodes[0]["obs"].tolist() == [10]
    assert episodes[0]["last_terminated"].tolist() == [True]


def test_data_container_transition_getitem_and_use_counter():
    """Test transition getitem updates data_use_counter correctly."""
    container = DataContainer(capacity=3, mode="circular", unit="transition", device="cpu")
    container.push({"value": torch.tensor([1, 2, 3])})

    assert len(container) == 3
    assert container.data_use_counter.tolist() == [0, 0, 0]

    item = container[1]
    assert item["value"].item() == 2
    assert container.data_use_counter[1] == 1

    sliced = container[0:2]
    assert sliced["value"].tolist() == [1, 2]
    assert container.data_use_counter[0] == 1
    assert container.data_use_counter[1] == 2

    last = container[-1]
    assert last["value"].item() == 3
    assert container.data_use_counter[2] == 1

    items = container[np.array([0, 2])]
    assert items["value"].tolist() == [1, 3]
    assert container.data_use_counter[0] == 2
    assert container.data_use_counter[2] == 2


def test_data_container_transition_negative_and_array_indexing():
    """Test transition negative and array indexing paths."""
    container = DataContainer(capacity=4, mode="circular", unit="transition", device="cpu")
    container.push({"value": torch.tensor([10, 11, 12, 13])})

    assert container[-1]["value"].item() == 13

    indices = np.array([0, -1])
    items = container[indices]
    assert items["value"].tolist() == [10, 13]

    with pytest.raises(IndexError):
        _ = container[-5]

    with pytest.raises(IndexError):
        _ = container[np.array([4])]


def test_data_container_transition_circular_wrap():
    """Test circular wrap overwrites transition data as expected."""
    container = DataContainer(capacity=5, mode="circular", unit="transition", device="cpu")
    container.push({"value": torch.tensor([0, 1, 2])})
    container.push({"value": torch.tensor([3, 4, 5, 6])})

    assert container.size == 5
    assert container.pointer == 2
    assert container.data_use_counter.tolist() == [0, 0, 0, 0, 0]
    assert container.data["value"].tolist() == [5, 6, 2, 3, 4]


def test_data_container_transition_getitem_invalid_index_type_raises():
    """Test transition getitem rejects unsupported index types."""
    container = DataContainer(capacity=3, mode="circular", unit="transition", device="cpu")
    container.push({"value": torch.tensor([1, 2])})

    with pytest.raises(IndexError):
        _ = container["invalid"]


def test_data_container_transition_fixed_overflow_raises():
    """Test fixed mode raises on capacity overflow."""
    container = DataContainer(capacity=3, mode="fixed", unit="transition", device="cpu")
    container.push({"value": torch.tensor([1, 2])})

    with pytest.raises(RuntimeError):
        container.push({"value": torch.tensor([3, 4])})


def test_data_container_clear_resets_state():
    """Test clear resets transition container state."""
    container = DataContainer(capacity=4, mode="circular", unit="transition", device="cpu")
    container.push({"value": torch.tensor([1, 2])})
    container.clear()

    assert container.size == 0
    assert container.pointer == 0
    assert np.all(container.data_use_counter == -1)


def test_data_container_transition_push_list_items():
    """Test pushing a list of transition dicts."""
    container = DataContainer(capacity=4, mode="circular", unit="transition", device="cpu")
    items = [
        {"value": torch.tensor([1])},
        {"value": torch.tensor([2, 3])},
    ]

    container.push(items)

    assert container.size == 3
    assert container.data["value"][: container.size].tolist() == [1, 2, 3]


def test_data_container_transition_push_dict_shape_mismatch_raises():
    """Test push rejects dicts with mismatched shapes."""
    container = DataContainer(capacity=4, mode="circular", unit="transition", device="cpu")
    bad = {"a": torch.tensor([1, 2]), "b": torch.tensor([1, 2, 3])}

    with pytest.raises((ValueError, RuntimeError)):
        container.push(bad)


def test_data_container_episode_len_and_getitem_updates_use_counter():
    """Test episode getitem and use counter updates."""
    container = DataContainer(capacity=8, mode="circular", unit="episode", device="cpu")
    ep1 = TensorDict({"value": torch.tensor([1, 2])}, batch_size=[2])
    ep2 = TensorDict({"value": torch.tensor([3, 4, 5])}, batch_size=[3])

    container.push(ep1)
    container.push(ep2)

    assert len(container) == 2

    ep = container[0]
    assert ep["value"].tolist() == [1, 2]
    assert container.data_use_counter[:2].tolist() == [1, 1]

    eps = container[0:2]
    assert len(eps) == 2
    assert container.data_use_counter[:5].tolist() == [2, 2, 1, 1, 1]

    idx_eps = container[np.array([1, -1])]
    assert len(idx_eps) == 2
    assert idx_eps[0]["value"].tolist() == [3, 4, 5]
    assert container.data_use_counter[:5].tolist() == [2, 2, 3, 3, 3]


def test_data_container_episode_invalid_index_type_raises():
    """Test episode getitem rejects unsupported index types."""
    container = DataContainer(capacity=4, mode="circular", unit="episode", device="cpu")
    ep = TensorDict({"value": torch.tensor([1])}, batch_size=[1])
    container.push(ep)

    with pytest.raises(TypeError):
        _ = container["invalid"]


def test_data_container_episode_check_range_out_of_bounds():
    """Test episode index range checks for out-of-bounds access."""
    container = DataContainer(capacity=4, mode="circular", unit="episode", device="cpu")
    ep = TensorDict({"value": torch.tensor([1])}, batch_size=[1])
    container.push(ep)

    with pytest.raises(IndexError):
        _ = container[1]

    with pytest.raises(IndexError):
        _ = container[np.array([1])]

    with pytest.raises(IndexError):
        _ = container[torch.tensor([-2])]


def test_data_container_episode_overlaps_cleared_on_overwrite():
    """Test episode overlap cleanup when overwriting.
    DataContainer is weak here..."""
    container = DataContainer(capacity=6, mode="circular", unit="episode", device="cpu")
    ep1 = TensorDict({"value": torch.tensor([1, 2, 3])}, batch_size=[3])
    ep2 = TensorDict({"value": torch.tensor([4, 5])}, batch_size=[2])
    ep3 = TensorDict({"value": torch.tensor([6, 7, 8, 9])}, batch_size=[4])

    container.push(ep1)
    container.push(ep2)
    # Mark ep1 as used so we can verify it is cleared on overlap.
    _ = container[0]
    assert container.data_use_counter[:3].tolist() == [1, 1, 1]

    container.push(ep3)

    assert len(container) == 1  # BUG? Should be 2 episodes?
    assert container.data_use_counter[:4].tolist() == [0, 0, 0, 0]
    assert container.data_use_counter[4:].tolist() == [-1, -1]


def test_data_container_episode_circular_large_episode_keeps_last_capacity():
    """Test large episode keeps last capacity in circular mode."""
    container = DataContainer(capacity=3, mode="circular", unit="episode", device="cpu")
    ep = TensorDict({"value": torch.tensor([1, 2, 3, 4])}, batch_size=[4])

    container.push(ep)

    assert len(container) == 1
    assert container.pointer == 0
    assert container.size == 3
    assert container.data["value"].tolist() == [2, 3, 4]


def test_data_container_episode_clear_resets_state():
    """Test clear resets episode container state."""
    container = DataContainer(capacity=4, mode="circular", unit="episode", device="cpu")
    ep = TensorDict({"value": torch.tensor([1, 2])}, batch_size=[2])
    container.push(ep)
    container.clear()

    assert container.size == 0
    assert container.pointer == 0
    assert len(container.episode_registry) == 0
    assert np.all(container.data_use_counter == -1)


def test_storage_add_transition_pushes_done_episode():
    """Test add_transition pushes done episodes and updates stats."""
    storage = _make_storage(auto_truncate=True)

    env_ret_1 = EnvRet(
        env_id="env-1",
        observation=torch.tensor(0),
        last_terminated=False,
        last_truncated=False,
        info={"episode_info": {"score": 1.0}},
    )
    policy_1 = PolicyResponse(env_id="env-1", action=torch.tensor(10))

    storage.add_transition(env_ret_1, policy_1)
    assert len(storage) == 0

    env_ret_2 = EnvRet(
        env_id="env-1",
        observation=torch.tensor(1),
        last_terminated=True,
        last_truncated=False,
        info={"episode_info": {"score": 2.0}},
    )
    policy_2 = PolicyResponse(env_id="env-1", action=torch.tensor(11))

    storage.add_transition(env_ret_2, policy_2)

    assert len(storage) == 1
    episode = storage[0]
    assert episode["obs"].shape[0] == 2
    assert episode["action"].shape[0] == 2


def test_storage_len_get_size_get_data_and_clear():
    """Test length/size getters and clear behavior."""
    storage = _make_storage(auto_truncate=True)

    env_ret = EnvRet(
        env_id="env-1",
        observation=torch.tensor(0),
        last_terminated=True,
        last_truncated=False,
    )
    policy = PolicyResponse(env_id="env-1", action=torch.tensor(1))

    storage.add_transition(env_ret, policy)

    assert len(storage) == 1
    assert storage.get_size() == 1
    assert storage.size == 1

    data = storage.get_data()
    assert isinstance(data, TensorDict)
    assert len(data) == 1

    storage.clear()
    assert len(storage) == 0


def test_storage_add_transition_mismatched_env_ids_raises():
    """Test mismatched env ids raise ValueError."""
    storage = _make_storage(auto_truncate=True)
    env_ret = EnvRet(env_id="env-1", observation=torch.tensor(0))
    policy = PolicyResponse(env_id="env-2", action=torch.tensor(0))

    with pytest.raises(ValueError):
        storage.add_transition(env_ret, policy)


def test_storage_add_transition_updates_env_ts():
    """Test add_transition updates env timestamp."""
    storage = _make_storage(auto_truncate=False)

    env_ret = EnvRet(
        env_id="env-1",
        observation=torch.tensor(0),
        last_terminated=False,
        last_truncated=False,
    )
    policy = PolicyResponse(env_id="env-1", action=torch.tensor(1))

    before = env_ret.ts_env_sent_ns
    storage.add_transition(env_ret, policy)
    after = env_ret.ts_env_sent_ns

    assert after >= before  # I only figure out this validation way.


def test_storage_add_data_async_env_only_updates_and_ts():
    """Test async env-only updates timestamp without storing."""
    storage = _make_storage(auto_truncate=True)

    env_ret = EnvRet(
        env_id="env-1",
        observation=torch.tensor(0),
        last_terminated=True,
        last_truncated=False,
    )

    before = env_ret.ts_env_sent_ns
    storage.add_data_async(env_ret)
    after = env_ret.ts_env_sent_ns

    assert after >= before
    assert len(storage) == 0


def test_storage_add_data_async_pushes_on_policy_resp_and_rejects_invalid():
    """Test async flow stores on policy resp and rejects invalid."""
    storage = _make_storage(auto_truncate=True)

    env_ret = EnvRet(
        env_id="env-1",
        observation=torch.tensor(0),
        last_terminated=True,
        last_truncated=False,
    )
    policy = PolicyResponse(env_id="env-1", action=torch.tensor(5))

    storage.add_data_async(env_ret)
    assert len(storage) == 0

    storage.add_data_async(policy)
    assert len(storage) == 1
    assert storage[0]["obs"].shape[0] == 1

    with pytest.raises(TypeError):
        storage.add_data_async("not-a-valid-item")


def test_storage_add_episode_splits_multi_env():
    """Test add_episode splits vectorized episodes."""
    storage = _make_storage(auto_truncate=False, unit="episode")

    episode = {
        "obs": torch.tensor([[1, 2], [3, 4], [5, 6]]),
        "action": torch.tensor([[7, 8], [9, 10], [11, 12]]),
    }

    storage.add_episode(episode, num_envs=2)

    assert len(storage) == 2
    episodes = [storage[0], storage[1]]
    obs_values = [ep["obs"].tolist() for ep in episodes]
    assert sorted(obs_values) == sorted([[1, 3, 5], [2, 4, 6]])


def test_storage_add_episode_single_env_dict():
    """Test add_episode stores single-env dict."""
    storage = _make_storage(auto_truncate=False, unit="episode")

    episode = {
        "obs": torch.tensor([1, 2, 3]),
        "action": torch.tensor([4, 5, 6]),
    }

    storage.add_episode(episode, num_envs=1)

    assert len(storage) == 1
    assert storage[0]["obs"].tolist() == [1, 2, 3]


def test_storage_truncate_one_episode_manual_and_invalid_item():
    """Test manual truncate and invalid item handling."""
    storage = _make_storage(auto_truncate=False)

    env_ret = EnvRet(
        env_id="env-1",
        observation=torch.tensor(0),
        last_terminated=False,
        last_truncated=False,
    )
    policy = PolicyResponse(env_id="env-1", action=torch.tensor(1))
    storage.add_transition(env_ret, policy)

    storage.truncate_one_episode("env-1")
    assert len(storage) == 1
    assert storage[0]["obs"].shape[0] == 1

    class _NoEnvId:
        pass

    with pytest.raises(TypeError):
        storage.truncate_one_episode(_NoEnvId())


def test_storage_truncate_one_episode_object_with_env_id():
    """Test truncate_one_episode accepts objects with env_id."""
    storage = _make_storage(auto_truncate=False)

    env_ret = EnvRet(
        env_id="env-1",
        observation=torch.tensor(0),
        last_terminated=False,
        last_truncated=False,
    )
    policy = PolicyResponse(env_id="env-1", action=torch.tensor(1))
    storage.add_transition(env_ret, policy)

    class _HasEnvId:
        env_id = "env-1"

    storage.truncate_one_episode(_HasEnvId())

    assert len(storage) == 1


def test_storage_truncate_episodes_calls_truncate_one_episode():
    """Test truncate_episodes truncates each env id."""
    storage = _make_storage(auto_truncate=False)

    env_ret = EnvRet(
        env_id="env-1",
        observation=torch.tensor(0),
        last_terminated=False,
        last_truncated=False,
    )
    policy = PolicyResponse(env_id="env-1", action=torch.tensor(1))
    storage.add_transition(env_ret, policy)

    storage.truncate_episodes(["env-1"])  # only one but it is a list

    assert len(storage) == 1


def test_storage_print_timing_summary_resets():
    """Test print_timing_summary reset behavior."""
    storage = _make_storage(auto_truncate=False)
    storage.timing_raw = {"x": {"count": 1, "total": 1.0, "avg": 1.0}}

    storage.print_timing_summary(reset=True)

    assert storage.timing_raw == {}


def test_storage_print_timing_summary_logs_contents(caplog):
    """Test print_timing_summary logs timing details."""
    storage = _make_storage(auto_truncate=False)
    storage.timing_raw = {"transition_pair_to_buffer": {"count": 2, "total": 1.5, "avg": 0.75}}

    with caplog.at_level("DEBUG"):
        storage.print_timing_summary(reset=False)

    assert "Buffer storage timing:" in caplog.text
    assert "transition_pair_to_buffer" in caplog.text
    assert "count=2" in caplog.text
