"""Unit tests for EpisodeTable."""

import pytest

from rlightning.buffer.utils.table import EpisodeTable

# TODO: add test for new functions


def test_episode_table_init_requires_positive_storages():
    """EpisodeTable should require num_storages >= 1."""
    with pytest.raises(ValueError):
        EpisodeTable(num_storages=0)


def test_episode_table_register_envs_balances_round_robin():
    """EpisodeTable should assign envs to lowest-load shard with RR tie-break."""
    table = EpisodeTable(num_storages=2)
    table.register_envs(["env-a", "env-b", "env-c"])

    assert table.get_storage_idx_for_env("env-a") == 0
    assert table.get_storage_idx_for_env("env-b") == 1
    assert table.get_storage_idx_for_env("env-c") == 0


def test_episode_table_get_storage_idx_is_stable_for_existing_env():
    """get_storage_idx should return the same shard for an existing env."""
    table = EpisodeTable(num_storages=3)
    idx_first = table.get_storage_idx_for_env("env-x")
    idx_second = table.get_storage_idx_for_env("env-x")

    assert idx_first == idx_second
    assert table._storage_env_count[idx_first] == 1


def test_episode_table_envs_for_storage_lists_envs():
    """envs_for_storage should list env ids assigned to a shard."""
    table = EpisodeTable(num_storages=2, env_ids=["env-1", "env-2", "env-3"])

    envs0 = table.get_envs_for_storage(0)
    envs1 = table.get_envs_for_storage(1)

    assert set(envs0) == {"env-1", "env-3"}
    assert set(envs1) == {"env-2"}


def test_episode_table_envs_for_storage_validates_index():
    """envs_for_storage should validate storage index."""
    table = EpisodeTable(num_storages=2)

    with pytest.raises(IndexError):
        _ = table.get_envs_for_storage(2)
