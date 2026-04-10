"""Unit tests for EnvGroup.get_env_stats."""

import pytest

from rlightning.env.env_group import EnvGroup

class _DummyEnv:
    def __init__(self, stats):
        self._stats = stats
        self.reset_flags = []
        self.finish_rollout_called = 0

    def get_env_stats(self, reset=False):
        self.reset_flags.append(reset)
        return self._stats

    def finish_rollout(self):
        self.finish_rollout_called += 1


class _DummySubmitter:
    def submit(self, fn, *args, _block: bool = False, **kwargs):
        return fn(*args, **kwargs)


def _make_env_group(env_list, env_servers):
    env_group = EnvGroup.__new__(EnvGroup)
    env_group.env_list = env_list
    env_group.env_servers = env_servers
    env_group._task_submitter = _DummySubmitter()
    return env_group


def test_get_env_stats_aggregates_means_across_envs_and_servers():
    env_local = _DummyEnv(stats={"reward": [3.0, 2], "success": [1.0, 1]})
    env_server = _DummyEnv(stats={"reward": [5.0, 2], "fail": [2.0, 4]})
    env_group = _make_env_group([env_local], [env_server])

    stats = env_group.get_env_stats(reset=True)

    assert stats["reward"] == pytest.approx(2.0)  # (3 + 5) / (2 + 2)
    assert stats["success"] == pytest.approx(1.0)  # 1 / 1
    assert stats["fail"] == pytest.approx(0.5)  # 2 / 4
    assert env_local.reset_flags == [True]
    assert env_server.reset_flags == [True]
    assert env_local.finish_rollout_called == 1
    assert env_server.finish_rollout_called == 1
