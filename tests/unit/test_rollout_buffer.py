"""Unit tests for RolloutBuffer."""

import pytest

from rlightning.buffer.base_buffer import DataBuffer
from rlightning.buffer.rollout_buffer import RolloutBuffer
from rlightning.buffer.sampler import AllDataSampler, UniformSampler
from rlightning.utils.config import BufferConfig

def _make_rollout_config(sampler_type: str = "all") -> BufferConfig:
    return BufferConfig.from_dict(
        {
            "type": "RolloutBuffer",
            "capacity": 8,
            "sampler": {"type": sampler_type},
            "storage": {"type": "unified", "device": "cpu", "unit": "transition"},
        }
    )


def test_rollout_buffer_sample_raises_when_batch_too_large():
    """RolloutBuffer should reject batch_size larger than buffer size."""
    buf = RolloutBuffer(_make_rollout_config())
    buf.size = lambda: 2

    with pytest.raises(ValueError):
        buf.sample(batch_size=3)


def test_rollout_buffer_sample_allows_equal_batch_size(monkeypatch):
    """RolloutBuffer should allow batch_size equal to buffer size."""
    buf = RolloutBuffer(_make_rollout_config())
    buf.size = lambda: 2

    called = {}

    def _fake_sample(self, batch_size, shuffle=True, drop_last=True):
        called["args"] = (batch_size, shuffle, drop_last)
        return ["sampled"]

    monkeypatch.setattr(DataBuffer, "sample", _fake_sample)

    cleared = {"flag": False}

    def _fake_clear():
        cleared["flag"] = True

    buf.clear = _fake_clear

    out = buf.sample(batch_size=2, shuffle=False, drop_last=False)
    assert out == ["sampled"]
    assert called["args"] == (2, False, False)
    assert cleared["flag"] is True


def test_rollout_buffer_sample_logs_warning_and_clears(caplog, monkeypatch):
    """RolloutBuffer should warn on partial sampling and clear after sampling."""
    buf = RolloutBuffer(_make_rollout_config(sampler_type="uniform"))
    buf.sampler = UniformSampler()
    buf.size = lambda: 5

    called = {}

    def _fake_sample(self, batch_size, shuffle=True, drop_last=True):
        called["args"] = (batch_size, shuffle, drop_last)
        return ["sampled"]

    monkeypatch.setattr(DataBuffer, "sample", _fake_sample)

    cleared = {"flag": False}

    def _fake_clear():
        cleared["flag"] = True

    buf.clear = _fake_clear

    with caplog.at_level("WARNING"):
        out = buf.sample(batch_size=3, shuffle=False, drop_last=False)

    assert out == ["sampled"]
    assert called["args"] == (3, False, False)
    assert cleared["flag"] is True
    assert "remaining unsampled data will be discarded" in caplog.text


def test_rollout_buffer_sample_no_warning_with_all_sampler(caplog, monkeypatch):
    """AllDataSampler should avoid partial-sampling warnings."""
    buf = RolloutBuffer(_make_rollout_config(sampler_type="all"))
    buf.sampler = AllDataSampler()
    buf.size = lambda: 5

    def _fake_sample(self, batch_size, shuffle=True, drop_last=True):
        return ["sampled"]

    monkeypatch.setattr(DataBuffer, "sample", _fake_sample)

    buf.clear = lambda: None

    with caplog.at_level("WARNING"):
        _ = buf.sample(batch_size=3)

    assert "remaining unsampled data will be discarded" not in caplog.text


def test_rollout_buffer_sample_allows_none_batch_size(caplog, monkeypatch):
    """RolloutBuffer should accept batch_size=None without warning."""
    buf = RolloutBuffer(_make_rollout_config(sampler_type="all"))
    buf.sampler = AllDataSampler()
    buf.size = lambda: 5

    called = {}

    def _fake_sample(self, batch_size, shuffle=True, drop_last=True):
        called["args"] = (batch_size, shuffle, drop_last)
        return ["sampled"]

    monkeypatch.setattr(DataBuffer, "sample", _fake_sample)

    buf.clear = lambda: None

    with caplog.at_level("WARNING"):
        out = buf.sample(batch_size=None, shuffle=False, drop_last=False)

    assert out == ["sampled"]
    assert called["args"] == (None, False, False)
    assert "remaining unsampled data will be discarded" not in caplog.text
