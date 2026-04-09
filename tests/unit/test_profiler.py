"""
Tests for the profiler module.

This module contains unit tests for the profiler utilities including timing
context managers, decorators, and latency computation functions.
"""

import time
from unittest.mock import patch

import pytest

from rlightning.utils.profiler.profiler import record_timing, timer, timer_wrap


@pytest.fixture(autouse=True)
def _enable_profiler(monkeypatch):
    monkeypatch.setenv("RLIGHTNING_DEBUG", "1")


class TestSimpleTimer:
    """Test cases for the timer context manager."""

    def test_timer_basic_usage(self):
        """Test basic functionality of timer context manager."""
        timing_data = {}

        with timer("test_function", timing_data, level="info"):
            time.sleep(0.01)  # Sleep for 10ms

        # Verify timing data was recorded
        assert "test_function" in timing_data
        assert timing_data["test_function"]["count"] == 1
        assert timing_data["test_function"]["total"] > 0
        assert timing_data["test_function"]["avg"] > 0

    def test_timer_multiple_calls(self):
        """Test timer with multiple calls to same function."""
        timing_data = {}

        # First call
        with timer("test_function", timing_data, level="info"):
            time.sleep(0.01)

        # Second call
        with timer("test_function", timing_data, level="info"):
            time.sleep(0.02)

        # Verify cumulative statistics
        assert timing_data["test_function"]["count"] == 2
        assert timing_data["test_function"]["total"] > 0.03
        assert timing_data["test_function"]["avg"] > 0.015

    def test_timer_different_functions(self):
        """Test timer with different function names."""
        timing_data = {}

        with timer("func1", timing_data, level="info"):
            time.sleep(0.01)

        with timer("func2", timing_data, level="debug"):
            time.sleep(0.01)

        # Verify separate entries for different functions
        assert "func1" in timing_data
        assert "func2" in timing_data
        assert timing_data["func1"]["count"] == 1
        assert timing_data["func2"]["count"] == 1


class TestSimpleTimerWrap:
    """Test cases for the timer_wrap decorator."""

    def test_timer_wrap_basic_usage(self):
        """Test basic functionality of timer_wrap decorator."""

        class TestClass:
            timing_raw: dict = {}

            @timer_wrap(name="test_method", level="info")
            def test_method(self):
                time.sleep(0.01)
                return "result"

        obj = TestClass()
        result = obj.test_method()

        # Verify return value
        assert result == "result"

        # Verify timing data was recorded
        assert hasattr(obj, "timing_raw")
        assert "test_method" in obj.timing_raw
        assert obj.timing_raw["test_method"]["count"] == 1
        assert obj.timing_raw["test_method"]["total"] > 0

    def test_timer_wrap_default_name(self):
        """Test timer_wrap using default function name."""

        class TestClass:
            timing_raw: dict = {}

            @timer_wrap()
            def my_function(self):
                time.sleep(0.01)
                return 42

        obj = TestClass()
        result = obj.my_function()

        # Verify return value
        assert result == 42

        # Verify timing data was recorded with function name
        assert "my_function" in obj.timing_raw

    def test_timer_wrap_multiple_methods(self):
        """Test timer_wrap with multiple decorated methods."""

        class TestClass:
            timing_raw: dict = {}

            @timer_wrap(name="method1")
            def method1(self):
                time.sleep(0.01)
                return 1

            @timer_wrap(name="method2")
            def method2(self):
                time.sleep(0.02)
                return 2

        obj = TestClass()
        obj.method1()
        obj.method2()

        # Verify both methods are tracked separately
        assert "method1" in obj.timing_raw
        assert "method2" in obj.timing_raw
        assert obj.timing_raw["method1"]["count"] == 1
        assert obj.timing_raw["method2"]["count"] == 1

    def test_timer_wrap_multiple_calls(self):
        """Test timer_wrap with multiple calls to same method."""

        class TestClass:
            timing_raw: dict = {}

            @timer_wrap(name="repeat_method")
            def repeat_method(self):
                time.sleep(0.01)
                return "done"

        obj = TestClass()

        # Call method multiple times
        for _ in range(3):
            obj.repeat_method()

        # Verify cumulative statistics
        assert obj.timing_raw["repeat_method"]["count"] == 3
        assert obj.timing_raw["repeat_method"]["total"] > 0.03


class TestIntegration:
    """Integration tests combining multiple profiler functions."""

    def test_timer_and_record_timing_compatibility(self):
        """Test that timer and record_timing use compatible data structures."""
        timing_data = {}

        # Use timer
        with timer("context_timer", timing_data):
            time.sleep(0.01)

        # Use record_timing
        record_timing("manual_timing", 0.02, timing_data)

        # Verify both use same structure
        assert "context_timer" in timing_data
        assert "manual_timing" in timing_data

        for key in timing_data:
            assert "count" in timing_data[key]
            assert "total" in timing_data[key]
            assert "avg" in timing_data[key]

    def test_decorator_and_context_manager_compatibility(self):
        """Test that decorator and context manager produce compatible data."""

        class TestClass:
            timing_raw: dict = {}

            @timer_wrap(name="decorated_method")
            def decorated_method(self):
                time.sleep(0.01)
                return "done"

        obj = TestClass()
        obj.decorated_method()

        timing_data = {}
        with timer("context_method", timing_data):
            time.sleep(0.01)

        # Verify both produce same data structure
        assert "count" in obj.timing_raw["decorated_method"]  # type: ignore
        assert "total" in obj.timing_raw["decorated_method"]  # type: ignore
        assert "avg" in obj.timing_raw["decorated_method"]  # type: ignore

        assert "count" in timing_data["context_method"]
        assert "total" in timing_data["context_method"]
        assert "avg" in timing_data["context_method"]


if __name__ == "__main__":
    pytest.main([__file__])
