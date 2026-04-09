"""Unit tests for BasePolicy behaviors."""

from __future__ import annotations

import asyncio
import builtins
import threading
from typing import Any, Dict

import numpy as np
import pytest
import torch
from torch import nn

from rlightning.policy.base_policy import BasePolicy, PolicyRole
from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.config import PolicyConfig, TrainConfig
from rlightning.utils.utils import InternalFlag


@pytest.fixture(autouse=True)
def _cpu_safe_cuda(monkeypatch):
    if torch.cuda.is_available():
        return

    monkeypatch.setattr(EnvRet, "cuda", lambda self, device="cuda": self, raising=False)
    monkeypatch.setattr(PolicyResponse, "cuda", lambda self, device="cuda": self, raising=False)


class _DummyPolicy(BasePolicy):
    """Concrete policy for exercising BasePolicy behaviors."""

    def __init__(self, config: PolicyConfig, role_type: PolicyRole) -> None:
        super().__init__(config, role_type)
        self.update_weights_called = False
        self.train_called = False
        self.loaded_state = None

    def construct_network(self, env_meta: Any = None, *args: Any, **kwargs: Any) -> None:
        self.linear = nn.Linear(2, 2)
        self.frozen = nn.Linear(2, 2)
        for param in self.frozen.parameters():
            param.requires_grad = False

    def setup_optimizer(self, optim_cfg: Any) -> None:
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def rollout_step(self, env_ret: EnvRet, **kwargs: Any) -> PolicyResponse:
        return PolicyResponse(env_id=env_ret.env_id, action=torch.tensor(1))

    def postprocess(self, env_ret: EnvRet | None = None, policy_resp: PolicyResponse | None = None) -> Any:
        return (env_ret, policy_resp, torch.tensor([1.0]))

    def update_dataset(self, data: Any) -> None:
        self.dataset = data

    def train(self, *args: Any, **kwargs: Any) -> Any:
        self.train_called = True
        return None

    def get_trainable_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        params: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, model in self.model_list:
            module = model.module if hasattr(model, "module") else model
            params[name] = module.state_dict()
        return params

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> None:
        self.loaded_state = state_dict

    def update_weights(self) -> None:
        self.update_weights_called = True
        return None


class _AsyncDummyPolicy(_DummyPolicy):
    """Async variant for testing _rollout_async path."""

    async def rollout_step(self, env_ret: EnvRet, **kwargs: Any) -> PolicyResponse:
        return PolicyResponse(env_id=env_ret.env_id, action=torch.tensor(2))


class _WarmupPolicy(_DummyPolicy):
    """Policy override to observe warmup rebuild behavior."""

    def init_train(self, train_config: TrainConfig, env_meta: Any = None) -> None:
        self.init_train_called = True
        self.is_init = True

    def _setup_sampling_params(self) -> None:
        self.setup_sampling_called = True


class _FakeLoop:
    """Minimal loop stub for call_soon_threadsafe in hook tests."""

    def __init__(self) -> None:
        self.calls = []

    def call_soon_threadsafe(self, func, *args):
        self.calls.append(func)
        func(*args)


class _FailingRolloutPolicy(_DummyPolicy):
    """Policy that raises in rollout_step."""

    def rollout_step(self, env_ret: EnvRet, **kwargs: Any) -> PolicyResponse:
        raise RuntimeError("test rollout error")


class _FailingPostprocessPolicy(_DummyPolicy):
    """Policy that raises in postprocess."""

    def postprocess(self, env_ret: EnvRet | None = None, policy_resp: PolicyResponse | None = None) -> Any:
        raise RuntimeError("test postprocess error")


def _make_policy_config(rollout_mode: str = "sync") -> PolicyConfig:
    return PolicyConfig.from_dict(
        {
            "type": "DummyPolicy",
            "rollout_mode": rollout_mode,
        }
    )


def test_sanity_check_invalid_role_raises():
    """_sanity_check should reject unsupported role_type values."""
    config = _make_policy_config()
    with pytest.raises(ValueError):
        _ = _DummyPolicy(config, role_type="bad-role")


def test_find_model_selects_trainable_modules_only():
    """_find_model should include only modules with requires_grad params."""
    policy = _DummyPolicy(_make_policy_config(), PolicyRole.TRAIN)
    policy.construct_network()
    policy._find_model()

    names = [name for name, _ in policy.model_list]
    assert "linear" in names
    assert "frozen" not in names


def test_find_model_raises_on_invalid_model_list_entry():
    """_find_model should reject non-nn.Module entries in model_list."""
    policy = _DummyPolicy(_make_policy_config(), PolicyRole.TRAIN)
    policy.model_list = [("bad", object())]

    with pytest.raises(ValueError):
        policy._find_model()


def test_init_eval_sync_sets_idle_event_and_eval_mode():
    """init_eval should set idle event and switch models to eval mode."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy.init_eval(env_meta=None)

    assert policy.is_init is True
    assert policy._idle_as_infer.is_set() is True
    assert policy.linear.training is False


def test_init_eval_async_sets_async_fields():
    """init_eval should initialize async admission controls and counters."""
    policy = _DummyPolicy(_make_policy_config("async"), PolicyRole.EVAL)
    loop = asyncio.new_event_loop()

    try:
        try:
            old_loop = asyncio.get_event_loop()
        except RuntimeError:
            old_loop = None

        asyncio.set_event_loop(loop)
        policy.init_eval(env_meta=None)

        assert policy.is_init is True
        assert policy._loop is loop
        assert policy._accept_new_requests.is_set() is True
        assert policy.num_requests == 0
        assert isinstance(policy._num_requests_lock, type(threading.Lock()))
        assert isinstance(policy._inflight_zero_cv, threading.Condition)
        assert policy._update_weights_signal.is_set() is False
    finally:
        asyncio.set_event_loop(old_loop)
        loop.close()


def test_init_eval_invalid_rollout_mode_raises():
    """init_eval should reject unknown rollout_mode."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy.rollout_mode = "invalid"

    with pytest.raises(ValueError):
        policy.init_eval(env_meta=None)


def test_init_train_sets_train_mode_and_optimizer():
    """init_train should enable train mode and initialize optimizer."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.TRAIN)
    train_cfg = TrainConfig(max_epochs=1)

    policy.init_train(train_cfg)

    assert policy.is_init is True
    assert policy.linear.training is True
    assert hasattr(policy, "optimizer")


def test_is_initialized_reflects_state():
    """is_initialized should reflect init_train/init_eval state."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.TRAIN)

    assert policy.is_initialized() is False

    policy.init_train(TrainConfig(max_epochs=1))

    assert policy.is_initialized() is True


def test_rollout_sync_clears_and_sets_idle_event():
    """_rollout should clear idle flag during execution and set it afterward."""

    class _StateCapturingPolicy(_DummyPolicy):
        """Policy that captures idle state during rollout execution."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.idle_during_rollout = None

        def rollout_step(self, env_ret, **kwargs):
            # Capture idle state during rollout execution
            self.idle_during_rollout = self._idle_as_infer.is_set()
            return super().rollout_step(env_ret, **kwargs)

    policy = _StateCapturingPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy._idle_as_infer = threading.Event()
    policy._idle_as_infer.set()

    env_ret = EnvRet(env_id="env-1", observation=np.array([1.0]))
    resp = policy._rollout(env_ret)

    # Verify idle was cleared during rollout execution
    assert policy.idle_during_rollout is False, "idle should be cleared during rollout"

    # Verify idle is set after rollout completes
    assert policy._idle_as_infer.is_set() is True, "idle should be set after rollout"

    # Verify response is correct
    assert resp.env_id == "env-1"


@pytest.mark.parametrize(
    "remote_storage,remote_env",
    [
        ("1", "0"),  # Only REMOTE_STORAGE
        ("0", "1"),  # Only REMOTE_ENV
        ("1", "1"),  # Both flags set
    ],
    ids=["storage_only", "env_only", "both"],
)
def test_post_rollout_hook_converts_when_remote(monkeypatch, remote_storage, remote_env):
    """_post_rollout_hook should call numpy on response for any remote flag."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy._idle_as_infer = threading.Event()

    policy_resp = PolicyResponse(env_id="env-4", action=torch.tensor(1))
    numpy_called = {"called": False}

    def _numpy():
        numpy_called["called"] = True
        return policy_resp

    policy_resp.numpy = _numpy

    monkeypatch.setenv("RLIGHTNING_REMOTE_STORAGE", remote_storage)
    monkeypatch.setenv("RLIGHTNING_REMOTE_ENV", remote_env)

    result = policy._post_rollout_hook(policy_resp)

    assert (
        numpy_called["called"] is True
    ), f"numpy() should be called when REMOTE_STORAGE={remote_storage}, REMOTE_ENV={remote_env}"
    assert result is policy_resp
    assert policy._idle_as_infer.is_set() is True


def test_rollout_async_tracks_inflight_requests():
    """_rollout_async should increment then decrement inflight count."""

    async def _run():
        class _StateCapturingAsyncPolicy(_AsyncDummyPolicy):
            """Policy that captures num_requests during rollout execution."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.num_requests_during_rollout = None

            async def rollout_step(self, env_ret, **kwargs):
                # Capture num_requests during rollout execution
                self.num_requests_during_rollout = self.num_requests
                return await super().rollout_step(env_ret, **kwargs)

        policy = _StateCapturingAsyncPolicy(_make_policy_config("async"), PolicyRole.EVAL)
        policy.rollout_mode = "async"
        policy._accept_new_requests = asyncio.Event()
        policy._accept_new_requests.set()
        policy.num_requests = 0
        policy._num_requests_lock = threading.Lock()
        policy._inflight_zero_cv = threading.Condition(policy._num_requests_lock)

        env_ret = EnvRet(env_id="env-2", observation=np.array([0.0]))
        resp = await policy._rollout_async(env_ret)
        return resp, policy.num_requests, policy.num_requests_during_rollout

    resp, num_requests_after, num_requests_during = asyncio.run(_run())

    # Verify num_requests was incremented during rollout
    assert num_requests_during == 1, "num_requests should be 1 during rollout"

    # Verify num_requests is decremented after rollout
    assert num_requests_after == 0, "num_requests should be 0 after rollout"

    # Verify response is correct
    assert resp.env_id == "env-2"


@pytest.mark.parametrize(
    "initial_count,should_notify",
    [
        (1, True),  # 1 → 0, should notify
        (2, False),  # 2 → 1, should NOT notify
        (3, False),  # 3 → 2, should NOT notify
    ],
    ids=["reaches_zero", "still_one", "still_two"],
)
def test_post_rollout_hook_async_notify_behavior(initial_count, should_notify):
    """_post_rollout_hook_async should only notify when reaching zero."""

    class _NotifyCounter:
        def __init__(self) -> None:
            self.called = False

        def notify_all(self) -> None:
            self.called = True

    policy = _DummyPolicy(_make_policy_config("async"), PolicyRole.EVAL)
    policy.num_requests = initial_count
    policy._num_requests_lock = threading.Lock()
    notifier = _NotifyCounter()
    policy._inflight_zero_cv = notifier

    policy_resp = PolicyResponse(env_id="env-notify", action=torch.tensor(1))
    result = policy._post_rollout_hook_async(policy_resp)

    assert policy.num_requests == initial_count - 1
    assert notifier.called is should_notify, (
        f"notify_all should{' ' if should_notify else ' NOT '}be called "
        f"when num_requests goes from {initial_count} to {initial_count - 1}"
    )
    assert result is policy_resp


def test_update_weight_hooks_sync_manage_events():
    """_pre/_post_update_weights_hook should manage sync events properly."""

    class _TrackingEvent:
        """Event wrapper that tracks wait() calls."""

        def __init__(self, name):
            self._event = threading.Event()
            self._event.set()
            self._name = name
            self.wait_calls = []

        def wait(self):
            self.wait_calls.append(self._name)
            return self._event.wait()

        def set(self):
            self._event.set()

        def clear(self):
            self._event.clear()

        def is_set(self):
            return self._event.is_set()

    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy.rollout_mode = "sync"
    update_signal = _TrackingEvent("update_signal")
    idle_event = _TrackingEvent("idle")
    policy._update_weights_signal = update_signal
    policy._idle_as_infer = idle_event
    policy._weight_update_done = None

    policy._pre_update_weights_hook()

    # Verify both signals were waited on in correct order
    assert "update_signal" in update_signal.wait_calls, "Should wait for update signal"
    assert "idle" in idle_event.wait_calls, "Should wait for idle"

    # Verify idle is cleared (blocks rollout during weight update)
    assert policy._idle_as_infer.is_set() is False

    policy._post_update_weights_hook()

    # Verify idle is restored (allows rollout)
    assert policy._idle_as_infer.is_set() is True
    # Verify update signal is cleared
    assert policy._update_weights_signal.is_set() is False


def test_update_weight_hooks_async_manage_events():
    """_pre/_post_update_weights_hook should manage async admission events."""

    class _TrackingEvent:
        """Event wrapper that tracks wait() calls."""

        def __init__(self, name):
            self._event = threading.Event()
            self._event.set()
            self._name = name
            self.wait_calls = []

        def wait(self):
            self.wait_calls.append(self._name)
            return self._event.wait()

        def set(self):
            self._event.set()

        def clear(self):
            self._event.clear()

        def is_set(self):
            return self._event.is_set()

    policy = _DummyPolicy(_make_policy_config("async"), PolicyRole.EVAL)
    policy.rollout_mode = "async"

    update_signal = _TrackingEvent("update_signal")
    policy._update_weights_signal = update_signal
    policy._weight_update_done = None

    policy._accept_new_requests = asyncio.Event()
    policy._accept_new_requests.set()
    policy.num_requests = 0
    policy._num_requests_lock = threading.Lock()
    policy._inflight_zero_cv = threading.Condition(policy._num_requests_lock)
    policy._loop = _FakeLoop()

    policy._pre_update_weights_hook()

    # Verify update signal was waited on
    assert "update_signal" in update_signal.wait_calls, "Should wait for update signal"

    # Verify call_soon_threadsafe was used to clear accept_new_requests
    assert len(policy._loop.calls) >= 1, "Should call call_soon_threadsafe"

    # Verify accept_new_requests is cleared (stops accepting new requests)
    assert policy._accept_new_requests.is_set() is False

    policy._post_update_weights_hook()

    # Verify call_soon_threadsafe was used to set accept_new_requests
    assert len(policy._loop.calls) >= 2, "Should call call_soon_threadsafe for set"

    # Verify accept_new_requests is restored (allows new requests)
    assert policy._accept_new_requests.is_set() is True
    # Verify update signal is cleared
    assert policy._update_weights_signal.is_set() is False


def test_postprocess_sync_handles_remote_and_sets_idle(monkeypatch):
    """_postprocess should convert outputs for remote and restore idle flag."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy._idle_as_infer = threading.Event()
    policy._idle_as_infer.set()

    monkeypatch.setenv("RLIGHTNING_REMOTE_STORAGE", "1")

    env_ret = EnvRet(env_id="env-3", observation=np.array([1.0]))
    env_cuda_called = {"called": False}
    env_numpy_called = {"called": False}

    def _env_cuda(device=None):  # noqa: ARG001
        env_cuda_called["called"] = True
        return env_ret

    def _env_numpy():
        env_numpy_called["called"] = True
        return env_ret

    env_ret.cuda = _env_cuda
    env_ret.numpy = _env_numpy

    policy_resp = PolicyResponse(env_id="env-3", action=torch.tensor(1))
    resp_cuda_called = {"called": False}
    resp_numpy_called = {"called": False}

    def _resp_cuda(device=None):  # noqa: ARG001
        resp_cuda_called["called"] = True
        return policy_resp

    def _resp_numpy():
        resp_numpy_called["called"] = True
        return policy_resp

    policy_resp.cuda = _resp_cuda
    policy_resp.numpy = _resp_numpy

    result = policy._postprocess(env_ret=env_ret, policy_resp=policy_resp)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert env_cuda_called["called"] is True
    assert resp_cuda_called["called"] is True
    assert env_numpy_called["called"] is True
    assert resp_numpy_called["called"] is True
    assert isinstance(result[2], np.ndarray)
    assert policy._idle_as_infer.is_set() is True


def test_check_idle_sync_allows_eval_and_async_asserts():
    """check_idle should work for sync eval and assert for async mode."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy._idle_as_infer = threading.Event()
    policy._idle_as_infer.set()

    policy.check_idle()

    policy.rollout_mode = "async"
    with pytest.raises(AssertionError):
        policy.check_idle()


def test_get_num_requests_defaults_to_zero():
    """get_num_requests should return 0 when num_requests is None."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy.num_requests = None

    assert policy.get_num_requests() == 0


def test_reset_training_state_rebuilds_and_clears():
    """reset_training_state should clear internal state and rebuild."""
    policy = _WarmupPolicy(_make_policy_config("sync"), PolicyRole.TRAIN)
    policy.dataset = "data"
    policy.optimizer = object()
    policy.optimizer_steps = 5
    policy.model_list = [("x", nn.Linear(1, 1))]
    policy._modules["rsl_rl/test"] = nn.Linear(1, 1)

    train_cfg = TrainConfig(max_epochs=1)
    policy.reset_training_state(train_cfg, seed=42)

    assert policy.dataset is None
    assert policy.optimizer is None
    assert policy.optimizer_steps == 0
    assert policy.model_list == []
    assert "rsl_rl/test" not in policy._modules
    assert policy.init_train_called is True
    assert policy.setup_sampling_called is True


def test_save_weights_writes_state_dict(tmp_path):
    """save_weights should write model state dicts to disk."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.TRAIN)
    policy.construct_network()
    policy._find_model()

    ckpt_path = tmp_path / "epoch_1" / "model.pt"
    policy.save_checkpoint(ckpt_path)

    assert ckpt_path.exists()
    state = torch.load(ckpt_path)
    assert "linear" in state


def test_print_timing_summary_logs_and_resets(caplog):
    """print_timing_summary should log timing entries and reset when requested."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy.timing_raw = {"rollout": {"count": 1, "total": 1.0, "avg": 1.0}}

    with caplog.at_level("DEBUG"):
        policy.print_timing_summary(reset=True)

    assert "Policy" in caplog.text
    assert "rollout" in caplog.text
    assert policy.timing_raw == {}


def test_notify_update_weights_sets_signal():
    """notify_update_weights should set the update signal event."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy._update_weights_signal = threading.Event()
    policy._weight_update_done = None

    policy.notify_update_weights()

    assert policy._update_weights_signal.is_set() is True


# ============================================================================
# P0 - Core functionality tests for update_weights
# ============================================================================


def test_update_weights_calls_hooks_and_buffer(monkeypatch):
    """update_weights should call pre hook, buffer update, and post hook."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy.rollout_mode = "sync"
    policy._update_weights_signal = threading.Event()
    policy._update_weights_signal.set()
    policy._idle_as_infer = threading.Event()
    policy._idle_as_infer.set()
    policy._weight_update_done = None
    policy.timing_raw = {}

    call_order = []
    call_count = {"pre": 0}

    original_pre_hook = policy._pre_update_weights_hook

    def mock_pre_hook():
        call_count["pre"] += 1
        call_order.append("pre")
        if call_count["pre"] >= 2:
            raise KeyboardInterrupt  # Terminate loop
        original_pre_hook()

    def mock_update_buffer():
        call_order.append("buffer")

    original_post_hook = policy._post_update_weights_hook

    def mock_post_hook():
        call_order.append("post")
        original_post_hook()
        # Re-set the signal for next iteration
        policy._update_weights_signal.set()

    monkeypatch.setattr(policy, "_pre_update_weights_hook", mock_pre_hook)
    monkeypatch.setattr(policy, "update_weights_from_buffer", mock_update_buffer)
    monkeypatch.setattr(policy, "_post_update_weights_hook", mock_post_hook)

    with pytest.raises(KeyboardInterrupt):
        BasePolicy.update_weights(policy)

    assert call_order == ["pre", "buffer", "post", "pre"]


def test_update_weights_handles_exception_gracefully(monkeypatch):
    """update_weights should catch exceptions from update_weights_from_buffer and continue."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy.rollout_mode = "sync"
    policy._update_weights_signal = threading.Event()
    policy._update_weights_signal.set()
    policy._idle_as_infer = threading.Event()
    policy._idle_as_infer.set()
    policy._weight_update_done = None
    policy.timing_raw = {}

    call_order = []
    call_count = {"pre": 0}

    original_pre_hook = policy._pre_update_weights_hook

    def mock_pre_hook():
        call_count["pre"] += 1
        call_order.append("pre")
        if call_count["pre"] >= 2:
            raise KeyboardInterrupt  # Terminate loop after second iteration
        original_pre_hook()

    def mock_update_buffer_raises():
        call_order.append("buffer_error")
        raise ValueError("test buffer error")

    original_post_hook = policy._post_update_weights_hook

    def mock_post_hook():
        call_order.append("post")
        original_post_hook()
        policy._update_weights_signal.set()

    monkeypatch.setattr(policy, "_pre_update_weights_hook", mock_pre_hook)
    monkeypatch.setattr(policy, "update_weights_from_buffer", mock_update_buffer_raises)
    monkeypatch.setattr(policy, "_post_update_weights_hook", mock_post_hook)

    with pytest.raises(KeyboardInterrupt):
        BasePolicy.update_weights(policy)

    # Exception was caught, post hook still called (finally block)
    assert call_order == ["pre", "buffer_error", "post", "pre"]


# ============================================================================
# P1 - Exception handling tests for rollout and postprocess
# ============================================================================


def test_rollout_logs_and_reraises_exception():
    """_rollout should log and re-raise exceptions from rollout_step."""
    policy = _FailingRolloutPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy._idle_as_infer = threading.Event()
    policy._idle_as_infer.set()

    env_ret = EnvRet(env_id="env-fail", observation=np.array([1.0]))

    with pytest.raises(RuntimeError, match="test rollout error"):
        policy._rollout(env_ret)


def test_rollout_async_logs_and_reraises_exception():
    """_rollout_async should log and re-raise exceptions from rollout_step."""

    class _FailingAsyncRolloutPolicy(_DummyPolicy):
        """Async policy that raises in rollout_step."""

        async def rollout_step(self, env_ret: EnvRet, **kwargs: Any) -> PolicyResponse:  # noqa: ARG002
            raise RuntimeError("test async rollout error")

    async def _run():
        policy = _FailingAsyncRolloutPolicy(_make_policy_config("async"), PolicyRole.EVAL)
        policy.rollout_mode = "async"
        policy._accept_new_requests = asyncio.Event()
        policy._accept_new_requests.set()
        policy.num_requests = 0
        policy._num_requests_lock = threading.Lock()
        policy._inflight_zero_cv = threading.Condition(policy._num_requests_lock)

        env_ret = EnvRet(env_id="env-fail-async", observation=np.array([0.0]))
        await policy._rollout_async(env_ret)

    with pytest.raises(RuntimeError, match="test async rollout error"):
        asyncio.run(_run())


def test_postprocess_logs_and_reraises_exception():
    """_postprocess should log and re-raise exceptions from postprocess."""
    policy = _FailingPostprocessPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy._idle_as_infer = threading.Event()
    policy._idle_as_infer.set()

    env_ret = EnvRet(env_id="env-pp-fail", observation=np.array([1.0]))
    policy_resp = PolicyResponse(env_id="env-pp-fail", action=torch.tensor(1))

    with pytest.raises(RuntimeError, match="test postprocess error"):
        policy._postprocess(env_ret=env_ret, policy_resp=policy_resp)


# ============================================================================
# P2 - Enhanced coverage tests
# ============================================================================


def test_pre_rollout_hook_records_timing_when_debug(monkeypatch):
    """_pre_rollout_hook should record timing when DEBUG flag is set."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy._idle_as_infer = threading.Event()
    policy._idle_as_infer.set()
    policy.timing_raw = {}

    # Enable DEBUG flag
    monkeypatch.setenv("RLIGHTNING_DEBUG", "1")

    env_ret = EnvRet(env_id="env-debug", observation=np.array([1.0]))

    record_called = {"called": False}

    def mock_record_timing(name, _value, _timing_dict, level="debug"):  # noqa: ARG001
        record_called["called"] = True
        assert name == "transition_env_to_policy"

    monkeypatch.setattr("rlightning.policy.base_policy.profiler.record_timing", mock_record_timing)

    policy._pre_rollout_hook(env_ret)

    assert record_called["called"] is True


def test_pre_rollout_hook_async_records_timing_when_debug(monkeypatch):
    """_pre_rollout_hook_async should record timing when DEBUG flag is set."""

    async def _run():
        policy = _AsyncDummyPolicy(_make_policy_config("async"), PolicyRole.EVAL)
        policy.rollout_mode = "async"
        policy._accept_new_requests = asyncio.Event()
        policy._accept_new_requests.set()
        policy.num_requests = 0
        policy._num_requests_lock = threading.Lock()
        policy.timing_raw = {}

        env_ret = EnvRet(env_id="env-debug-async", observation=np.array([0.0]))
        env_ret.ts_env_sent_ns = 0

        record_called = {"called": False}

        def mock_record_timing(name, _value, _timing_dict, level="debug"):  # noqa: ARG001
            record_called["called"] = True
            assert name == "transition_env_to_policy"

        monkeypatch.setattr("rlightning.policy.base_policy.profiler.record_timing", mock_record_timing)

        await policy._pre_rollout_hook_async(env_ret)

        return record_called["called"]

    # Enable DEBUG flag
    monkeypatch.setenv("RLIGHTNING_DEBUG", "1")

    record_called = asyncio.run(_run())
    assert record_called is True


def test_get_num_requests_returns_actual_count():
    """get_num_requests should return actual count when num_requests is set."""
    policy = _DummyPolicy(_make_policy_config("async"), PolicyRole.EVAL)
    policy.num_requests = 5

    assert policy.get_num_requests() == 5


def test_post_postprocess_hook_handles_list_result(monkeypatch):
    """_post_postprocess_hook should convert list results when remote."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy._idle_as_infer = threading.Event()

    # Enable remote flag
    monkeypatch.setenv("RLIGHTNING_REMOTE_STORAGE", "1")

    numpy_calls = {"count": 0}

    def make_mock_numpy(original):
        def _numpy():
            numpy_calls["count"] += 1
            return original

        return _numpy

    env_ret1 = EnvRet(env_id="env-list-1", observation=np.array([1.0]))
    env_ret1.numpy = make_mock_numpy(env_ret1)

    env_ret2 = EnvRet(env_id="env-list-2", observation=np.array([2.0]))
    env_ret2.numpy = make_mock_numpy(env_ret2)

    result = [env_ret1, env_ret2]

    converted = policy._post_postprocess_hook(result)

    assert isinstance(converted, list)
    assert len(converted) == 2
    assert numpy_calls["count"] == 2
    assert policy._idle_as_infer.is_set() is True


def test_check_idle_allows_train_role():
    """check_idle should pass for TRAIN role without waiting."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.TRAIN)
    policy._idle_as_infer = threading.Event()
    # Don't set the event - would block if EVAL role waited

    # Should not raise and not block
    policy.check_idle()


def test_post_rollout_hook_skips_cpu_when_not_remote(monkeypatch):
    """_post_rollout_hook should not call numpy() when not in remote mode."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.EVAL)
    policy._idle_as_infer = threading.Event()

    # Ensure remote flags are NOT set
    monkeypatch.setenv("RLIGHTNING_REMOTE_STORAGE", "0")
    monkeypatch.setenv("RLIGHTNING_REMOTE_ENV", "0")

    numpy_called = {"called": False}

    policy_resp = PolicyResponse(env_id="env-local", action=torch.tensor(1))

    def _numpy():
        numpy_called["called"] = True
        return policy_resp

    policy_resp.numpy = _numpy

    result = policy._post_rollout_hook(policy_resp)

    assert numpy_called["called"] is False
    assert result is policy_resp
    assert policy._idle_as_infer.is_set() is True


def test_save_weights_unwraps_ddp_module(tmp_path, monkeypatch):
    """save_weights should unwrap DDP modules and save inner module state."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.TRAIN)
    policy.construct_network()

    # Create a mock DDP wrapper that passes isinstance check
    class MockDDP(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    original_linear = policy.linear
    wrapped_linear = MockDDP(original_linear)

    # Patch isinstance to recognize MockDDP as DDP
    original_isinstance = builtins.isinstance

    def patched_isinstance(obj, classinfo):
        if classinfo is DDP and type(obj).__name__ == "MockDDP":
            return True
        return original_isinstance(obj, classinfo)

    monkeypatch.setattr(builtins, "isinstance", patched_isinstance)

    policy.model_list = [("linear", wrapped_linear)]

    ckpt_path = tmp_path / "epoch_2" / "model.pt"
    policy.save_checkpoint(ckpt_path)

    assert ckpt_path.exists()
    state = torch.load(ckpt_path)
    assert "linear" in state
    # Verify the state dict came from the inner module
    assert "weight" in state["linear"]
    assert "bias" in state["linear"]


def test_save_checkpoint_without_model_attribute(tmp_path):
    """save_checkpoint should work for policies that only populate model_list."""
    policy = _DummyPolicy(_make_policy_config("sync"), PolicyRole.TRAIN)
    policy.construct_network()
    policy._find_model()
    if hasattr(policy, "model"):
        delattr(policy, "model")

    ckpt_path = tmp_path / "epoch_3" / "model.pt"
    policy.save_checkpoint(ckpt_path)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "linear" in state
    assert "weight" in state["linear"]
