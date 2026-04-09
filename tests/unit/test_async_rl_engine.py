"""
Unit tests for AsyncRLEngine.

Tests the core functionality of the async RL engine including initialization,
warm-up, and basic engine operations without external dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest

from rlightning.engine.async_rl_engine import AsyncRLEngine



class _SimpleMocker:
    def __init__(self) -> None:
        self._patchers = []

    def Mock(self, *args, **kwargs):
        return MagicMock(*args, **kwargs)

    def patch(self, target, *args, **kwargs):
        patcher = patch(target, *args, **kwargs)
        mocked = patcher.start()
        self._patchers.append(patcher)
        return mocked

    def stopall(self) -> None:
        while self._patchers:
            self._patchers.pop().stop()


@pytest.fixture
def mocker():
    helper = _SimpleMocker()
    try:
        yield helper
    finally:
        helper.stopall()


@pytest.fixture(autouse=True)
def debug_mode_env(monkeypatch):
    monkeypatch.setenv("RLIGHTNING_DEBUG", "1")


@pytest.fixture
def mock_config(mocker):
    config = mocker.Mock()
    config.train.max_epochs = 2
    config.train.max_rollout_steps = 10
    config.train.batch_size = 4
    return config


@pytest.fixture
def mock_env_group(mocker):
    env_group = mocker.Mock()
    env_group.observation_space.return_value = mocker.Mock()
    env_group.action_space.return_value = mocker.Mock()
    # warm_up uses reset() and step() returning (requests, truncations)
    env_group.reset.return_value = (mocker.Mock(), [])
    env_group.step.return_value = (mocker.Mock(), [])
    env_group.init.return_value = (mocker.Mock(), {})
    return env_group


@pytest.fixture
def mock_policy_group(mocker):
    policy_group = mocker.Mock()
    policy_group.init_eval.return_value = mocker.Mock()
    policy_group.train_list = [mocker.Mock()]
    return policy_group


@pytest.fixture
def mock_buffer(mocker):
    buffer = mocker.Mock()
    buffer.size.return_value = 10  # Always has enough data
    return buffer


def test_engine_initialization(mock_config, mock_env_group, mock_policy_group, mock_buffer):
    engine = AsyncRLEngine(
        config=mock_config,
        env_group=mock_env_group,
        policy_group=mock_policy_group,
        buffer=mock_buffer,
    )

    assert engine.config == mock_config
    assert engine.env_group == mock_env_group
    assert engine.policy_group == mock_policy_group
    assert engine.buffer == mock_buffer


def test_warm_up_calls_initialization_methods(mock_config, mock_env_group, mock_policy_group, mock_buffer):
    AsyncRLEngine(
        config=mock_config,
        env_group=mock_env_group,
        policy_group=mock_policy_group,
        buffer=mock_buffer,
    )

    mock_env_group.init.assert_called_once()
    mock_policy_group.init_eval.assert_called_once()
    mock_policy_group.init_train.assert_called_once()
    mock_buffer.init.assert_called_once()


def test_warm_up_performs_dummy_rollout(mock_config, mock_env_group, mock_policy_group, mock_buffer, mocker):
    mock_requests = mocker.Mock()
    mock_responses = mocker.Mock()
    mock_responses.ids.return_value = ["episode_1"]

    mock_env_group.reset.return_value = (mock_requests, [])
    mock_env_group.step.return_value = (mock_requests, [])
    mock_policy_group.rollout_batch.return_value = mock_responses

    AsyncRLEngine(
        config=mock_config,
        env_group=mock_env_group,
        policy_group=mock_policy_group,
        buffer=mock_buffer,
    )

    assert mock_env_group.reset.call_count >= 1
    assert mock_policy_group.rollout_batch.call_count >= 10
    assert mock_buffer.add_batched_data_async.call_count >= 20


def test_warm_up_performs_dummy_training(mock_config, mock_env_group, mock_policy_group, mock_buffer):
    AsyncRLEngine(
        config=mock_config,
        env_group=mock_env_group,
        policy_group=mock_policy_group,
        buffer=mock_buffer,
    )

    mock_policy_group.update_dataset.assert_called_once()
    mock_policy_group.train.assert_called_once()
    mock_policy_group.sync_weights.assert_called_once()


def test_engine_initialization_without_debug_warmup(
    mock_config, mock_env_group, mock_policy_group, mock_buffer, mocker, monkeypatch
):
    monkeypatch.setenv("RLIGHTNING_DEBUG", "0")
    mock_policy_group.send_weights = mocker.Mock()
    mock_policy_group.notify_update_weights = mocker.Mock()

    warm_up_mock = mocker.patch("rlightning.engine.async_rl_engine.AsyncRLEngine.warm_up")

    AsyncRLEngine(
        config=mock_config,
        env_group=mock_env_group,
        policy_group=mock_policy_group,
        buffer=mock_buffer,
    )

    warm_up_mock.assert_not_called()
    mock_env_group.init.assert_called_once()
    mock_policy_group.init_eval.assert_called_once()
    mock_policy_group.init_train.assert_called_once()
    mock_buffer.init.assert_called_once()
    mock_policy_group.sync_weights.assert_called_once()


def test_rollout_loop_structure(mock_config, mock_env_group, mock_policy_group, mock_buffer):
    mock_requests = MagicMock()
    mock_responses = MagicMock()

    mock_env_group.reset.return_value = (mock_requests, [])
    mock_env_group.step_async.return_value = None
    mock_env_group.collect_async.return_value = (mock_requests, [])
    mock_policy_group.rollout_batch.return_value = mock_responses

    engine = AsyncRLEngine(
        config=mock_config,
        env_group=mock_env_group,
        policy_group=mock_policy_group,
        buffer=mock_buffer,
    )

    assert hasattr(engine, "rollout")
    assert callable(engine.rollout)


def test_train_loop_structure(mock_config, mock_env_group, mock_policy_group, mock_buffer):
    engine = AsyncRLEngine(
        config=mock_config,
        env_group=mock_env_group,
        policy_group=mock_policy_group,
        buffer=mock_buffer,
    )

    assert hasattr(engine, "train")
    assert callable(engine.train)


def test_update_weights_loop_structure(mock_config, mock_env_group, mock_policy_group, mock_buffer):
    engine = AsyncRLEngine(
        config=mock_config,
        env_group=mock_env_group,
        policy_group=mock_policy_group,
        buffer=mock_buffer,
    )

    assert hasattr(engine, "sync_weights")
    assert callable(engine.sync_weights)


def test_run_method_starts_threads(mock_config, mock_env_group, mock_policy_group, mock_buffer, mocker):
    mock_thread = mocker.patch("threading.Thread")
    mock_thread_instance = mocker.Mock()
    mock_thread.return_value = mock_thread_instance

    engine = AsyncRLEngine(
        config=mock_config,
        env_group=mock_env_group,
        policy_group=mock_policy_group,
        buffer=mock_buffer,
    )

    engine.coordinator._done_event.set()

    engine.run()
    assert mock_thread.call_count == 4
    assert mock_thread_instance.start.call_count == 4
    assert mock_thread_instance.join.call_count == 4


@pytest.mark.parametrize(
    "max_epochs,max_rollout_steps,batch_size",
    [
        (1, 5, 2),
        (3, 15, 8),
        (10, 100, 32),
    ],
)
def test_engine_with_different_configs(
    max_epochs,
    max_rollout_steps,
    batch_size,
    mock_env_group,
    mock_policy_group,
    mock_buffer,
):
    config = MagicMock()
    config.train.max_epochs = max_epochs
    config.train.max_rollout_steps = max_rollout_steps
    config.train.batch_size = batch_size

    engine = AsyncRLEngine(
        config=config,
        env_group=mock_env_group,
        policy_group=mock_policy_group,
        buffer=mock_buffer,
    )

    assert engine.config.train.max_epochs == max_epochs
    assert engine.config.train.max_rollout_steps == max_rollout_steps
    assert engine.config.train.batch_size == batch_size
    assert engine.config.train.batch_size == batch_size
