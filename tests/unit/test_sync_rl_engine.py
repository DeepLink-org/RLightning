import builtins
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from rlightning.engine.sync_rl_engine import SyncRLEngine
from rlightning.env import EnvMeta
from rlightning.policy.base_policy import BasePolicy, PolicyRole
from rlightning.policy.policy_group import PolicyGroup
from rlightning.types import BatchedData, EnvRet
from rlightning.utils.config import ClusterConfig, PolicyConfig, TrainConfig
from rlightning.utils.registry import POLICIES

# Resolve repo root from tests/unit/test_sync_rl_engine.py.
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def policy_intialization_entrypoint(cfg: DictConfig, as_remote: bool, role: str) -> BasePolicy:
    """Help function to initialize BasePolicy from config.

    This function can be used to test whether the configuration is valid.

    Args:
        cfg (DictConfig): Configuration of BasePolicy.
        as_remote (bool): Enable remote mode or not.
        role (str): Indicate current policy is for 'train' or 'test'.

    Raises:
        NotImplementedError: Remote mode is not support yet.

    Returns:
        BasePolicy: An instance of BasePolicy.
    """

    print(f"--- Complete Config ---\n{cfg}")

    policy_cfg = PolicyConfig.from_omegaconf(cfg)

    policy_cfg.as_remote = as_remote
    policy_cls = POLICIES.get(policy_cfg.type)

    if as_remote:

        class DistributedPolicyCls(policy_cls, DistributedMixin):
            pass

        policy_cls = DistributedPolicyCls

    if as_remote:
        policy_name = f"policy-{role}-{policy_cfg.type}-test"

        raise NotImplementedError
    else:
        policy = policy_cls(policy_cfg, role_type=PolicyRole(role))

    return policy


def make_engine_with_mocks(**cfg_overrides) -> Tuple[SyncRLEngine, Dict]:
    """Create a SyncRLEngine instance with mocked components for testing."""

    # Create instance without calling __init__
    engine = SyncRLEngine.__new__(SyncRLEngine)

    # Minimal config with nested train and policy attributes
    train_cfg = TrainConfig(
        max_epochs=2,
        max_rollout_steps=2,
        batch_size=10,
        lr=0.0003,
        mode="sync",
        parallel=None,
        eval_interval=2,
        save_interval=5,
        save_dir="./save_dir",
    )
    config = SimpleNamespace(train=train_cfg)

    if cfg_overrides.get("train"):
        for k, v in cfg_overrides["train"].items():
            setattr(config.train, k, v)

    if cfg_overrides.get("policy"):
        config.policy = cfg_overrides["policy"]
    else:
        config.policy = mock.Mock()

    config.cluster = ClusterConfig()

    # Create mocks for components
    env_group = mock.Mock()

    # auto_reset context manager mock
    auto_reset_cm = mock.MagicMock()
    auto_reset_cm.__enter__.return_value = env_group
    auto_reset_cm.__exit__.return_value = False
    env_group.auto_reset.return_value = auto_reset_cm

    # init returns metadata
    env_meta = EnvMeta(
        env_id=None,
        action_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,)),
        num_envs=2,
    )
    env_group.init.return_value = [env_meta]

    batched_action_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(env_meta.num_envs,) + env_meta.action_space.shape,
        dtype=np.float32,
    )
    batched_observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(env_meta.num_envs,) + env_meta.observation_space.shape,
        dtype=np.float32,
    )

    # reset/step yield a batched_env_ret and truncations
    batched_env_ret = BatchedData(
        ids=[0],
        data=[
            EnvRet(
                env_id=0,
                observation=torch.tensor(batched_observation_space.sample()),
                last_reward=torch.zeros(env_meta.num_envs),
                last_terminated=torch.zeros(env_meta.num_envs, dtype=torch.bool),
                last_truncated=torch.zeros(env_meta.num_envs, dtype=torch.bool),
                info={},
            )
        ],
    )
    truncations = None
    env_group.reset.return_value = (batched_env_ret, truncations)
    env_group.step.return_value = (batched_env_ret, truncations)
    # get_env_stats returns a dict for logging
    env_group.get_env_stats.return_value = {"reward_mean": 1.2345}

    buffer = mock.MagicMock()
    buffer.init.return_value = None
    buffer.add_batched_transition.return_value = None
    buffer.truncate_episodes.return_value = None
    # buffer.sample returns a dummy dataset
    sample_data = {"data": "sample"}
    buffer.sample.return_value = sample_data
    buffer.clear.return_value = None
    # Provide sampler attribute used in train
    buffer.sampler = SimpleNamespace(replacement=True)

    # Provide a default policy_group mock so rollout/train won’t fail
    batched_policy_resp = mock.Mock()
    policy_group = mock.Mock()
    policy_group.rollout_batch.return_value = batched_policy_resp
    policy_group.send_weights.return_value = None
    policy_group.notify_update_weights.return_value = None
    policy_group.update_dataset.return_value = None
    policy_group.train.return_value = None

    # timing_raw used by profiler.timer but we can leave as empty dict
    engine.timing_raw = {}

    # attach mocks and config to engine
    engine.env_group = env_group
    engine.policy_group = policy_group
    engine.buffer = buffer
    engine.config = config
    engine.epoch = 0

    return engine, {
        "env_meta": env_meta,
        "batched_env_ret": batched_env_ret,
        # "batched_policy_resp": batched_policy_resp,
        "sample_data": sample_data,
    }


def test_warm_up_calls_sequence():
    engine, ctx = make_engine_with_mocks()

    # Mock policy_group with expected methods
    policy_group = mock.Mock()
    episode_meta = {"episode": "meta"}
    policy_group.init_eval.return_value = ctx["env_meta"]
    policy_group.init_train.return_value = None

    # batched policy response mock with ids method
    batched_policy_resp = mock.Mock()
    batched_policy_resp.ids.return_value = [1, 2]
    policy_group.rollout_batch.return_value = batched_policy_resp

    policy_group.update_dataset.return_value = None
    policy_group.train.return_value = None
    policy_group.send_weights.return_value = None
    policy_group.notify_update_weights.return_value = None
    policy_group.train_list = [mock.Mock()]

    engine.policy_group = policy_group

    # Call warm_up and assert all expected interactions occur
    engine.env_meta_list = [ctx["env_meta"]]
    engine.warm_up()

    # truncate_episodes called with ids() of last batched_policy_resp
    engine.buffer.truncate_episodes.assert_called_once_with(batched_policy_resp.ids())

    # dummy run train: sample and update_dataset and train called
    engine.buffer.sample.assert_called_once_with(batch_size=engine.config.train.batch_size)  # defined in warm_up
    engine.policy_group.update_dataset.assert_called_once_with(ctx["sample_data"])
    engine.policy_group.train.assert_called_once()

    # buffer cleared
    engine.buffer.clear.assert_called_once()


def test_rollout_calls_add_batched_transition_and_get_env_stats(monkeypatch, capsys):
    # Use small max_rollout_steps to verify loop behavior
    engine, ctx = make_engine_with_mocks()
    engine.config.train.max_rollout_steps = 3

    # Ensure auto_reset context manager returns env_group (mock already configured)
    engine.rollout(obj_set="train", prefix="rollout/")

    # Expected number of calls: max_rollout_steps + 1 iterations
    expected_calls = engine.config.train.max_rollout_steps + 1
    assert engine.buffer.add_batched_transition.call_count == expected_calls

    # get_env_stats should be called once with reset=True and prefix
    engine.env_group.get_env_stats.assert_called_once_with(reset=True)


def test_update_weights_calls_policy_group():
    engine, ctx = make_engine_with_mocks()
    engine.sync_weights()

    engine.policy_group.sync_weights.assert_called_once_with()


def test_run_invokes_flow_and_update_weights_calls(monkeypatch):
    engine, ctx = make_engine_with_mocks()
    # Replace rollout/train/update_weights with mocks to track calls
    engine.rollout = mock.Mock()
    engine.update_dataset = mock.Mock()
    engine.train = mock.Mock()
    engine.sync_weights = mock.Mock()

    # Ensure epochs small and intervals set so update branches run
    engine.config.train.max_epochs = 3
    engine.config.train.save_interval = 1
    engine.config.train.eval_interval = 1

    # Disable verbose/progress to avoid side-effects
    os.environ["RLIGHTNING_VERBOSE"] = "0"

    engine.run()

    # rollout should be called at least once per epoch for training and evals as configured
    assert engine.rollout.call_count >= engine.config.train.max_epochs
    # train should be called once per epoch
    assert engine.train.call_count == engine.config.train.max_epochs
    # sync_weights should be called once per epoch
    assert engine.sync_weights.call_count == engine.config.train.max_epochs
    # restore flags if needed (no-op in this test scope)
