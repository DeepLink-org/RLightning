# FIXME: it can pass individually but fails in `make test-isaaclab`

"""This file implemens test cases for tasks registered in isaac_marl"""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("isaaclab")

from rlightning.env.isaac_env import IsaacManagerBasedRLEnv
from rlightning.types import PolicyResponse
from rlightning.utils.config import EnvConfig

pytestmark = [pytest.mark.integration, pytest.mark.isaaclab, pytest.mark.gpu]


@pytest.fixture(scope="module", autouse=True)
def _auto_ray_cluster():
    # Ensure a Ray cluster is available for this module's tests
    from tests.test_utils import setup_ray_cluster, teardown_ray_cluster

    setup_ray_cluster(num_gpus=1)
    yield
    teardown_ray_cluster()


@pytest.fixture
def env_instance():
    """Fixture to manage the lifecycle of the environment."""
    env = _make_env()
    yield env

    # Teardown: This code runs after the test function finishes
    print("Fixture teardown: Closing env...")
    try:
        env.close()
    except Exception as e:
        print(f"Warning: Error closing env: {e}")
    print("Fixture teardown: Env closed.")


def _make_env():
    task = "Tracking-Flat-G1-v0"

    max_episode_steps = 10
    root = Path(__file__).resolve().parents[3]
    motion_dir = root / ".data" / "lafan1" / "retargeted" / "wbc_tracking"
    examples_dir = root / "examples"

    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    # The tracking example pulls in the humanoid retargeting stack, which is
    # optional and not installed by the plain `isaaclab` extra.
    pytest.importorskip("mink")
    if not motion_dir.exists():
        pytest.skip(f"WBC motion asset directory not found: {motion_dir}")
    if not any(motion_dir.glob("*.npz")):
        pytest.skip(f"No WBC motion files found under: {motion_dir}")

    config = EnvConfig(
        name=task,
        task=task,
        backend="isaac_manager_based",
        max_episode_steps=max_episode_steps,
        env_kwargs=dict(
            env_spec="wbc_tracking.envs",
            launcher=dict(headless=True),
            env_cfg=dict(
                module="wbc_tracking.envs.flat_env_cfg::G1FlatEnvCfg",
                override=dict(
                    commands=dict(
                        motion={
                            "motion_dir": str(motion_dir),
                        }
                    )
                ),
            ),
        ),
    )

    original_argv = sys.argv[:]
    # Clear the parameters to prevent Isaac Sim from crashing due to reading
    # the -m parameter of pytest.
    sys.argv = [sys.argv[0]]

    try:
        env = IsaacManagerBasedRLEnv(config)
    finally:
        sys.argv = original_argv
    return env


def test_env_functionalities(env_instance):
    env = env_instance
    env_ret = env.reset()

    for step_cnt in range(100):

        action = torch.tensor(env.get_action_space().sample())
        policy_resp = PolicyResponse(env_id=env.env_id, action=action)
        env_ret = env.step(policy_resp)

        observation, reward, terminated, truncated, info = (
            env_ret.observation,
            env_ret.last_reward,
            env_ret.last_terminated,
            env_ret.last_truncated,
            env_ret.info,
        )

        print("reward:", reward.shape, reward.mean())
        print(
            "step:",
            step_cnt,
            "terminated envs:",
            terminated.sum(),
            "truncated envs:",
            truncated.sum(),
        )
