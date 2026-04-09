"""Minimal end-to-end training tests for OpenPI PPO examples."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e._cli_smoke_utils import (
    artifact_mtime,
    assert_subprocess_success,
    assert_updated_artifact,
    example_python,
    resolve_gpu_ids,
    run_example,
    short_ray_tmpdir,
)

pytestmark = [pytest.mark.e2e, pytest.mark.integration, pytest.mark.slow]


def _assert_openpi_assets(root: Path) -> None:
    model_path = Path("/data/ckpts/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT")
    if not model_path.exists():
        pytest.skip(f"OpenPI model path not found: {model_path}")

    libero_path = root / "examples" / "openpi_ppo" / ".venv" / "LIBERO"
    if not libero_path.exists():
        pytest.skip(f"LIBERO assets not found: {libero_path}")


def _run_openpi_smoke(
    *,
    config_name: str,
    log_name: str,
    required_gpus: int,
    ray_key: str,
    extra_overrides: list[str] | None = None,
    timeout_s: int = 2400,
) -> None:
    root = Path(__file__).resolve().parents[2]
    _assert_openpi_assets(root)

    example_py = example_python(root, "openpi_ppo")
    visible_gpus = ",".join(resolve_gpu_ids(required_count=required_gpus, max_used_mb=1024))
    expected_ckpt = root / "runs" / "openpi_ppo" / log_name / "weights" / "epoch_last.pt"
    previous_ckpt_mtime = artifact_mtime(expected_ckpt)

    cmd = [
        str(example_py),
        "-m",
        "examples.openpi_ppo.train_ppo",
        "--config-name",
        config_name,
        "log=tensorboard",
        "+debug=False",
        "+cluster.ray_address=local",
        "+train.seed=0",
        "train.max_epochs=1",
        "train.max_rollout_steps=2",
        "train.warm_up_rollout_steps=2",
        "train.batch_size=16",
        "train.mini_batch_size=8",
        "train.micro_batch_size=8",
        "train.update_epoch=1",
        "train.rollout_epoch=1",
        f"log.name={log_name}",
        *(extra_overrides or []),
    ]

    try:
        result = run_example(
            root,
            cmd,
            timeout_s=timeout_s,
            extra_env={
                "CUDA_VISIBLE_DEVICES": visible_gpus,
                "RAY_TMPDIR": short_ray_tmpdir(ray_key),
            },
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{log_name} timed out")

    assert_subprocess_success(
        result,
        log_name,
        expected_paths=[expected_ckpt],
    )
    assert_updated_artifact(expected_ckpt, previous_ckpt_mtime, log_name)


@pytest.mark.gpu
def test_openpi_ppo_tiny_sync_smoke():
    _run_openpi_smoke(
        config_name="train_ppo_tiny",
        log_name="openpi_tiny_sync_smoke",
        required_gpus=3,
        ray_key="pi",
        extra_overrides=[
            "env.0.num_envs=8",
        ],
    )


@pytest.mark.gpu
def test_openpi_ppo_sync_smoke():
    _run_openpi_smoke(
        config_name="train_ppo",
        log_name="openpi_sync_smoke",
        required_gpus=3,
        ray_key="ps",
        extra_overrides=[
            "env.0.num_envs=8",
        ],
    )


@pytest.mark.gpu
def test_openpi_ppo_tiny_ddp_smoke():
    _run_openpi_smoke(
        config_name="train_ppo_tiny_ddp",
        log_name="openpi_tiny_ddp_smoke",
        required_gpus=8,
        ray_key="pd",
        extra_overrides=[
            "cluster=4t4e",
            "env=libero_x4",
            "env.0.num_envs=8",
            "train.mini_batch_size=16",
            "train.micro_batch_size=4",
        ],
        timeout_s=3600,
    )
