"""WBC tracking CLI smoke tests for minimal full-flow training."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e._cli_smoke_utils import (
    assert_subprocess_success,
    assert_updated_artifact,
    example_python,
    find_new_hydra_run_dir,
    resolve_gpu_ids,
    run_example,
    short_ray_tmpdir,
    snapshot_hydra_runs,
)

pytestmark = [pytest.mark.e2e, pytest.mark.integration, pytest.mark.slow]


def _wbc_minimal_stability_overrides() -> list[str]:
    return [
        "+env.env_kwargs.env_cfg.override.scene.terrain.visual_material=null",
        "+env.env_kwargs.env_cfg.override.scene.terrain.physics_material=null",
        "+env.env_kwargs.env_cfg.override.scene.contact_forces.debug_vis=false",
        "+env.env_kwargs.env_cfg.override.commands.motion.debug_vis=false",
    ]


def _wbc_multi_task_stability_overrides(env_indices: list[int]) -> list[str]:
    overrides: list[str] = []
    for env_idx in env_indices:
        prefix = f"+env.{env_idx}.env_kwargs.env_cfg.override"
        overrides.extend(
            [
                f"{prefix}.scene.terrain.visual_material=null",
                f"{prefix}.scene.terrain.physics_material=null",
                f"{prefix}.scene.contact_forces.debug_vis=false",
                f"{prefix}.commands.motion.debug_vis=false",
            ]
        )
    return overrides


@pytest.mark.gpu
@pytest.mark.isaaclab
def test_wbc_tracking_launch_smoke():
    motion_dir = Path(".data/lafan1/retargeted/wbc_tracking")
    if not motion_dir.exists():
        pytest.skip(f"WBC motion asset directory not found: {motion_dir}")

    root = Path(__file__).resolve().parents[2]
    example_py = example_python(root, "wbc_tracking")
    visible_gpus = ",".join(resolve_gpu_ids(required_count=3, max_used_mb=1024))
    log_name = "wbc_tracking_launch_smoke"
    hydra_runs_before = snapshot_hydra_runs(root)
    cmd = [
        str(example_py),
        str(root / "examples" / "wbc_tracking" / "train.py"),
        "--config-name",
        "launch",
        "debug=False",
        "cluster.ray_address=local",
        "+train.seed=0",
        "env.num_envs=16",
        "train.max_rollout_steps=2",
        "train.max_epochs=1",
        "train.batch_size=16",
        f"log.name={log_name}",
        *_wbc_minimal_stability_overrides(),
    ]

    try:
        result = run_example(
            root,
            cmd,
            timeout_s=2400,
            extra_env={
                "CUDA_VISIBLE_DEVICES": visible_gpus,
                "RAY_TMPDIR": short_ray_tmpdir("w"),
            },
        )
    except subprocess.TimeoutExpired:
        pytest.fail("wbc_tracking launch smoke test timed out")

    assert_subprocess_success(
        result,
        "wbc_tracking launch smoke test",
        allow_known_wbc_cleanup_error=True,
    )
    run_dir = find_new_hydra_run_dir(root, hydra_runs_before, log_name)
    checkpoint = run_dir / "checkpoints" / "epoch_last.pt"
    assert_updated_artifact(
        checkpoint,
        previous_mtime=None,
        name="wbc_tracking launch smoke test",
    )


@pytest.mark.gpu
@pytest.mark.isaaclab
def test_wbc_tracking_ddp_smoke():
    motion_dir = Path(".data/lafan1/retargeted/wbc_tracking")
    if not motion_dir.exists():
        pytest.skip(f"WBC motion asset directory not found: {motion_dir}")

    root = Path(__file__).resolve().parents[2]
    example_py = example_python(root, "wbc_tracking")
    visible_gpus = ",".join(resolve_gpu_ids(required_count=6, max_used_mb=1024))
    log_name = "wbc_tracking_ddp_smoke"
    hydra_runs_before = snapshot_hydra_runs(root)
    cmd = [
        str(example_py),
        str(root / "examples" / "wbc_tracking" / "train.py"),
        "--config-name",
        "launch_ddp",
        "debug=False",
        "cluster.ray_address=local",
        "+train.seed=0",
        "env.0.num_envs=16",
        "env.1.num_envs=16",
        "train.max_rollout_steps=2",
        "train.max_epochs=1",
        "train.batch_size=16",
        f"log.name={log_name}",
        *_wbc_multi_task_stability_overrides([0, 1]),
    ]

    try:
        result = run_example(
            root,
            cmd,
            timeout_s=3000,
            extra_env={
                "CUDA_VISIBLE_DEVICES": visible_gpus,
                "RAY_TMPDIR": short_ray_tmpdir("wd"),
            },
        )
    except subprocess.TimeoutExpired:
        pytest.fail("wbc_tracking ddp smoke test timed out")

    assert_subprocess_success(
        result,
        "wbc_tracking ddp smoke test",
        allow_known_wbc_cleanup_error=True,
    )
    run_dir = find_new_hydra_run_dir(root, hydra_runs_before, log_name)
    checkpoint = run_dir / "checkpoints" / "epoch_last.pt"
    assert_updated_artifact(
        checkpoint,
        previous_mtime=None,
        name="wbc_tracking ddp smoke test",
    )


@pytest.mark.gpu
@pytest.mark.isaaclab
def test_wbc_tracking_local_smoke():
    motion_dir = Path(".data/lafan1/retargeted/wbc_tracking")
    if not motion_dir.exists():
        pytest.skip(f"WBC motion asset directory not found: {motion_dir}")

    root = Path(__file__).resolve().parents[2]
    log_name = "wbc_tracking_local_smoke"
    hydra_runs_before = snapshot_hydra_runs(root)
    cmd = [
        "bash",
        str(root / "examples" / "wbc_tracking" / "launch_local.sh"),
        "debug=False",
        "env=single_task",
        "+train.seed=0",
        "env.num_envs=16",
        "train.max_epochs=1",
        "train.max_rollout_steps=2",
        "train.batch_size=16",
        f"log.name={log_name}",
        *_wbc_minimal_stability_overrides(),
    ]

    try:
        result = run_example(
            root,
            cmd,
            timeout_s=1800,
            extra_env={
                "CUDA_VISIBLE_DEVICES": resolve_gpu_ids(required_count=1, max_used_mb=1024)[0],
                "RAY_TMPDIR": short_ray_tmpdir("l"),
            },
        )
    except subprocess.TimeoutExpired:
        pytest.fail("wbc_tracking local smoke test timed out")

    assert_subprocess_success(
        result,
        "wbc_tracking local smoke test",
    )
    run_dir = find_new_hydra_run_dir(root, hydra_runs_before, log_name)
    checkpoint = run_dir / "checkpoints" / "epoch_last.pt"
    assert_updated_artifact(
        checkpoint,
        previous_mtime=None,
        name="wbc_tracking local smoke test",
    )
