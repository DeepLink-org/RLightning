"""OpenVLA PPO CLI smoke tests for minimal full-flow training."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests.e2e._cli_smoke_utils import (
    artifact_mtime,
    assert_subprocess_success,
    assert_updated_artifact,
    example_python,
    require_model_path,
    resolve_gpu_ids,
    run_example,
    short_ray_tmpdir,
)

pytestmark = [pytest.mark.e2e, pytest.mark.integration, pytest.mark.slow]

_OPENVLA_DETERMINISTIC_SMOKE_OVERRIDES = [
    # Keep the real end-to-end path, but reduce rollout/reset randomness so
    # the smoke cases are less flaky across runs.
    "policy.sampling_params.do_sample=False",
    "policy.sampling_params.temperature_train=1.0",
    "env.0.use_fixed_reset_state_ids=True",
]


def _run_openvla_smoke(
    *,
    config_name: str,
    log_name: str,
    required_gpus: int,
    ray_key: str,
    train_micro_batch_size: int,
    eval_rollout_override: str = "train.max_eval_rollout_steps=2",
    extra_overrides: list[str] | None = None,
    timeout_s: int = 3600,
) -> None:
    root = Path(__file__).resolve().parents[2]
    require_model_path(root / "examples" / "openvla_ppo" / "conf" / "policy" / "openvla_ppo.yaml")

    visible_gpus = ",".join(resolve_gpu_ids(required_count=required_gpus, max_used_mb=1024))
    example_py = example_python(root, "openvla_ppo")
    expected_ckpt = root / "runs" / "openvla_ppo" / log_name / "weights" / "epoch_last.pt"
    previous_ckpt_mtime = artifact_mtime(expected_ckpt)

    cmd = [
        str(example_py),
        "-m",
        "examples.openvla_ppo.train_ppo",
        "--config-name",
        config_name,
        "log=tensorboard",
        "+debug=False",
        "+cluster.ray_address=local",
        "+train.seed=0",
        "train.max_epochs=1",
        "train.max_rollout_steps=2",
        eval_rollout_override,
        "train.batch_size=16",
        "train.mini_batch_size=8",
        f"train.micro_batch_size={train_micro_batch_size}",
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
def test_openvla_ppo_ddp_smoke():
    _run_openvla_smoke(
        config_name="train_ppo",
        log_name="openvla_ddp_smoke",
        required_gpus=4,
        ray_key="od",
        train_micro_batch_size=1,
        extra_overrides=[
            "cluster.train_worker_num=2",
            "cluster.eval_worker_num=1",
            "env.0.num_envs=16",
            *_OPENVLA_DETERMINISTIC_SMOKE_OVERRIDES,
        ],
    )


@pytest.mark.gpu
def test_openvla_ppo_sync_smoke():
    _run_openvla_smoke(
        config_name="train_ppo",
        log_name="openvla_sync_smoke",
        required_gpus=3,
        ray_key="os",
        train_micro_batch_size=8,
        extra_overrides=[
            "env.0.num_envs=16",
            *_OPENVLA_DETERMINISTIC_SMOKE_OVERRIDES,
        ],
    )


@pytest.mark.gpu
def test_openvla_ppo_colocate_x8_smoke():
    _run_openvla_smoke(
        config_name="train_ppo_colocate_ddp_x8",
        log_name="openvla_colocate_x8_smoke",
        required_gpus=8,
        ray_key="ox",
        train_micro_batch_size=1,
        eval_rollout_override="+train.max_eval_rollout_steps=2",
        extra_overrides=[*_OPENVLA_DETERMINISTIC_SMOKE_OVERRIDES],
        timeout_s=5400,
    )
