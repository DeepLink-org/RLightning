"""End-to-end smoke tests that run tiny training experiments."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def _build_pythonpath(root: Path) -> str:
    parts = [
        str(root),
        str(root / "examples"),
        str(root / "third_party"),
        str(root / "third_party" / "rw_rl"),
        str(root / "third_party" / "rw_rl" / "src"),
    ]
    existing = os.environ.get("PYTHONPATH", "")
    return ":".join(parts + ([existing] if existing else []))


def _run_subprocess(cmd: list[str], root: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _build_pythonpath(root)
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("PYTHONHASHSEED", "0")
    return subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        check=True,
        timeout=timeout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.mark.parametrize(
    ("case_name", "overrides", "timeout_s"),
    [
        ("syncrl_local", [], 180),
        (
            "syncrl_local_replay",
            [
                "buffer.type=ReplayBuffer",
                "buffer.sampler.type=uniform",
                "train.batch_size=4",
                "train.max_rollout_steps=6",
                "env.max_episode_steps=6",
            ],
            180,
        ),
        (
            "syncrl_local_multi_worker",
            [
                "env.num_workers=2",
                "train.batch_size=8",
                "train.max_rollout_steps=4",
            ],
            180,
        ),
    ],
)
def test_minimal_training_experiment_smoke(tmp_path, case_name: str, overrides: list[str], timeout_s: int):
    root = Path(__file__).resolve().parents[2]
    script = root / "tests" / "tests_utils" / "run_mini_experiment.py"
    run_dir = tmp_path / case_name

    cmd = [
        sys.executable,
        str(script),
        "--config-name",
        "syncrl_local",
        "+train.seed=0",
        f"hydra.run.dir={run_dir}",
        f"log.log_dir={tmp_path / 'runs'}",
        f"log.name={case_name}",
        *overrides,
    ]

    try:
        result = _run_subprocess(cmd, root=root, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        pytest.fail(f"Minimal experiment {case_name} timed out after {timeout_s}s")
    except subprocess.CalledProcessError as exc:
        out_tail = (exc.stdout or "").splitlines()[-80:]
        err_tail = (exc.stderr or "").splitlines()[-80:]
        pytest.fail(
            f"Minimal experiment {case_name} failed.\n"
            f"Last stdout lines:\n{chr(10).join(out_tail)}\n\n"
            f"Last stderr lines:\n{chr(10).join(err_tail)}"
        )

    checkpoint = run_dir / "checkpoints" / "epoch_last.pt"
    assert checkpoint.exists(), f"Expected checkpoint at {checkpoint}"
    assert "Done." in result.stdout or "Done." in result.stderr
