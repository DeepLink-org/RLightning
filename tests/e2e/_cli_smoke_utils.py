"""Shared helpers for CLI-driven E2E smoke tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml


def build_pythonpath(root: Path) -> str:
    parts = [
        str(root),
        str(root / "examples"),
        str(root / "third_party"),
        str(root / "third_party" / "rw_rl"),
        str(root / "third_party" / "rw_rl" / "src"),
    ]
    existing = os.environ.get("PYTHONPATH", "")
    return ":".join(parts + ([existing] if existing else []))


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_model_path(policy_yaml: Path) -> None:
    cfg = load_yaml(policy_yaml)
    model_path = cfg["model_cfg"]["model_path"]
    if not Path(model_path).exists():
        pytest.skip(f"Model path not found: {model_path}")


def example_python(root: Path, example_name: str) -> Path:
    python = root / "examples" / example_name / ".venv" / "bin" / "python"
    if not python.exists():
        pytest.skip(f"Example virtualenv not found: {python}")
    return python


def pick_free_gpu_ids(required_count: int, max_used_mb: int = 1024) -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("nvidia-smi is unavailable")

    gpu_rows: list[tuple[int, int]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        idx_str, used_str = [part.strip() for part in line.split(",", maxsplit=1)]
        gpu_rows.append((int(idx_str), int(used_str)))

    free_gpu_ids = [str(idx) for idx, used_mb in sorted(gpu_rows, key=lambda row: row[1]) if used_mb <= max_used_mb]
    if len(free_gpu_ids) < required_count:
        pytest.skip(f"Need {required_count} mostly idle GPUs (<= {max_used_mb} MiB used), found {free_gpu_ids}")
    return free_gpu_ids[:required_count]


def resolve_gpu_ids(required_count: int, max_used_mb: int = 1024) -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if configured:
        gpu_ids = [gpu_id.strip() for gpu_id in configured.split(",") if gpu_id.strip()]
        if len(gpu_ids) < required_count:
            pytest.skip(
                f"CUDA_VISIBLE_DEVICES={configured!r} exposes {len(gpu_ids)} GPU(s), "
                f"but this test needs {required_count}"
            )
        return gpu_ids[:required_count]
    return pick_free_gpu_ids(required_count=required_count, max_used_mb=max_used_mb)


def short_ray_tmpdir(key: str) -> str:
    path = Path("/tmp") / f"r{key}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def run_example(
    root: Path,
    cmd: list[str],
    timeout_s: int,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = build_pythonpath(root)
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONHASHSEED", "0")
    if extra_env is not None:
        env.update(extra_env)

    live_logs = env.get("RLIGHTNING_E2E_LIVE_LOGS") == "1"
    if live_logs:
        return subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            check=False,
            timeout=timeout_s,
        )

    return subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        check=False,
        timeout=timeout_s,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def artifact_mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


def assert_updated_artifact(path: Path, previous_mtime: float | None, name: str) -> None:
    if not path.exists():
        pytest.fail(f"{name} did not produce expected artifact: {path}")
    if previous_mtime is not None and path.stat().st_mtime <= previous_mtime:
        pytest.fail(f"{name} did not update expected artifact: {path}")


def snapshot_hydra_runs(root: Path) -> set[Path]:
    outputs_root = root / "outputs"
    if not outputs_root.exists():
        return set()
    return {path.resolve() for path in outputs_root.glob("*/*") if path.is_dir()}


def find_new_hydra_run_dir(root: Path, before: set[Path], expected_log_name: str) -> Path:
    candidates = []
    for path in snapshot_hydra_runs(root) - before:
        config_path = path / ".hydra" / "config.yaml"
        if not config_path.exists():
            continue
        try:
            cfg = load_yaml(config_path)
        except Exception:
            continue
        if cfg.get("log", {}).get("name") == expected_log_name:
            candidates.append(path)

    if not candidates:
        pytest.fail(f"Could not find new Hydra run dir for log.name={expected_log_name}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def known_wbc_cleanup_error(output: str) -> bool:
    return (
        "ray.exceptions.ActorDiedError" in output
        and "env_group.close()" in output
        and "IsaacManagerBasedRLEnv" in output
    )


def assert_subprocess_success(
    result: subprocess.CompletedProcess[str],
    name: str,
    *,
    allow_known_wbc_cleanup_error: bool = False,
    expected_paths: list[Path] | None = None,
) -> None:
    captured_output = result.stdout is not None or result.stderr is not None
    output = (result.stdout or "") + "\n" + (result.stderr or "") if captured_output else ""
    if result.returncode == 0:
        missing_paths = [path for path in expected_paths or [] if not path.exists()]
        if missing_paths:
            pytest.fail(
                f"{name} exited with code 0 but did not produce expected artifacts: "
                + ", ".join(str(path) for path in missing_paths)
            )
        if captured_output and "Done." not in output:
            pytest.fail(f"{name} finished without the expected completion marker.\n{output[-6000:]}")
        return

    if allow_known_wbc_cleanup_error and known_wbc_cleanup_error(output):
        missing_paths = [path for path in expected_paths or [] if not path.exists()]
        if missing_paths:
            pytest.fail(
                f"{name} hit the known cleanup error but did not produce expected artifacts: "
                + ", ".join(str(path) for path in missing_paths)
            )
        return

    out_tail = (result.stdout or "").splitlines()[-80:]
    err_tail = (result.stderr or "").splitlines()[-80:]
    pytest.fail(
        f"{name} failed with return code {result.returncode}.\n"
        f"Last stdout lines:\n{chr(10).join(out_tail)}\n\n"
        f"Last stderr lines:\n{chr(10).join(err_tail)}"
    )
