# Testing

This repository uses separate test layers and, for heavy integration stacks, separate virtual environments.

Recommended local workflow:

```bash
uv venv .venv
uv pip install --python .venv/bin/python --torch-backend cpu -e '.[dev]'
.venv/bin/python -m pytest -q tests/unit
```

Test layers:

- `tests/unit`: fast baseline tests for core logic and data handling.
- `tests/integration`: multi-component tests. These run in dedicated integration virtual environments because `mujoco`, `ale`, and `isaaclab` extras cannot all live in one environment.
- `tests/e2e`: minimal real training smoke runs that validate the end-to-end launch path and expected artifacts.

Integration environments:

- `core`: `pytest + mujoco + ale`
- `isaaclab`: `pytest + isaaclab + humanoid + humanoid-dev`

Setup commands:

```bash
bash scripts/testing/setup_integration_env.sh core
bash scripts/testing/setup_integration_env.sh isaaclab
```

Run commands:

```bash
bash scripts/testing/run_integration_tests.sh core -q
bash scripts/testing/run_integration_tests.sh core -vv -s -ra
bash scripts/testing/run_integration_tests.sh isaaclab -q
bash scripts/testing/run_integration_tests.sh isaaclab -vv -s -ra
```

One-command full test sweep:

```bash
bash scripts/testing/run_all_tests.sh
bash scripts/testing/run_all_tests.sh --verbose
bash scripts/testing/run_all_tests.sh unit
bash scripts/testing/run_all_tests.sh integration
bash scripts/testing/run_all_tests.sh integration-core integration-isaaclab
bash scripts/testing/run_all_tests.sh e2e
```

The `core` profile intentionally excludes the `isaaclab` marker. The `isaaclab` profile runs only the IsaacLab integration slice. Example-specific `.venv` directories are used by E2E tests, but they are not visible to integration tests that run inside the root pytest process.
