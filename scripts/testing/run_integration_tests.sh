#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

usage() {
    cat <<'EOF'
Usage: bash scripts/testing/run_integration_tests.sh <profile> [pytest args...]

Profiles:
  core      Run non-IsaacLab integration tests in .venv-int-core.
  isaaclab  Run IsaacLab integration tests in .venv-int-isaaclab.

Examples:
  bash scripts/testing/run_integration_tests.sh core -q
  bash scripts/testing/run_integration_tests.sh core -vv -s -ra
  bash scripts/testing/run_integration_tests.sh isaaclab -q
  bash scripts/testing/run_integration_tests.sh isaaclab -vv -s -ra
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

profile="$1"
shift

case "$profile" in
    core)
        venv_dir="$ROOT_DIR/.venv-int-core"
        default_args=(-o addopts='' -m 'not isaaclab' tests/integration)
        ;;
    isaaclab)
        venv_dir="$ROOT_DIR/.venv-int-isaaclab"
        default_args=(-o addopts='' -m isaaclab tests/integration/env/test_isaac_manager_based.py)
        ;;
    *)
        echo "Unknown profile: $profile" >&2
        usage
        exit 1
        ;;
esac

if [[ ! -x "$venv_dir/bin/python" ]]; then
    echo "Missing virtual environment: $venv_dir" >&2
    echo "Run: bash scripts/testing/setup_integration_env.sh $profile" >&2
    exit 1
fi

exec "$venv_dir/bin/python" -m pytest "${default_args[@]}" "$@"
