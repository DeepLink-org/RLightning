#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

usage() {
    cat <<'EOF'
Usage: bash scripts/testing/setup_integration_env.sh <profile>

Profiles:
  core      Create .venv-int-core with pytest + mujoco + ale extras.
  isaaclab  Create .venv-int-isaaclab with pytest + isaaclab + humanoid extras and humanoid-dev group.

Examples:
  bash scripts/testing/setup_integration_env.sh core
  bash scripts/testing/setup_integration_env.sh isaaclab
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

profile="$1"
extra_args=()

case "$profile" in
    core)
        venv_dir="$ROOT_DIR/.venv-int-core"
        project_spec='.[mujoco,ale]'
        ;;
    isaaclab)
        venv_dir="$ROOT_DIR/.venv-int-isaaclab"
        project_spec='.[isaaclab,humanoid]'
        extra_args=(--group humanoid-dev)
        ;;
    *)
        echo "Unknown profile: $profile" >&2
        usage
        exit 1
        ;;
esac

uv venv --allow-existing "$venv_dir"
uv pip install --python "$venv_dir/bin/python" -e "$project_spec" "pytest>=8.3.5" "${extra_args[@]:-}"

cat <<EOF
Prepared integration environment:
  profile: $profile
  venv:    $venv_dir

Run tests with:
  bash scripts/testing/run_integration_tests.sh $profile
EOF
