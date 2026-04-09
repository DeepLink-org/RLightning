#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

usage() {
    cat <<'EOF'
Usage: bash scripts/testing/run_all_tests.sh [options] [targets...]

Runs the local test workflow. With no targets, this runs:
  1. unit
  2. integration core
  3. integration isaaclab
  4. e2e

Targets:
  unit                 Run tests/unit in the main .venv.
  integration          Run both integration profiles: core + isaaclab.
  integration-core     Run only the core integration profile.
  integration-isaaclab Run only the isaaclab integration profile.
  e2e                  Run tests/e2e in the main .venv.
  all                  Run the full workflow. This is the default.

Options:
  -q, --quiet       Use concise pytest output. This is the default.
  -v, --verbose     Use verbose pytest output and print live E2E logs.
      --no-setup    Do not auto-create missing integration environments.
  -h, --help        Show this help message.

Examples:
  bash scripts/testing/run_all_tests.sh
  bash scripts/testing/run_all_tests.sh --verbose
  bash scripts/testing/run_all_tests.sh unit
  bash scripts/testing/run_all_tests.sh integration
  bash scripts/testing/run_all_tests.sh integration-core integration-isaaclab
  bash scripts/testing/run_all_tests.sh e2e
EOF
}

verbosity="quiet"
auto_setup="1"
targets=()
resolved_targets=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -q|--quiet)
            verbosity="quiet"
            shift
            ;;
        -v|--verbose)
            verbosity="verbose"
            shift
            ;;
        --no-setup)
            auto_setup="0"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        unit|integration|integration-core|integration-isaaclab|e2e|all)
            targets+=("$1")
            shift
            ;;
        *)
            echo "Unknown option or target: $1" >&2
            usage
            exit 1
            ;;
    esac
done

log_section() {
    echo
    echo "==> $1"
}

ensure_main_env() {
    if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
        echo "Missing main test environment: $ROOT_DIR/.venv" >&2
        echo "Prepare it first, for example:" >&2
        echo "  uv venv .venv" >&2
        echo "  uv pip install --python .venv/bin/python --torch-backend cpu -e '.[dev]'" >&2
        exit 1
    fi
}

ensure_integration_env() {
    local profile="$1"
    local venv_dir="$ROOT_DIR/.venv-int-$profile"

    if [[ -x "$venv_dir/bin/python" ]]; then
        return
    fi

    if [[ "$auto_setup" != "1" ]]; then
        echo "Missing integration environment: $venv_dir" >&2
        echo "Run: bash scripts/testing/setup_integration_env.sh $profile" >&2
        exit 1
    fi

    log_section "Preparing integration environment: $profile"
    bash "$ROOT_DIR/scripts/testing/setup_integration_env.sh" "$profile"
}

append_target_once() {
    local target="$1"
    local existing
    for existing in "${resolved_targets[@]}"; do
        if [[ "$existing" == "$target" ]]; then
            return
        fi
    done
    resolved_targets+=("$target")
}

resolve_targets() {
    local requested_targets=("${targets[@]}")

    if [[ ${#requested_targets[@]} -eq 0 ]]; then
        requested_targets=(all)
    fi

    local target
    for target in "${requested_targets[@]}"; do
        case "$target" in
            all)
                append_target_once unit
                append_target_once integration-core
                append_target_once integration-isaaclab
                append_target_once e2e
                ;;
            integration)
                append_target_once integration-core
                append_target_once integration-isaaclab
                ;;
            unit|integration-core|integration-isaaclab|e2e)
                append_target_once "$target"
                ;;
            *)
                echo "Unknown target: $target" >&2
                usage
                exit 1
                ;;
        esac
    done
}

run_unit_tests() {
    log_section "Unit tests"
    if [[ "$verbosity" == "verbose" ]]; then
        "$ROOT_DIR/.venv/bin/python" -m pytest -vv -s -ra tests/unit
    else
        "$ROOT_DIR/.venv/bin/python" -m pytest -q tests/unit
    fi
}

run_integration_tests() {
    local profile="$1"
    log_section "Integration tests ($profile)"
    if [[ "$verbosity" == "verbose" ]]; then
        bash "$ROOT_DIR/scripts/testing/run_integration_tests.sh" "$profile" -vv -s -ra
    else
        bash "$ROOT_DIR/scripts/testing/run_integration_tests.sh" "$profile" -q
    fi
}

run_e2e_tests() {
    log_section "E2E tests"
    if [[ "$verbosity" == "verbose" ]]; then
        RLIGHTNING_E2E_LIVE_LOGS=1 RAY_DEDUP_LOGS=0 \
            "$ROOT_DIR/.venv/bin/python" -m pytest -o addopts='' -vv -s -ra tests/e2e
    else
        "$ROOT_DIR/.venv/bin/python" -m pytest -o addopts='' -q tests/e2e
    fi
}

resolve_targets

need_main_env="0"
need_core_integration="0"
need_isaaclab_integration="0"

for target in "${resolved_targets[@]}"; do
    case "$target" in
        unit|e2e)
            need_main_env="1"
            ;;
        integration-core)
            need_core_integration="1"
            ;;
        integration-isaaclab)
            need_isaaclab_integration="1"
            ;;
    esac
done

if [[ "$need_main_env" == "1" ]]; then
    ensure_main_env
fi
if [[ "$need_core_integration" == "1" ]]; then
    ensure_integration_env core
fi
if [[ "$need_isaaclab_integration" == "1" ]]; then
    ensure_integration_env isaaclab
fi

for target in "${resolved_targets[@]}"; do
    case "$target" in
        unit)
            run_unit_tests
            ;;
        integration-core)
            run_integration_tests core
            ;;
        integration-isaaclab)
            run_integration_tests isaaclab
            ;;
        e2e)
            run_e2e_tests
            ;;
    esac
done

log_section "All tests passed"
