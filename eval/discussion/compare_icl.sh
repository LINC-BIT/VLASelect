#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${EVAL_ROOT}/common/interrupt_cleanup.sh"

RICL_SMOKE=${RICL_SMOKE:-0}
CUDA_DEVICES=${CUDA_DEVICES:-0}
MODEL_SELECTION="${MODEL_SELECTION:-}"
MWE=${MWE:-0}

: "${MWE_RUNTIME_LIMIT_SECONDS:=300}"
export MWE_RUNTIME_LIMIT_SECONDS
if [[ "$MWE" == "1" && "${MWE_TIMEOUT_APPLIED:-0}" != "1" ]]; then
    if command -v timeout >/dev/null 2>&1; then
        export MWE_TIMEOUT_APPLIED=1
        exec timeout --preserve-status -k 10s "${MWE_RUNTIME_LIMIT_SECONDS}s" bash "$SCRIPT_PATH" "$@"
    fi
    echo "[warn] timeout command not found; MWE runtime is not hard-capped" >&2
fi
if [[ "$MWE" == "1" ]]; then
    RICL_SMOKE=1
fi
vlaselect_install_cleanup_trap
LAUNCH_DIRECT=1 \
TAIL_LOG=1 \
RICL_SMOKE="$RICL_SMOKE" \
CUDA_DEVICES="$CUDA_DEVICES" \
bash "${EVAL_ROOT}/train/octo/ricl/online_rl_ricl.sh"
