#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "$REPO_ROOT/eval"
source "${REPO_ROOT}/eval/common/interrupt_cleanup.sh"
source "${REPO_ROOT}/eval/common/sanity_check.sh"

FAMILY=${MODEL_SIZE_LIMIT_FAMILY:-tinyvla}
MODEL_DIR=${MODEL_SIZE_LIMIT_MODEL_DIR:-}
SPARSITIES=${MODEL_SIZE_LIMIT_SPARSITIES:-0.00,0.25,0.50,0.75,0.90}
BUDGET_GB=${MODEL_SIZE_LIMIT_BUDGET_GB:-8,16,24,32,40,48,80}
TRAIN_BATCH_SIZE=${MODEL_SIZE_LIMIT_TRAIN_BATCH_SIZE:-2}
FBS_R=${MODEL_SIZE_LIMIT_FBS_R:-16}
DEVICE=${MODEL_SIZE_LIMIT_DEVICE:-}
DTYPE=${MODEL_SIZE_LIMIT_DTYPE:-bfloat16}
OUTPUT_DIR=${MODEL_SIZE_LIMIT_OUTPUT_DIR:-discussion/results}
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
    FAMILY=${MODEL_SIZE_LIMIT_FAMILY:-tinyvla}
    SPARSITIES=${MODEL_SIZE_LIMIT_SPARSITIES:-0.00,0.50}
    BUDGET_GB=${MODEL_SIZE_LIMIT_BUDGET_GB:-8,16}
    TRAIN_BATCH_SIZE=${MODEL_SIZE_LIMIT_TRAIN_BATCH_SIZE:-1}
fi
vlaselect_install_cleanup_trap
vlaselect_run_sanity_check "sweep_model_size.sh" "${REPO_ROOT}/eval" "$MWE" "16" "8"

CMD=(
    python discussion/sweep_model_size.py
    --family "$FAMILY"
    --sparsities "$SPARSITIES"
    --budget-gb "$BUDGET_GB"
    --train-batch-size "$TRAIN_BATCH_SIZE"
    --fbs-r "$FBS_R"
    --dtype "$DTYPE"
    --output-dir "$OUTPUT_DIR"
)

if [ -n "$MODEL_DIR" ]; then
    CMD+=(--model-dir "$MODEL_DIR")
fi

if [ -n "$DEVICE" ]; then
    CMD+=(--device "$DEVICE")
fi

printf '[run] family=%s device=%s train_batch_size=%s sparsities=%s budget_gb=%s\n' "$FAMILY" "${DEVICE:-auto}" "$TRAIN_BATCH_SIZE" "$SPARSITIES" "$BUDGET_GB"
"${CMD[@]}"
