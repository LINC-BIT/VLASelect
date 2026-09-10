#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "$REPO_ROOT/eval"
source "${REPO_ROOT}/eval/common/interrupt_cleanup.sh"
source "${REPO_ROOT}/eval/common/sanity_check.sh"
source "${REPO_ROOT}/eval/common/resource_summary.sh"

FAMILY=${DIRECT_TEST_FAMILY:-${MODEL_SIZE_LIMIT_FAMILY:-tinyvla}}
MODEL_DIR=${DIRECT_TEST_MODEL_DIR:-${MODEL_SIZE_LIMIT_MODEL_DIR:-}}
TARGET_ORIGINAL_GB=${DIRECT_TEST_TARGET_ORIGINAL_GB:-11.3}
BUDGET_GB=${DIRECT_TEST_BUDGET_GB:-32}
SPARSITY=${DIRECT_TEST_SPARSITY:-0.90}
TRAIN_BATCH_SIZE=${DIRECT_TEST_TRAIN_BATCH_SIZE:-${MODEL_SIZE_LIMIT_TRAIN_BATCH_SIZE:-1}}
FBS_R=${DIRECT_TEST_FBS_R:-${MODEL_SIZE_LIMIT_FBS_R:-16}}
BUILD_DEVICE=${DIRECT_TEST_BUILD_DEVICE:-cuda}
TRAIN_DEVICE=${DIRECT_TEST_TRAIN_DEVICE:-${MODEL_SIZE_LIMIT_DEVICE:-}}
DTYPE=${DIRECT_TEST_DTYPE:-${MODEL_SIZE_LIMIT_DTYPE:-bfloat16}}
OUTPUT_DIR=${DIRECT_TEST_OUTPUT_DIR:-${MODEL_SIZE_LIMIT_OUTPUT_DIR:-discussion/results}}
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
    TARGET_ORIGINAL_GB=${DIRECT_TEST_TARGET_ORIGINAL_GB:-0.03}
    BUDGET_GB=${DIRECT_TEST_BUDGET_GB:-0.10}
    SPARSITY=${DIRECT_TEST_SPARSITY:-0.50}
    TRAIN_BATCH_SIZE=${DIRECT_TEST_TRAIN_BATCH_SIZE:-1}
    MODEL_DIR=${DIRECT_TEST_MODEL_DIR:-/tmp/vlaselect_missing_model_for_direct_test_mwe}
fi

vlaselect_resource_summary_start "sweep_model_size_direct_test.sh"
vlaselect_install_cleanup_trap
vlaselect_run_sanity_check "sweep_model_size_direct_test.sh" "${REPO_ROOT}/eval" "$MWE" "16" "8"

CMD=(
    python discussion/sweep_model_size_direct_test.py
    --family "$FAMILY"
    --target-original-gb "$TARGET_ORIGINAL_GB"
    --budget-gb "$BUDGET_GB"
    --sparsity "$SPARSITY"
    --train-batch-size "$TRAIN_BATCH_SIZE"
    --fbs-r "$FBS_R"
    --build-device "$BUILD_DEVICE"
    --dtype "$DTYPE"
    --output-dir "$OUTPUT_DIR"
)

if [ -n "$MODEL_DIR" ]; then
    CMD+=(--model-dir "$MODEL_DIR")
fi

if [ -n "$TRAIN_DEVICE" ]; then
    CMD+=(--train-device "$TRAIN_DEVICE")
fi

printf '[run] family=%s build_device=%s train_device=%s target_original_gb=%s budget_gb=%s sparsity=%s train_batch_size=%s\n' \
    "$FAMILY" "$BUILD_DEVICE" "${TRAIN_DEVICE:-auto}" "$TARGET_ORIGINAL_GB" "$BUDGET_GB" "$SPARSITY" "$TRAIN_BATCH_SIZE"
"${CMD[@]}"
