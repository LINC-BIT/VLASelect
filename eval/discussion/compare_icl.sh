#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$EVAL_ROOT"
source "${EVAL_ROOT}/common/interrupt_cleanup.sh"
source "${EVAL_ROOT}/common/mwe_time.sh"
source "${EVAL_ROOT}/common/resource_summary.sh"

CUDA_DEVICES=${CUDA_DEVICES:-0}
MODEL_SELECTION="${MODEL_SELECTION:-}"
MWE=${MWE:-0}
export MWE
TAIL_LOG=${TAIL_LOG:-1}
STAMP=${ICL_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}
: "${ICL_NUM_EVAL_STEPS:=50}"
: "${ICL_MAX_EPISODE_STEPS:=50}"
: "${PROMPT_FEATURE_SCALE:=10.0}"
: "${RICL_LEARNING_RATE:=1e-2}"
: "${RICL_MAX_SPARSITY:=0.95}"
: "${RICL_ACTOR_LOGSTD:=1.5}"
: "${ICL_PLOT_METRIC:=success_once}"
: "${ICL_PLOT_SMOOTHING:=0.8}"
ICL_ENV_CHANGE_TIME_POINTS="${ICL_ENV_CHANGE_TIME_POINTS:-[1000000]}"

: "${MWE_TOTAL_RUNTIME_LIMIT_SECONDS:=240}"
if [[ -z "${VLASELECT_MWE_USE_TRAIN_SUCCESS_ONLY+x}" ]]; then
    if [[ "$MWE" == "1" ]]; then
        export VLASELECT_MWE_USE_TRAIN_SUCCESS_ONLY=1
    else
        export VLASELECT_MWE_USE_TRAIN_SUCCESS_ONLY=0
    fi
fi
vlaselect_resource_summary_start "compare_icl.sh"
vlaselect_install_cleanup_trap

MWE_PER_METHOD_RUNTIME_SECONDS=""
MWE_PER_METHOD_RUNTIME_MINUTES=""
if [[ "$MWE" == "1" ]]; then
    if [[ ! "$MWE_TOTAL_RUNTIME_LIMIT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ICL] MWE_TOTAL_RUNTIME_LIMIT_SECONDS must be a positive integer" >&2
        exit 2
    fi
    MWE_PER_METHOD_RUNTIME_SECONDS=$((MWE_TOTAL_RUNTIME_LIMIT_SECONDS / 2))
    if [[ "$MWE_PER_METHOD_RUNTIME_SECONDS" -lt 1 ]]; then
        MWE_PER_METHOD_RUNTIME_SECONDS=1
    fi
    MWE_PER_METHOD_RUNTIME_MINUTES="$(awk -v sec="$MWE_PER_METHOD_RUNTIME_SECONDS" 'BEGIN { printf "%.6f", sec / 60.0 }')"
fi

EFFECTIVE_ICL_ENV_CHANGE_TIME_POINTS="$ICL_ENV_CHANGE_TIME_POINTS"
if [[ "$MWE" == "1" ]]; then
    EFFECTIVE_ICL_ENV_CHANGE_TIME_POINTS="$(vlaselect_convert_mwe_schedule_seconds_to_minutes "$ICL_ENV_CHANGE_TIME_POINTS")"
fi

VLASELECT_EXP_NAME="discussion/icl/${STAMP}/vlaselect"
RICL_EXP_NAME="discussion/icl/${STAMP}/ricl"

run_mwe_method() {
    local method_name="$1"
    shift

    if ! command -v timeout >/dev/null 2>&1; then
        echo "[ICL] GNU timeout is required to enforce the MWE wall-clock limit" >&2
        return 127
    fi

    local status
    if timeout -k 10s "${MWE_PER_METHOD_RUNTIME_SECONDS}s" "$@"; then
        return 0
    else
        status=$?
    fi

    if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
        echo "[ICL] ${method_name} reached the ${MWE_PER_METHOD_RUNTIME_SECONDS}s wall-clock limit"
        return 0
    fi
    return "$status"
}

run_vlaselect() {
    local exp_name="$VLASELECT_EXP_NAME"
    echo "[ICL] Running VLASelect"
    if [[ "$MWE" == "1" ]]; then
        run_mwe_method "VLASelect" env \
            CUDA_DEVICES="$CUDA_DEVICES" \
            EXP_NAME="$exp_name" \
            LAUNCH_DIRECT=1 \
            TAIL_LOG="$TAIL_LOG" \
            ENV_ID_OVERRIDE=PickCubeObjectScaleUp1p2-v1 \
            ENVS_ID_OVERRIDE="['PickCubeObjectScaleUp1p2-v1']" \
            ENV_CHANGE_TIME_POINTS_OVERRIDE="$EFFECTIVE_ICL_ENV_CHANGE_TIME_POINTS" \
            ENABLE_RICL_INJECTION=0 \
            NUM_ENVS_OVERRIDE=4 \
            NUM_EVAL_ENVS_OVERRIDE=1 \
            NUM_STEPS_OVERRIDE=16 \
            NUM_EVAL_STEPS_OVERRIDE="$ICL_NUM_EVAL_STEPS" \
            NUM_MINIBATCHES_OVERRIDE=2 \
            UPDATE_EPOCHS_OVERRIDE=1 \
            WANDB_MODE=disabled \
            WANDB_SILENT=true \
            MWE_ACTIVE_RUNTIME_ONLY=0 \
            MAX_TIME_OVERRIDE="$MWE_PER_METHOD_RUNTIME_MINUTES" \
            bash "${EVAL_ROOT}/train/octo/ours_single_agent/online_rl_ours_single_agent_cl.sh"
    else
        env             CUDA_DEVICES="$CUDA_DEVICES"             EXP_NAME="$exp_name"             LAUNCH_DIRECT=1             TAIL_LOG="$TAIL_LOG"             ENV_ID_OVERRIDE=PickCubeObjectScaleUp1p2-v1             ENVS_ID_OVERRIDE="['PickCubeObjectScaleUp1p2-v1']"             ENV_CHANGE_TIME_POINTS_OVERRIDE="$EFFECTIVE_ICL_ENV_CHANGE_TIME_POINTS"             NUM_EVAL_STEPS_OVERRIDE="$ICL_NUM_EVAL_STEPS"             bash "${EVAL_ROOT}/train/octo/ours_single_agent/online_rl_ours_single_agent_cl.sh"
    fi
}

run_ricl() {
    local exp_name="$RICL_EXP_NAME"
    echo "[ICL] Running RICL"
    if [[ "$MWE" == "1" ]]; then
        run_mwe_method "RICL" env \
            CUDA_DEVICES="$CUDA_DEVICES" \
            EXP_NAME="$exp_name" \
            LAUNCH_DIRECT=1 \
            TAIL_LOG="$TAIL_LOG" \
            RICL_SMOKE=1 \
            ENABLE_RICL_INJECTION=1 \
            PROMPT_FEATURE_SCALE="$PROMPT_FEATURE_SCALE" \
            LEARNING_RATE_OVERRIDE="$RICL_LEARNING_RATE" \
            MAX_SPARSITY_OVERRIDE="$RICL_MAX_SPARSITY" \
            ACTOR_LOGSTD_OVERRIDE="$RICL_ACTOR_LOGSTD" \
            MAX_EPISODE_STEPS_OVERRIDE="$ICL_MAX_EPISODE_STEPS" \
            TOTAL_STEPS_OVERRIDE=5000000 \
            WANDB_MODE=disabled \
            WANDB_SILENT=true \
            MWE_ACTIVE_RUNTIME_ONLY=0 \
            MAX_RUNTIME_MINUTES_OVERRIDE="$MWE_PER_METHOD_RUNTIME_MINUTES" \
            bash "${EVAL_ROOT}/train/octo/ricl/online_rl_ricl.sh"
    else
        env             CUDA_DEVICES="$CUDA_DEVICES"             EXP_NAME="$exp_name"             LAUNCH_DIRECT=1             TAIL_LOG="$TAIL_LOG"             ENABLE_RICL_INJECTION=1             PROMPT_FEATURE_SCALE="$PROMPT_FEATURE_SCALE"             LEARNING_RATE_OVERRIDE="$RICL_LEARNING_RATE"             MAX_SPARSITY_OVERRIDE="$RICL_MAX_SPARSITY"             ACTOR_LOGSTD_OVERRIDE="$RICL_ACTOR_LOGSTD"             MAX_EPISODE_STEPS_OVERRIDE="$ICL_MAX_EPISODE_STEPS"             bash "${EVAL_ROOT}/train/octo/ricl/online_rl_ricl.sh"
    fi
}

run_vlaselect
run_ricl

VLASELECT_RUN_DIR="${EVAL_ROOT}/ckpt/${VLASELECT_EXP_NAME}/[agent]"
RICL_RUN_DIR="${EVAL_ROOT}/ckpt/${RICL_EXP_NAME}/[agent]"
PLOT_PATH="${EVAL_ROOT}/ckpt/discussion/icl/${STAMP}/icl_accuracy.png"
SUMMARY_PATH="${EVAL_ROOT}/ckpt/discussion/icl/${STAMP}/icl_summary.json"

require_metrics_history() {
    local method_name="$1"
    local run_dir="$2"
    local history_path="${run_dir}/metrics_history.json"

    if [[ -s "$history_path" ]]; then
        return 0
    fi

    echo "[ICL] error: ${method_name} produced no metrics history: ${history_path}" >&2
    if [[ "$MWE" == "1" ]]; then
        echo "[ICL] The method reached its wall-clock limit before a success metric was saved." >&2
        echo "[ICL] Retry with a larger MWE_TOTAL_RUNTIME_LIMIT_SECONDS (current: ${MWE_TOTAL_RUNTIME_LIMIT_SECONDS})." >&2
    fi
    return 1
}

require_metrics_history "VLASelect" "$VLASELECT_RUN_DIR"
require_metrics_history "RICL" "$RICL_RUN_DIR"

python "${SCRIPT_DIR}/plot_icl.py" \
    --vlaselect-run-dir "$VLASELECT_RUN_DIR" \
    --ricl-run-dir "$RICL_RUN_DIR" \
    --output "$PLOT_PATH" \
    --summary-output "$SUMMARY_PATH" \
    --metric "$ICL_PLOT_METRIC" \
    --smoothing "$ICL_PLOT_SMOOTHING"
echo "[ICL] Comparison plot: ${PLOT_PATH}"
echo "[ICL] Comparison summary: ${SUMMARY_PATH}"
