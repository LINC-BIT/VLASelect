#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$EVAL_ROOT"
source "${EVAL_ROOT}/common/interrupt_cleanup.sh"
source "${EVAL_ROOT}/common/mwe_time.sh"

CUDA_DEVICES=${CUDA_DEVICES:-0}
MODEL_SELECTION="${MODEL_SELECTION:-}"
MWE=${MWE:-0}
TAIL_LOG=${TAIL_LOG:-1}
STAMP=${ICL_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}
: "${ICL_NUM_EVAL_STEPS:=50}"
: "${ICL_MAX_EPISODE_STEPS:=50}"
: "${PROMPT_FEATURE_SCALE:=0.12}"
ICL_ENV_CHANGE_TIME_POINTS="${ICL_ENV_CHANGE_TIME_POINTS:-[1000000]}"

: "${MWE_TOTAL_RUNTIME_LIMIT_SECONDS:=300}"
vlaselect_install_cleanup_trap

MWE_PER_METHOD_RUNTIME_SECONDS=""
MWE_PER_METHOD_RUNTIME_MINUTES=""
if [[ "$MWE" == "1" ]]; then
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

run_vlaselect() {
    local exp_name="discussion/icl/${STAMP}/vlaselect"
    echo "[ICL] Running VLASelect"
    if [[ "$MWE" == "1" ]]; then
        env             CUDA_DEVICES="$CUDA_DEVICES"             EXP_NAME="$exp_name"             LAUNCH_DIRECT=1             TAIL_LOG="$TAIL_LOG"             ENV_ID_OVERRIDE=PickCubeObjectScaleUp1p2-v1             ENVS_ID_OVERRIDE="['PickCubeObjectScaleUp1p2-v1']"             ENV_CHANGE_TIME_POINTS_OVERRIDE="$EFFECTIVE_ICL_ENV_CHANGE_TIME_POINTS"             ENABLE_RICL_INJECTION=0             NUM_ENVS_OVERRIDE=8             NUM_EVAL_ENVS_OVERRIDE=2             NUM_STEPS_OVERRIDE=16             NUM_EVAL_STEPS_OVERRIDE="$ICL_NUM_EVAL_STEPS"             NUM_MINIBATCHES_OVERRIDE=2             UPDATE_EPOCHS_OVERRIDE=1             WANDB_MODE=disabled             WANDB_SILENT=true             MWE_ACTIVE_RUNTIME_ONLY=1             MAX_TIME_OVERRIDE="$MWE_PER_METHOD_RUNTIME_MINUTES"             bash "${EVAL_ROOT}/train/octo/ours_single_agent/online_rl_ours_single_agent_cl.sh"
    else
        env             CUDA_DEVICES="$CUDA_DEVICES"             EXP_NAME="$exp_name"             LAUNCH_DIRECT=1             TAIL_LOG="$TAIL_LOG"             ENV_ID_OVERRIDE=PickCubeObjectScaleUp1p2-v1             ENVS_ID_OVERRIDE="['PickCubeObjectScaleUp1p2-v1']"             ENV_CHANGE_TIME_POINTS_OVERRIDE="$EFFECTIVE_ICL_ENV_CHANGE_TIME_POINTS"             NUM_EVAL_STEPS_OVERRIDE="$ICL_NUM_EVAL_STEPS"             bash "${EVAL_ROOT}/train/octo/ours_single_agent/online_rl_ours_single_agent_cl.sh"
    fi
}

run_ricl() {
    local exp_name="discussion/icl/${STAMP}/ricl"
    echo "[ICL] Running RICL"
    if [[ "$MWE" == "1" ]]; then
        env             CUDA_DEVICES="$CUDA_DEVICES"             EXP_NAME="$exp_name"             LAUNCH_DIRECT=1             TAIL_LOG="$TAIL_LOG"             RICL_SMOKE=1             ENABLE_RICL_INJECTION=1             PROMPT_FEATURE_SCALE="$PROMPT_FEATURE_SCALE"             MAX_EPISODE_STEPS_OVERRIDE="$ICL_MAX_EPISODE_STEPS"             TOTAL_STEPS_OVERRIDE=5000000             WANDB_MODE=disabled             WANDB_SILENT=true             MWE_ACTIVE_RUNTIME_ONLY=1             MAX_RUNTIME_MINUTES_OVERRIDE="$MWE_PER_METHOD_RUNTIME_MINUTES"             bash "${EVAL_ROOT}/train/octo/ricl/online_rl_ricl.sh"
    else
        env             CUDA_DEVICES="$CUDA_DEVICES"             EXP_NAME="$exp_name"             LAUNCH_DIRECT=1             TAIL_LOG="$TAIL_LOG"             ENABLE_RICL_INJECTION=1             PROMPT_FEATURE_SCALE="$PROMPT_FEATURE_SCALE"             MAX_EPISODE_STEPS_OVERRIDE="$ICL_MAX_EPISODE_STEPS"             bash "${EVAL_ROOT}/train/octo/ricl/online_rl_ricl.sh"
    fi
}

run_vlaselect
run_ricl
