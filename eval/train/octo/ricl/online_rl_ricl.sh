#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/octo/ricl/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME.log"
mkdir -p "$LOG_DIR"

CUDA_DEVICES=${CUDA_DEVICES:-0}
EXP_NAME=${EXP_NAME:-}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
RICL_SMOKE=${RICL_SMOKE:-1}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
TOTAL_STEPS=${TOTAL_STEPS:-5000000}
NUM_ENVS=${NUM_ENVS:-128}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS:-8}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-16}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-4}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-50}
LR=${LR:-3e-4}
PRETRAINED_CHECKPOINT_PATH=${PRETRAINED_CHECKPOINT_PATH:-ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt}

if [ "$RICL_SMOKE" = "1" ]; then
    TOTAL_STEPS=${TOTAL_STEPS_OVERRIDE:-512}
    NUM_ENVS=${NUM_ENVS_OVERRIDE:-8}
    NUM_EVAL_ENVS=${NUM_EVAL_ENVS_OVERRIDE:-2}
    ROLLOUT_STEPS=${ROLLOUT_STEPS_OVERRIDE:-4}
    UPDATE_EPOCHS=${UPDATE_EPOCHS_OVERRIDE:-1}
    MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS_OVERRIDE:-10}
    LR=${LR_OVERRIDE:-3e-4}
fi

EXTRA_ARGS=()
RUN_DIR=""
if [ -n "$EXP_NAME" ]; then
    EXTRA_ARGS+=(--exp-name "$EXP_NAME")
    RUN_DIR="ckpt/$EXP_NAME"
fi

PYTHON_CMD=(
    python -u -m train.octo.ricl.online_rl
    --task-name PickCubeObjectScaleUp1p2-v1
    --total-steps "$TOTAL_STEPS"
    --num-envs "$NUM_ENVS"
    --num-eval-envs "$NUM_EVAL_ENVS"
    --rollout-steps "$ROLLOUT_STEPS"
    --update-epochs "$UPDATE_EPOCHS"
    --num_minibatch 4
    --save-interval-per-rollout 4
    --max-episode-steps "$MAX_EPISODE_STEPS"
    --lr "$LR"
    --pretrained-checkpoint "$PRETRAINED_CHECKPOINT_PATH"
    --normalize-state
    --reset-logstd
    --ricl-bank-capacity 4096
    --ricl-bank-add-per-iter 128
    --ricl-num-neighbors 4
    --ricl-retrieval-temperature 10.0
    --ricl-state-dim-cap 32
    --ricl-context-hidden-dim 128
    "${EXTRA_ARGS[@]}"
)

if [ "$LAUNCH_DIRECT" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
    cd "$EVAL_ROOT"
    exec "${PYTHON_CMD[@]}"
else
    cd "$EVAL_ROOT"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup "${PYTHON_CMD[@]}" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo "TRAIN_PID=$TRAIN_PID"
    echo "RUN_DIR=$RUN_DIR"
    echo "LOG_FILE=$LOG_FILE"
    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
