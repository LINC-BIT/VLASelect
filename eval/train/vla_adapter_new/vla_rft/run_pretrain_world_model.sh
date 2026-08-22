#!/usr/bin/env bash
set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR_BASE="train/vla_adapter_new/vla_rft/outputs/world_model"
LOG_DIR="$SCRIPT_DIR/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME-pretrain-world-model.log"
mkdir -p "$OUTPUT_DIR_BASE" "$LOG_DIR"

CUDA_DEVICES=${CUDA_DEVICES:-0}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
RUN_NAME=${RUN_NAME_OVERRIDE:-latest}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
REUSE_DATASET=${REUSE_DATASET_OVERRIDE:-false}
COLLECT_EPISODES=${COLLECT_EPISODES_OVERRIDE:-128}
EPOCHS=${EPOCHS_OVERRIDE:-10}
DATASET_PATH=${DATASET_PATH_OVERRIDE:-$OUTPUT_DIR_BASE/$RUN_NAME/teacher_rollouts.pt}

PYTHON_CMD=(
    python "$SCRIPT_DIR/pretrain_world_model.py"
    --mode train
    --seed 1
    --env-id HoldCubeInHand-v1
    --control-mode pd_joint_delta_pos
    --reward-mode normalized_dense
    --obs-mode rgb+state_dict
    --model-dir eval/ckpt/vla_adapter_new/LIBERO-Object
    --teacher-checkpoint ckpt/vla_adapter_new/model_impl/outputs/ppo_hold_cube_in_hand/20260430-103518/best_policy.pt
    --output-dir "$OUTPUT_DIR_BASE"
    --run-name "$RUN_NAME"
    --dataset-path "$DATASET_PATH"
    --reuse-dataset "$REUSE_DATASET"
    --num-collect-envs 8
    --collect-episodes "$COLLECT_EPISODES"
    --collect-micro-batch-size 32
    --batch-size 64
    --epochs "$EPOCHS"
    --learning-rate 3e-4
    --weight-decay 1e-5
    --latent-dim 256
    --max-reference-bank-size 256
    --cuda-device "$CUDA_DEVICES"
    --action-dim 16
    --state-dim 105
)

if [ "$LAUNCH_DIRECT" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
    exec "${PYTHON_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup "${PYTHON_CMD[@]}" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo "TRAIN_PID=$TRAIN_PID"
    echo "OUTPUT_DIR=$OUTPUT_DIR_BASE/$RUN_NAME"
    echo "LOG_FILE=$LOG_FILE"
    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
