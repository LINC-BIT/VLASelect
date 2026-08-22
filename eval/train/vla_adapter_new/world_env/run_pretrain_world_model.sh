#!/usr/bin/env bash
set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORLD_ENV_DIR="train/vla_adapter_new/world_env"
LOG_DIR="$WORLD_ENV_DIR/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME-pretrain-world-model.log"
OUTPUT_DIR_BASE="$WORLD_ENV_DIR/outputs/world_model"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR_BASE"

CUDA_DEVICES=${CUDA_DEVICES:-1}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
RUN_NAME=${RUN_NAME_OVERRIDE:-}
DATASET_PATH=${DATASET_PATH_OVERRIDE:-}

PYTHON_CMD=(
    python "$SCRIPT_DIR/pretrain_world_model.py"
    --mode all
    --seed 1
    --env-id HoldCubeInHand-v1
    --control-mode pd_joint_delta_pos
    --reward-mode normalized_dense
    --obs-mode rgb+state_dict
    --model-dir eval/ckpt/vla_adapter_new/LIBERO-Object
    --teacher-checkpoint ckpt/vla_adapter_new/model_impl/outputs/ppo_hold_cube_in_hand/20260430-103518/best_policy.pt
    --output-dir "$OUTPUT_DIR_BASE"
    --target-transitions 4000
    --target-episodes 80
    --num-collect-envs 8
    --image-size 64
    --latent-dim 256
    --epochs 10
    --batch-size 64
    --learning-rate 3e-4
    --weight-decay 1e-5
    --eval-micro-batch-size 32
    --cuda-device "$CUDA_DEVICES"
)

if [ -n "$RUN_NAME" ]; then
    PYTHON_CMD+=(--run-name "$RUN_NAME")
fi

if [ -n "$DATASET_PATH" ]; then
    PYTHON_CMD+=(--dataset-path "$DATASET_PATH")
fi

if [ "$LAUNCH_DIRECT" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
    exec "${PYTHON_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup "${PYTHON_CMD[@]}" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo "TRAIN_PID=$TRAIN_PID"
    echo "OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE"
    echo "LOG_FILE=$LOG_FILE"
    echo "RUN_NAME=$RUN_NAME"
    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
