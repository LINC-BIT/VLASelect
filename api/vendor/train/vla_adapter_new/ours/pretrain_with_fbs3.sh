#!/usr/bin/env bash
set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OURS_DIR="train/vla_adapter_new/ours"
LOG_DIR="$OURS_DIR/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME-pretrain-with-fbs3.log"
OUTPUT_DIR_BASE="$OURS_DIR/outputs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR_BASE"

CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
ENV_ID=${ENV_ID_OVERRIDE:-HoldCubeInHand-v1}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS_OVERRIDE:-50}
EARLY_STOP_ZERO_SUCCESS_MINUTES=${EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE:-45000}
RUN_NAME=${RUN_NAME_OVERRIDE:-}
RESUME_FROM=${RESUME_FROM_OVERRIDE:-train/vla_adapter_new/ours/pretrained_model_with_fbs.pth}
SAVE_VIDEO=${SAVE_VIDEO_OVERRIDE:-false}
TARGET_KL=${TARGET_KL_OVERRIDE:-0.2}
MINIBATCH_TARGET_KL_FACTOR=${MINIBATCH_TARGET_KL_FACTOR_OVERRIDE:-1.5}
UPDATE_MICRO_BATCH_SIZE=${UPDATE_MICRO_BATCH_SIZE_OVERRIDE:-8}
EVAL_MICRO_BATCH_SIZE=${EVAL_MICRO_BATCH_SIZE_OVERRIDE:-64}
HEAD_LEARNING_RATE=${HEAD_LEARNING_RATE_OVERRIDE:-1e-6}
STATE_LEARNING_RATE=${STATE_LEARNING_RATE_OVERRIDE:-1e-6}
VALUE_HEAD_LEARNING_RATE=${VALUE_HEAD_LEARNING_RATE_OVERRIDE:-3e-6}
BACKBONE_LEARNING_RATE=${BACKBONE_LEARNING_RATE_OVERRIDE:-1e-6}
BACKBONE_WARMUP_UPDATES=${BACKBONE_WARMUP_UPDATES_OVERRIDE:-0}
POLICY_TRUNK_WARMUP_UPDATES=${POLICY_TRUNK_WARMUP_UPDATES_OVERRIDE:-0}
NUM_MINIBATCHES=${NUM_MINIBATCHES_OVERRIDE:-1}
UPDATE_EPOCHS=${UPDATE_EPOCHS_OVERRIDE:-1}
MAX_GRAD_NORM=${MAX_GRAD_NORM_OVERRIDE:-0.05}
KL_BACKTRACK_ATTEMPTS=${KL_BACKTRACK_ATTEMPTS_OVERRIDE:-8}
KL_BACKTRACK_FACTOR=${KL_BACKTRACK_FACTOR_OVERRIDE:-0.5}
KL_PROBE_BATCH_SIZE=${KL_PROBE_BATCH_SIZE_OVERRIDE:-128}
KL_INITIAL_STEP_SCALE=${KL_INITIAL_STEP_SCALE_OVERRIDE:-5e-5}
KL_STEP_SCALE_GROWTH=${KL_STEP_SCALE_GROWTH_OVERRIDE:-2.0}

PYTHON_CMD=(
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_DIR/pretrain_with_fbs3.py"
    --mode train
    --seed 1
    --env-id "$ENV_ID"
    --control-mode pd_joint_delta_pos
    --reward-mode normalized_dense
    --obs-mode rgb+state_dict
    --model-dir eval/ckpt/vla_adapter_new/LIBERO-Object
    --output-dir "$OUTPUT_DIR_BASE"
    --total-timesteps 100000000
    --num-envs 256
    --num-eval-envs 8
    --num-steps 50
    --num-minibatches "$NUM_MINIBATCHES"
    --update-epochs "$UPDATE_EPOCHS"
    --learning-rate 3e-5
    --head-learning-rate "$HEAD_LEARNING_RATE"
    --state-learning-rate "$STATE_LEARNING_RATE"
    --value-head-learning-rate "$VALUE_HEAD_LEARNING_RATE"
    --backbone-learning-rate "$BACKBONE_LEARNING_RATE"
    --weight-decay 1e-6
    --gamma 0.8
    --gae-lambda 0.9
    --clip-coef 0.2
    --ent-coef 0.0
    --vf-coef 0.5
    --max-grad-norm "$MAX_GRAD_NORM"
    --target-kl "$TARGET_KL"
    --minibatch-target-kl-factor "$MINIBATCH_TARGET_KL_FACTOR"
    --kl-backtrack-attempts "$KL_BACKTRACK_ATTEMPTS"
    --kl-backtrack-factor "$KL_BACKTRACK_FACTOR"
    --kl-probe-batch-size "$KL_PROBE_BATCH_SIZE"
    --kl-initial-step-scale "$KL_INITIAL_STEP_SCALE"
    --kl-step-scale-growth "$KL_STEP_SCALE_GROWTH"
    --eval-episodes 50
    --eval-every-updates 20
    --max-runtime-hours "$MAX_RUNTIME_HOURS"
    --rollout-micro-batch-size 256
    --eval-micro-batch-size "$EVAL_MICRO_BATCH_SIZE"
    --update-micro-batch-size "$UPDATE_MICRO_BATCH_SIZE"
    --rollout-progress-log-interval 10
    --freeze-vla-backbone false
    --backbone-warmup-updates "$BACKBONE_WARMUP_UPDATES"
    --policy-trunk-warmup-updates "$POLICY_TRUNK_WARMUP_UPDATES"
    --run-setup-smoke false
    --save-video "$SAVE_VIDEO"
    --save-train-video-freq 10
    --train-video-num-envs 4
    --test-video-num-envs 4
    --test-video-episodes 4
    --action-dim 16
    --state-dim 105
    --resume-from "$RESUME_FROM"
    --early-stop-zero-success-minutes "$EARLY_STOP_ZERO_SUCCESS_MINUTES"
    --cuda-device "$CUDA_DEVICES"
)

if [ -n "$RUN_NAME" ]; then
    PYTHON_CMD+=(--run-name "$RUN_NAME")
fi

if [ "$LAUNCH_DIRECT" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
    export TORCH_DIST_TIMEOUT_HOURS=${TORCH_DIST_TIMEOUT_HOURS:-6}
    export NCCL_SHM_DISABLE=1
    export NCCL_IB_DISABLE=1
    exec "${PYTHON_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES TORCH_DIST_TIMEOUT_HOURS=${TORCH_DIST_TIMEOUT_HOURS:-6} NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 nohup "${PYTHON_CMD[@]}" > "$LOG_FILE" 2>&1 &

    TRAIN_PID=$!
    echo "TRAIN_PID=$TRAIN_PID"
    echo "OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE"
    echo "LOG_FILE=$LOG_FILE"
    echo "ENV_ID=$ENV_ID"
    echo "RUN_NAME=$RUN_NAME"

    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
