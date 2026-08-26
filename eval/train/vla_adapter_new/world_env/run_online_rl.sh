#!/usr/bin/env bash
set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORLD_ENV_DIR="train/vla_adapter_new/world_env"
LOG_DIR="$WORLD_ENV_DIR/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME-online-rl.log"
OUTPUT_DIR_BASE_DEFAULT="$WORLD_ENV_DIR/outputs/online_rl"

CUDA_DEVICES=${CUDA_DEVICES:-0}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
EXP_NAME=${EXP_NAME:-}
RUN_NAME=${RUN_NAME_OVERRIDE:-}
OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE_OVERRIDE:-$OUTPUT_DIR_BASE_DEFAULT}
ENV_ID=${ENV_ID_OVERRIDE:-HoldHammerInHandObjectScaleDown1p6-v1}
ENVS_ID=${ENVS_ID_OVERRIDE:-"['HoldHammerInHandObjectScaleDown1p6-v1','HoldWrenchInHandObjectScaleUp1p2-v1','HoldWoodBlockInHandObjectScaleDown1p6-v1','HoldHammerInHandObjectScaleUp1p6-v1','HoldHammerInHandObjectScaleDown1p4-v1','HoldWrenchInHandObjectScaleUp1p6-v1','HoldWrenchInHandObjectScaleUp1p4-v1','HoldHammerInHandObjectScaleDown1p2-v1','HoldHammerInHandObjectScaleUp1p4-v1','HoldWrenchInHandObjectScaleDown1p6-v1']"}
ENV_CHANGE_TIME_POINTS=${ENV_CHANGE_TIME_POINTS_OVERRIDE:-"[31,62,96,131,151,163,207,247,271,300]"}
WORLD_MODEL_CKPT=${WORLD_MODEL_CKPT_OVERRIDE:-${WORLD_MODEL_CHECKPOINT_OVERRIDE:-${WORLD_MODEL_CKPT:-ckpt/vla_adapter_new/world_env/outputs/world_model/20260503-075340/checkpoints/best_with_reference.pt}}}
RESUME_FROM=${RESUME_FROM_OVERRIDE:-}
SAVE_VIDEO=${SAVE_VIDEO_OVERRIDE:-false}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS_OVERRIDE:-5.1}
EARLY_STOP_ZERO_SUCCESS_MINUTES=${EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE:-45000}
NUM_ENVS=${NUM_ENVS_OVERRIDE:-${NUM_ENVS:-32}}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS_OVERRIDE:-8}
NUM_STEPS=${NUM_STEPS_OVERRIDE:-50}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS_OVERRIDE:-100000000}
NUM_MINIBATCHES=${NUM_MINIBATCHES_OVERRIDE:-16}
UPDATE_EPOCHS=${UPDATE_EPOCHS_OVERRIDE:-2}
EVAL_EVERY_UPDATES=${EVAL_EVERY_UPDATES_OVERRIDE:-50}
EVAL_EPISODES=${EVAL_EPISODES_OVERRIDE:-50}
ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE:-256}
EVAL_MICRO_BATCH_SIZE=${EVAL_MICRO_BATCH_SIZE_OVERRIDE:-256}
UPDATE_MICRO_BATCH_SIZE=${UPDATE_MICRO_BATCH_SIZE_OVERRIDE:-32}
RUN_SETUP_SMOKE=${RUN_SETUP_SMOKE_OVERRIDE:-false}
TEST_VIDEO_NUM_ENVS=${TEST_VIDEO_NUM_ENVS_OVERRIDE:-4}
TEST_VIDEO_EPISODES=${TEST_VIDEO_EPISODES_OVERRIDE:-4}

if [ -n "$EXP_NAME" ]; then
    if [[ "$EXP_NAME" == */* ]]; then
        OUTPUT_DIR_BASE="ckpt/${EXP_NAME%/*}"
        if [ -z "$RUN_NAME" ]; then
            RUN_NAME="${EXP_NAME##*/}"
        fi
    else
        OUTPUT_DIR_BASE="ckpt"
        if [ -z "$RUN_NAME" ]; then
            RUN_NAME="$EXP_NAME"
        fi
    fi
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR_BASE"

PYTHON_CMD=(
    python "$SCRIPT_DIR/online_rl.py"
    --mode train
    --seed 1
    --env-id "$ENV_ID"
    --envs-id "$ENVS_ID"
    --env-change-time-points "$ENV_CHANGE_TIME_POINTS"
    --control-mode pd_joint_delta_pos
    --reward-mode normalized_dense
    --obs-mode rgb+state_dict
    --model-dir ckpt/vla_adapter_new/LIBERO-Object
    --output-dir "$OUTPUT_DIR_BASE"
    --world-model-checkpoint "$WORLD_MODEL_CKPT"
    --static-model-checkpoint ckpt/vla_adapter_new/ours/outputs/20260502-112804/best_policy.pt
    --total-timesteps "$TOTAL_TIMESTEPS"
    --num-envs "$NUM_ENVS"
    --num-eval-envs "$NUM_EVAL_ENVS"
    --num-steps "$NUM_STEPS"
    --num-minibatches "$NUM_MINIBATCHES"
    --update-epochs "$UPDATE_EPOCHS"
    --backbone-learning-rate 3e-5
    --head-learning-rate 3e-5
    --state-learning-rate 3e-5
    --value-head-learning-rate 3e-5
    --weight-decay 1e-6
    --gamma 0.8
    --gae-lambda 0.9
    --clip-coef 0.2
    --ent-coef 0.0
    --vf-coef 0.5
    --max-grad-norm 0.5
    --target-kl 0.2
    --minibatch-target-kl-factor 1.0
    --eval-episodes "$EVAL_EPISODES"
    --eval-every-updates "$EVAL_EVERY_UPDATES"
    --max-runtime-hours "$MAX_RUNTIME_HOURS"
    --rollout-micro-batch-size "$ROLLOUT_MICRO_BATCH_SIZE"
    --eval-micro-batch-size "$EVAL_MICRO_BATCH_SIZE"
    --update-micro-batch-size "$UPDATE_MICRO_BATCH_SIZE"
    --rollout-progress-log-interval 10
    --freeze-vla-backbone false
    --backbone-warmup-updates 0
    --run-setup-smoke "$RUN_SETUP_SMOKE"
    --save-video "$SAVE_VIDEO"
    --test-video-num-envs "$TEST_VIDEO_NUM_ENVS"
    --test-video-episodes "$TEST_VIDEO_EPISODES"
    --action-dim 16
    --state-dim 105
    --static-sparsity 0.8
    --real-reward-weight 0.3
    --verified-reward-weight 0.7
    --wm-reward-weight 0.45
    --wm-success-weight 0.4
    --wm-reference-weight 0.15
    --wm-state-temperature 0.25
    --wm-reward-clip 2.0
    --success-termination-threshold 0.6
    --success-bonus 0.2
    --early-stop-zero-success-minutes "$EARLY_STOP_ZERO_SUCCESS_MINUTES"
    --cuda-device "$CUDA_DEVICES"
)

if [ -n "$RUN_NAME" ]; then
    PYTHON_CMD+=(--run-name "$RUN_NAME")
fi

if [ -n "$RESUME_FROM" ]; then
    PYTHON_CMD+=(--resume-from "$RESUME_FROM")
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
    echo "EXP_NAME=$EXP_NAME"
    echo "RUN_NAME=$RUN_NAME"
    echo "WORLD_MODEL_CKPT=$WORLD_MODEL_CKPT"
    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
