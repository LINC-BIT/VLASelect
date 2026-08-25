#!/usr/bin/env bash
set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="$SCRIPT_DIR/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME-conrft-hold-cube.log"
CUDA_DEVICES=${CUDA_DEVICES:-1}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
EXP_NAME=${EXP_NAME:-}
OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE_OVERRIDE:-"train/vla_adapter_new/conrft/outputs"}
ENV_ID=${ENV_ID_OVERRIDE:-HoldHammerInHandObjectScaleDown1p6-v1}
ENV_IDS=${ENV_IDS_OVERRIDE:-"['HoldHammerInHandObjectScaleDown1p6-v1','HoldWrenchInHandObjectScaleUp1p2-v1','HoldWoodBlockInHandObjectScaleDown1p6-v1','HoldHammerInHandObjectScaleUp1p6-v1','HoldHammerInHandObjectScaleDown1p4-v1','HoldWrenchInHandObjectScaleUp1p6-v1','HoldWrenchInHandObjectScaleUp1p4-v1','HoldHammerInHandObjectScaleDown1p2-v1','HoldHammerInHandObjectScaleUp1p4-v1','HoldWrenchInHandObjectScaleDown1p6-v1']"}
ENV_CHANGE_TIME_POINTS=${ENV_CHANGE_TIME_POINTS_OVERRIDE:-"[31,62,96,131,151,163,207,247,271,300]"}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS_OVERRIDE:-5.1}
EARLY_STOP_ZERO_SUCCESS_MINUTES=${EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE:-45000}
RUN_NAME=${RUN_NAME_OVERRIDE:-}
SAVE_VIDEO=${SAVE_VIDEO_OVERRIDE:-false}
NUM_ENVS=${NUM_ENVS_OVERRIDE:-256}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS_OVERRIDE:-8}
NUM_STEPS=${NUM_STEPS_OVERRIDE:-50}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS_OVERRIDE:-100000000}
NUM_MINIBATCHES=${NUM_MINIBATCHES_OVERRIDE:-16}
UPDATE_EPOCHS=${UPDATE_EPOCHS_OVERRIDE:-2}
EVAL_EVERY_UPDATES=${EVAL_EVERY_UPDATES_OVERRIDE:-20}
EVAL_EPISODES=${EVAL_EPISODES_OVERRIDE:-50}
ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE:-256}
EVAL_MICRO_BATCH_SIZE=${EVAL_MICRO_BATCH_SIZE_OVERRIDE:-256}
UPDATE_MICRO_BATCH_SIZE=${UPDATE_MICRO_BATCH_SIZE_OVERRIDE:-32}
RUN_SETUP_SMOKE=${RUN_SETUP_SMOKE_OVERRIDE:-false}

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
    python -u "$SCRIPT_DIR/online_rl.py"
    --mode train
    --seed 1
    --env-id "$ENV_ID"
    --envs-id "$ENV_IDS"
    --env-change-time-points "$ENV_CHANGE_TIME_POINTS"
    --control-mode pd_joint_delta_pos
    --reward-mode normalized_dense
    --obs-mode rgb+state_dict
    --model-dir ckpt/vla_adapter_new/LIBERO-Object
    --output-dir "$OUTPUT_DIR_BASE"
    --teacher-checkpoint ckpt/vla_adapter_new/model_impl/outputs/ppo_hold_cube_in_hand/20260430-103518/best_policy.pt
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
    --supervised-learning-rate 3e-5
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
    --action-dim 16
    --state-dim 105
    --early-stop-zero-success-minutes "$EARLY_STOP_ZERO_SUCCESS_MINUTES"
    --cuda-device "$CUDA_DEVICES"
    --supervised-updates-per-iter 8
    --supervised-batch-size 256
    --supervised-online-ratio 0.5
    --warmup-success-steps-before-supervised 100
    --min-success-steps-for-supervised 10
    --online-buffer-capacity 20000
    --expert-buffer-capacity 40000
    --expert-target-success-trajectories 24
    --expert-collect-num-envs 8
    --expert-collect-max-steps 5000
    --expert-collect-seed 0
    --static-sparsity 0.8
)

if [ -n "$RUN_NAME" ]; then
    PYTHON_CMD+=(--run-name "$RUN_NAME")
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
    echo "ENV_ID=$ENV_ID"
    echo "ENV_IDS=$ENV_IDS"
    echo "ENV_CHANGE_TIME_POINTS=$ENV_CHANGE_TIME_POINTS"
    echo "EXP_NAME=$EXP_NAME"
    echo "RUN_NAME=$RUN_NAME"

    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
