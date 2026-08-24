#!/usr/bin/env bash
set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OURS_DIR="$SCRIPT_DIR/../ours"
LOG_DIR="$SCRIPT_DIR/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME-edgevla-edgeta-online-rl-cl.log"
OUTPUT_DIR_BASE_DEFAULT="train/edgevla/edgeta/outputs/online_rl_cl"

CUDA_DEVICES=${CUDA_DEVICES:-1,2,3,4}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
EXP_NAME=${EXP_NAME:-}
RUN_NAME=${RUN_NAME_OVERRIDE:-}
OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE_OVERRIDE:-$OUTPUT_DIR_BASE_DEFAULT}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS_OVERRIDE:-400}
SAVE_VIDEO=${SAVE_VIDEO_OVERRIDE:-false}
CONTINUE_TRAIN_FROM=${CONTINUE_TRAIN_FROM_OVERRIDE:-}
LARGE_AGENT_CHECKPOINT=${LARGE_AGENT_CHECKPOINT_OVERRIDE:-ckpt/edgevla/ours/outputs/bc_unitree_g1_lift_apple_fbs/20260511-171959/best_policy.pt}
ENV_ID=${ENV_ID_OVERRIDE:-UnitreeG1LiftApple-v1}
ENVS_ID=${ENVS_ID_OVERRIDE:-"['UnitreeG1LiftCubeObjectScaleDown1p3-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeObjectPurple-v1','UnitreeG1LiftSphereLightStronger50-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectScaleDown1p1-v1','UnitreeG1LiftSphereObjectScaleDown1p3-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectPurple-v1']"}
ENV_CHANGE_TIME_POINTS=${ENV_CHANGE_TIME_POINTS_OVERRIDE:-"[31,62,96,131,151,163,207,247,271,300]"}
NUM_ENVS=${NUM_ENVS_OVERRIDE:-128}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS_OVERRIDE:-8}
NUM_STEPS=${NUM_STEPS_OVERRIDE:-64}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS_OVERRIDE:-100000000}
NUM_MINIBATCHES=${NUM_MINIBATCHES_OVERRIDE:-16}
UPDATE_EPOCHS=${UPDATE_EPOCHS_OVERRIDE:-2}
EVAL_EVERY_UPDATES=${EVAL_EVERY_UPDATES_OVERRIDE:-1}
EVAL_EPISODES=${EVAL_EPISODES_OVERRIDE:-16}
ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE:-256}
EVAL_MICRO_BATCH_SIZE=${EVAL_MICRO_BATCH_SIZE_OVERRIDE:-256}
UPDATE_MICRO_BATCH_SIZE=${UPDATE_MICRO_BATCH_SIZE_OVERRIDE:-32}
RUN_SETUP_SMOKE=${RUN_SETUP_SMOKE_OVERRIDE:-false}
TRAIN_VIDEO_NUM_ENVS=${TRAIN_VIDEO_NUM_ENVS_OVERRIDE:-4}
TEST_VIDEO_NUM_ENVS=${TEST_VIDEO_NUM_ENVS_OVERRIDE:-4}
TEST_VIDEO_EPISODES=${TEST_VIDEO_EPISODES_OVERRIDE:-4}
EARLY_STOP_ZERO_SUCCESS_MINUTES=${EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE:-45000}

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
    python -u "$OURS_DIR/online_rl_cl.py"
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
    --total-timesteps "$TOTAL_TIMESTEPS"
    --num-envs "$NUM_ENVS"
    --num-eval-envs "$NUM_EVAL_ENVS"
    --num-steps "$NUM_STEPS"
    --num-minibatches "$NUM_MINIBATCHES"
    --update-epochs "$UPDATE_EPOCHS"
    --learning-rate 6e-5
    --head-learning-rate 6e-5
    --state-learning-rate 6e-5
    --value-head-learning-rate 6e-5
    --backbone-learning-rate 6e-5
    --weight-decay 1e-6
    --gamma 0.99
    --gae-lambda 0.95
    --clip-coef 0.2
    --ent-coef 1e-3
    --vf-coef 0.5
    --max-grad-norm 0.5
    --target-kl 0.02
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
    --save-train-video-freq 10
    --train-video-num-envs "$TRAIN_VIDEO_NUM_ENVS"
    --test-video-num-envs "$TEST_VIDEO_NUM_ENVS"
    --test-video-episodes "$TEST_VIDEO_EPISODES"
    --action-dim 12
    --env-action-dim 25
    --state-dim 73
    --large-agent-checkpoint "$LARGE_AGENT_CHECKPOINT"
    --small-model-generation-strategy target-single-traj
    --small-model-generation-policy small
    --small-model-feedback-schedule before_per_rollout
    --small-model-regeneration-schedule before_per_rollout
    --small-model-feedback-alpha 1.0
    --small-model-regeneration-increment-ratio 1.0
    --reset-optimizer-after-regeneration true
    --max-sparsity 0.8
    --early-stop-zero-success-minutes "$EARLY_STOP_ZERO_SUCCESS_MINUTES"
    --cuda-device "$CUDA_DEVICES"
)

if [ -n "$RUN_NAME" ]; then
    PYTHON_CMD+=(--run-name "$RUN_NAME")
fi

if [ -n "$CONTINUE_TRAIN_FROM" ]; then
    PYTHON_CMD+=(--continue-train-from "$CONTINUE_TRAIN_FROM")
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
    echo "ENV_ID=$ENV_ID"
    echo "ENVS_ID=$ENVS_ID"
    echo "ENV_CHANGE_TIME_POINTS=$ENV_CHANGE_TIME_POINTS"
    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
