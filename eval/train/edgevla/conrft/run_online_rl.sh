#!/usr/bin/env bash
set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="$SCRIPT_DIR/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME-conrft-unitree-g1-lift-apple.log"
OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE_OVERRIDE:-"train/edgevla/conrft/outputs"}
ENV_ID=${ENV_ID_OVERRIDE:-UnitreeG1LiftApple-v1}
ENVS_ID=${ENVS_ID_OVERRIDE:-"['UnitreeG1LiftCubeObjectScaleDown1p3-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeObjectPurple-v1','UnitreeG1LiftSphereLightStronger50-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectScaleDown1p1-v1','UnitreeG1LiftSphereObjectScaleDown1p3-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectPurple-v1']"}
ENV_CHANGE_TIME_POINTS=${ENV_CHANGE_TIME_POINTS_OVERRIDE:-"[31,62,96,131,151,163,207,247,271,300]"}
CUDA_DEVICES=${CUDA_DEVICES:-1,2,3,4}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS_OVERRIDE:-400}
EARLY_STOP_ZERO_SUCCESS_MINUTES=${EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE:-45000}
RUN_NAME=${RUN_NAME_OVERRIDE:-}
SAVE_VIDEO=${SAVE_VIDEO_OVERRIDE:-false}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS_OVERRIDE:-100000000}
NUM_ENVS=${NUM_ENVS_OVERRIDE:-128}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS_OVERRIDE:-8}
NUM_STEPS=${NUM_STEPS_OVERRIDE:-64}
NUM_MINIBATCHES=${NUM_MINIBATCHES_OVERRIDE:-16}
UPDATE_EPOCHS=${UPDATE_EPOCHS_OVERRIDE:-2}
EVAL_EPISODES=${EVAL_EPISODES_OVERRIDE:-16}
ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE:-256}
EVAL_MICRO_BATCH_SIZE=${EVAL_MICRO_BATCH_SIZE_OVERRIDE:-256}
UPDATE_MICRO_BATCH_SIZE=${UPDATE_MICRO_BATCH_SIZE_OVERRIDE:-32}
ROLLOUT_PROGRESS_LOG_INTERVAL=${ROLLOUT_PROGRESS_LOG_INTERVAL_OVERRIDE:-10}
SUPERVISED_UPDATES_PER_ITER=${SUPERVISED_UPDATES_PER_ITER_OVERRIDE:-8}
SUPERVISED_BATCH_SIZE=${SUPERVISED_BATCH_SIZE_OVERRIDE:-256}
SUPERVISED_ONLINE_RATIO=${SUPERVISED_ONLINE_RATIO_OVERRIDE:-0.5}
WARMUP_SUCCESS_STEPS_BEFORE_SUPERVISED=${WARMUP_SUCCESS_STEPS_BEFORE_SUPERVISED_OVERRIDE:-100}
MIN_SUCCESS_STEPS_FOR_SUPERVISED=${MIN_SUCCESS_STEPS_FOR_SUPERVISED_OVERRIDE:-10}
ONLINE_BUFFER_CAPACITY=${ONLINE_BUFFER_CAPACITY_OVERRIDE:-20000}
EXPERT_BUFFER_CAPACITY=${EXPERT_BUFFER_CAPACITY_OVERRIDE:-40000}
EXPERT_TARGET_SUCCESS_TRAJECTORIES=${EXPERT_TARGET_SUCCESS_TRAJECTORIES_OVERRIDE:-24}
EXPERT_COLLECT_NUM_ENVS=${EXPERT_COLLECT_NUM_ENVS_OVERRIDE:-8}
EXPERT_COLLECT_MAX_STEPS=${EXPERT_COLLECT_MAX_STEPS_OVERRIDE:-5000}
EXPERT_COLLECT_SEED=${EXPERT_COLLECT_SEED_OVERRIDE:-0}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT_OVERRIDE:-"ckpt/edgevla/env_verify/outputs/ppo_unitree_g1_lift_apple/20260511-063605/best_policy.pt"}
STATIC_MODEL_CHECKPOINT=${STATIC_MODEL_CHECKPOINT_OVERRIDE:-"ckpt/edgevla/ours/outputs/bc_unitree_g1_lift_apple_fbs/20260511-171959/best_policy.pt"}

mkdir -p "$LOG_DIR" "$OUTPUT_DIR_BASE"

PYTHON_CMD=(
    python -u "$SCRIPT_DIR/online_rl.py"
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
    --teacher-checkpoint "$TEACHER_CHECKPOINT"
    --static-model-checkpoint "$STATIC_MODEL_CHECKPOINT"
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
    --supervised-learning-rate 6e-5
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
    --eval-every-updates 1
    --max-runtime-hours "$MAX_RUNTIME_HOURS"
    --rollout-micro-batch-size "$ROLLOUT_MICRO_BATCH_SIZE"
    --eval-micro-batch-size "$EVAL_MICRO_BATCH_SIZE"
    --update-micro-batch-size "$UPDATE_MICRO_BATCH_SIZE"
    --rollout-progress-log-interval "$ROLLOUT_PROGRESS_LOG_INTERVAL"
    --freeze-vla-backbone false
    --backbone-warmup-updates 0
    --run-setup-smoke false
    --save-video "$SAVE_VIDEO"
    --save-train-video-freq 10
    --train-video-num-envs 4
    --test-video-num-envs 4
    --test-video-episodes 4
    --action-dim 12
    --env-action-dim 25
    --state-dim 73
    --early-stop-zero-success-minutes "$EARLY_STOP_ZERO_SUCCESS_MINUTES"
    --cuda-device "$CUDA_DEVICES"
    --supervised-updates-per-iter "$SUPERVISED_UPDATES_PER_ITER"
    --supervised-batch-size "$SUPERVISED_BATCH_SIZE"
    --supervised-online-ratio "$SUPERVISED_ONLINE_RATIO"
    --warmup-success-steps-before-supervised "$WARMUP_SUCCESS_STEPS_BEFORE_SUPERVISED"
    --min-success-steps-for-supervised "$MIN_SUCCESS_STEPS_FOR_SUPERVISED"
    --online-buffer-capacity "$ONLINE_BUFFER_CAPACITY"
    --expert-buffer-capacity "$EXPERT_BUFFER_CAPACITY"
    --expert-target-success-trajectories "$EXPERT_TARGET_SUCCESS_TRAJECTORIES"
    --expert-collect-num-envs "$EXPERT_COLLECT_NUM_ENVS"
    --expert-collect-max-steps "$EXPERT_COLLECT_MAX_STEPS"
    --expert-collect-seed "$EXPERT_COLLECT_SEED"
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
    echo "ENVS_ID=$ENVS_ID"
    echo "ENV_CHANGE_TIME_POINTS=$ENV_CHANGE_TIME_POINTS"
    echo "RUN_NAME=$RUN_NAME"
    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
