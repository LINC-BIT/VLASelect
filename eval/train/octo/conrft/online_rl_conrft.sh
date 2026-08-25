#!/usr/bin/env bash
set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/octo/conrft/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME.log"
mkdir -p "$LOG_DIR"

CUDA_DEVICES=${CUDA_DEVICES:-0}
EXP_NAME=${EXP_NAME:-}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
SMOKE=${SMOKE:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
ENV_ID=${ENV_ID_OVERRIDE:-PickCubeLightStronger50-v1}
ENVS_ID=${ENVS_ID_OVERRIDE:-"['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']"}
ENV_CHANGE_TIME_POINTS=${ENV_CHANGE_TIME_POINTS_OVERRIDE:-"[31,62,96,131,151,163,207,247,271,300]"}
ENV_CONFIG_PATH=${ENV_CONFIG_PATH_OVERRIDE:-datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json}
EXPERT_DEMO_PATH=${EXPERT_DEMO_PATH_OVERRIDE:-datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.h5}
STATE_NORM_STATS_PATH=${STATE_NORM_STATS_PATH_OVERRIDE:-ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth}
CHECKPOINT_PATH=${CHECKPOINT_PATH_OVERRIDE:-ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS_OVERRIDE:-100000000}
NUM_ENVS=${NUM_ENVS_OVERRIDE:-256}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS_OVERRIDE:-32}
NUM_MINIBATCHES=${NUM_MINIBATCHES_OVERRIDE:-16}
UPDATE_EPOCHS=${UPDATE_EPOCHS_OVERRIDE:-2}
SUPERVISED_UPDATES_PER_ITER=${SUPERVISED_UPDATES_PER_ITER_OVERRIDE:-16}
SUPERVISED_BATCH_SIZE=${SUPERVISED_BATCH_SIZE_OVERRIDE:-256}
SUPERVISED_ONLINE_RATIO=${SUPERVISED_ONLINE_RATIO_OVERRIDE:-0.5}
WARMUP_SUCCESS_STEPS_BEFORE_SUPERVISED=${WARMUP_SUCCESS_STEPS_BEFORE_SUPERVISED_OVERRIDE:-100}
MAX_TIME=${MAX_TIME_OVERRIDE:-301}
ALLOW_RANDOM_INPUT_FALLBACK=${ALLOW_RANDOM_INPUT_FALLBACK:-0}
FORCE_RANDOM_MODEL_INIT=${FORCE_RANDOM_MODEL_INIT:-0}
RANDOM_INPUT_ROOT=${RANDOM_INPUT_ROOT_OVERRIDE:-train/octo/conrft/smoke_inputs}
AUTO_GENERATE_EXPERT_DEMO=${AUTO_GENERATE_EXPERT_DEMO:-1}
EXPERT_DEMO_OUTPUT_PATH=${EXPERT_DEMO_OUTPUT_PATH_OVERRIDE:-$EXPERT_DEMO_PATH}
if [[ "$SMOKE" == "1" && -z "${EXPERT_DEMO_OUTPUT_PATH_OVERRIDE:-}" ]]; then
    EXPERT_DEMO_OUTPUT_PATH="$RANDOM_INPUT_ROOT/expert_demo.h5"
fi
EXPERT_DEMO_TARGET_SUCCESS_TRAJECTORIES=${EXPERT_DEMO_TARGET_SUCCESS_TRAJECTORIES:-24}
EXPERT_DEMO_NUM_ENVS=${EXPERT_DEMO_NUM_ENVS:-8}
EXPERT_DEMO_MAX_STEPS=${EXPERT_DEMO_MAX_STEPS:-5000}
EXPERT_DEMO_REUSE_CACHE=${EXPERT_DEMO_REUSE_CACHE:-1}
EXPERT_DEMO_LOG_PREFIX=${EXPERT_DEMO_LOG_PREFIX:-octo-conrft-teacher-demo}

if [[ "$ALLOW_RANDOM_INPUT_FALLBACK" == "1" ]]; then
    mkdir -p "$RANDOM_INPUT_ROOT"
    PREPARE_ARGS=()
    if [[ ! -f "$ENV_CONFIG_PATH" ]]; then
        ENV_CONFIG_PATH="$RANDOM_INPUT_ROOT/env_config.json"
        PREPARE_ARGS+=(--env-config-out "$ENV_CONFIG_PATH")
    fi
    if [[ ! -f "$STATE_NORM_STATS_PATH" ]]; then
        STATE_NORM_STATS_PATH="$RANDOM_INPUT_ROOT/state_norm_stats.pth"
        PREPARE_ARGS+=(--state-norm-out "$STATE_NORM_STATS_PATH")
    fi
    if [[ ! -f "$EXPERT_DEMO_PATH" ]]; then
        EXPERT_DEMO_PATH="$RANDOM_INPUT_ROOT/expert_demo.h5"
        PREPARE_ARGS+=(--expert-demo-out "$EXPERT_DEMO_PATH")
    fi
    if [[ ${#PREPARE_ARGS[@]} -gt 0 ]]; then
        python -u train/octo/prepare_smoke_assets.py "${PREPARE_ARGS[@]}"
    fi
fi

if [[ "$FORCE_RANDOM_MODEL_INIT" == "1" ]]; then
    CHECKPOINT_PATH="$RANDOM_INPUT_ROOT/missing_large_model.pt"
fi

if [[ "$AUTO_GENERATE_EXPERT_DEMO" == "1" && ! -f "$EXPERT_DEMO_PATH" ]]; then
    if [[ "$SMOKE" == "1" ]]; then
        EXPERT_DEMO_TARGET_SUCCESS_TRAJECTORIES=${EXPERT_DEMO_TARGET_SUCCESS_TRAJECTORIES_MWE:-2}
        EXPERT_DEMO_NUM_ENVS=${EXPERT_DEMO_NUM_ENVS_MWE:-2}
        EXPERT_DEMO_MAX_STEPS=${EXPERT_DEMO_MAX_STEPS_MWE:-200}
    fi
    mkdir -p "$(dirname "$EXPERT_DEMO_OUTPUT_PATH")"
    python -u train/octo/generate_teacher_demo.py         --output-path "$EXPERT_DEMO_OUTPUT_PATH"         --env-config-path "$ENV_CONFIG_PATH"         --state-norm-stats-path "$STATE_NORM_STATS_PATH"         --checkpoint "$CHECKPOINT_PATH"         --target-success-trajectories "$EXPERT_DEMO_TARGET_SUCCESS_TRAJECTORIES"         --num-envs "$EXPERT_DEMO_NUM_ENVS"         --max-steps "$EXPERT_DEMO_MAX_STEPS"         --seed 0         --reuse-if-exists         --log-prefix "$EXPERT_DEMO_LOG_PREFIX"
    EXPERT_DEMO_PATH="$EXPERT_DEMO_OUTPUT_PATH"
fi

EXTRA_ARGS=()
RUN_DIR=""
if [ -n "$EXP_NAME" ]; then
    EXTRA_ARGS+=(--exp-name "$EXP_NAME")
    RUN_DIR="ckpt/$EXP_NAME"
fi

PYTHON_CMD=(
    python -u -m train.octo.conrft.online_rl
    --env-id "$ENV_ID"
    --envs-id "$ENVS_ID"
    --env-change-time-points "$ENV_CHANGE_TIME_POINTS"
    --env_config_path "$ENV_CONFIG_PATH"
    --expert_demo_path "$EXPERT_DEMO_PATH"
    --state-norm-stats-path "$STATE_NORM_STATS_PATH"
    --checkpoint "$CHECKPOINT_PATH"
    --total_timesteps "$TOTAL_TIMESTEPS"
    --learning_rate 3e-5
    --supervised_learning_rate 3e-5
    --eval_freq 1
    --max-sparsity 0.8
    --num_envs "$NUM_ENVS"
    --num_eval_envs "$NUM_EVAL_ENVS"
    --num_minibatches "$NUM_MINIBATCHES"
    --update_epochs "$UPDATE_EPOCHS"
    --supervised_updates_per_iter "$SUPERVISED_UPDATES_PER_ITER"
    --supervised_batch_size "$SUPERVISED_BATCH_SIZE"
    --supervised_online_ratio "$SUPERVISED_ONLINE_RATIO"
    --warmup_success_steps_before_supervised "$WARMUP_SUCCESS_STEPS_BEFORE_SUPERVISED"
    --tag conrft_baseline_cl
    --max_time "$MAX_TIME"
    "${EXTRA_ARGS[@]}"
)

if [ "$LAUNCH_DIRECT" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
    exec "${PYTHON_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup "${PYTHON_CMD[@]}" > "$LOG_FILE" 2>&1 &

    TRAIN_PID=$!
    echo "TRAIN_PID=$TRAIN_PID"
    echo "RUN_DIR=$RUN_DIR"
    echo "LOG_FILE=$LOG_FILE"

    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
