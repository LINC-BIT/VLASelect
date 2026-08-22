#!/usr/bin/env bash
set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/octo/world_env/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME-online_rl.log"
mkdir -p "$LOG_DIR"

CUDA_DEVICES=${CUDA_DEVICES:-2}
WORLD_MODEL_CKPT=${WORLD_MODEL_CKPT:-}
EXP_NAME=${EXP_NAME:-}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
ENV_CONFIG_PATH=${ENV_CONFIG_PATH_OVERRIDE:-datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json}
STATE_NORM_STATS_PATH=${STATE_NORM_STATS_PATH_OVERRIDE:-ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth}
CHECKPOINT_PATH=${CHECKPOINT_PATH_OVERRIDE:-ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS_OVERRIDE:-100000000}
NUM_ENVS=${NUM_ENVS_OVERRIDE:-256}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS_OVERRIDE:-32}
NUM_MINIBATCHES=${NUM_MINIBATCHES_OVERRIDE:-16}
UPDATE_EPOCHS=${UPDATE_EPOCHS_OVERRIDE:-2}
MAX_TIME=${MAX_TIME_OVERRIDE:-301}
ALLOW_RANDOM_INPUT_FALLBACK=${ALLOW_RANDOM_INPUT_FALLBACK:-0}
FORCE_RANDOM_MODEL_INIT=${FORCE_RANDOM_MODEL_INIT:-0}
RANDOM_INPUT_ROOT=${RANDOM_INPUT_ROOT_OVERRIDE:-train/octo/world_env/smoke_inputs}

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
    if [[ -z "$WORLD_MODEL_CKPT" || ! -f "$WORLD_MODEL_CKPT" ]]; then
        WORLD_MODEL_CKPT="$RANDOM_INPUT_ROOT/world_model.pt"
        PREPARE_ARGS+=(--world-model-out "$WORLD_MODEL_CKPT")
    fi
    if [[ ${#PREPARE_ARGS[@]} -gt 0 ]]; then
        python -u train/octo/prepare_smoke_assets.py "${PREPARE_ARGS[@]}"
    fi
fi

if [ -z "$WORLD_MODEL_CKPT" ]; then
    echo "Please set WORLD_MODEL_CKPT to the pretrained world model checkpoint path."
    exit 1
fi

if [[ "$FORCE_RANDOM_MODEL_INIT" == "1" ]]; then
    CHECKPOINT_PATH="$RANDOM_INPUT_ROOT/missing_large_model.pt"
fi

EXTRA_ARGS=()
RUN_DIR=""
if [ -n "$EXP_NAME" ]; then
    EXTRA_ARGS+=(--exp-name "$EXP_NAME")
    RUN_DIR="ckpt/$EXP_NAME"
fi

PYTHON_CMD=(
    python -u -m train.octo.world_env.online_rl
    --env-id PickCubeObjectScaleUp1p2-v1
    --envs-id "['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']"
    --env-change-time-points "[31,62,96,131,151,163,207,247,271,300]"
    --env_config_path "$ENV_CONFIG_PATH"
    --state-norm-stats-path "$STATE_NORM_STATS_PATH"
    --checkpoint "$CHECKPOINT_PATH"
    --world-model-checkpoint "$WORLD_MODEL_CKPT"
    --total_timesteps "$TOTAL_TIMESTEPS"
    --learning_rate 2e-5
    --eval_freq 1
    --max-sparsity 0.8
    --num_envs "$NUM_ENVS"
    --num_eval_envs "$NUM_EVAL_ENVS"
    --num_minibatches "$NUM_MINIBATCHES"
    --update_epochs "$UPDATE_EPOCHS"
    --real-reward-weight 0.3
    --verified-reward-weight 0.7
    --success-termination-threshold 0.6
    --success-bonus 0.2
    --tag world_env_baseline_cl
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
