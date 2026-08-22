#!/usr/bin/env bash

set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
STAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="train/octo/flare/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/${TIME}-flare_1h.log"
CURVE_OUTPUT="eurosys27-baselines-paper/flare/codex_run_logs/${STAMP}-success_curve.png"
mkdir -p "$LOG_DIR"
mkdir -p "eurosys27-baselines-paper/flare/codex_run_logs"

CUDA_DEVICES=${CUDA_DEVICES:-1}
EXP_NAME=${EXP_NAME:-}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
ENV_CONFIG_PATH=${ENV_CONFIG_PATH_OVERRIDE:-datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json}

EXTRA_ARGS=()
RUN_DIR=""
if [ -n "$EXP_NAME" ]; then
    EXTRA_ARGS+=(--exp-name "$EXP_NAME")
    RUN_DIR="ckpt/$EXP_NAME"
fi

PYTHON_CMD=(
    python -u -m train.octo.flare.online_rl
    --env-id PickCubeObjectScaleUp1p2-v1
    --envs-id "['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']"
    --env-change-time-points "[31,62,96,131,151,163,207,247,271,300]"
    --env_config_path "$ENV_CONFIG_PATH"
    --state-norm-stats-path ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth
    --checkpoint ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt
    --total_timesteps 100000000
    --learning_rate 2e-5
    --eval_freq 1
    --max-sparsity 0.8
    --num_envs 256
    --num_eval_envs 32
    --num_minibatches 16
    --update_epochs 1
    --ent_coef 0.0
    --small_model_generation_strategy source
    --curve_output "$CURVE_OUTPUT"
    --curve_title "FLaRe CL Success Curve"
    --tag flare-cl
    --max_time 301
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
    echo "log file: $LOG_FILE"
    echo "curve output: $CURVE_OUTPUT"
    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
