#!/usr/bin/env bash

set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
STAMP=$(date +"%Y%m%d-%H%M%S")

LOG_DIR="train/octo/self_improv/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME.log"
PLOT_OUTPUT="eurosys27-baselines-paper/self_improv/codex_run_logs/${STAMP}-success_curve.png"

CUDA_DEVICES="${CUDA_DEVICES:-5}"
EXP_NAME="${EXP_NAME:-PickCubeObjectScaleUp1p2-v1/baselines/self_improv/online_rl/${STAMP}-cl-critic-progress}"
TAIL_LOG="${TAIL_LOG:-1}"
ENABLE_SELF_CURVE_WATCHER="${ENABLE_SELF_CURVE_WATCHER:-1}"
LAUNCH_DIRECT="${LAUNCH_DIRECT:-0}"
RUN_NAME="$EXP_NAME"
RUN_DIR="ckpt/$RUN_NAME"
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
ENV_CONFIG_PATH=${ENV_CONFIG_PATH_OVERRIDE:-datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json}

mkdir -p "$LOG_DIR"

PYTHON_CMD=(
    python -u -m train.octo.self_improv.online_rl
    --exp-name "$RUN_NAME"
    --env-id PickCubeObjectScaleUp1p2-v1
    --envs-id "['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']"
    --env-change-time-points "[31,62,96,131,151,163,207,247,271,300]"
    --env_config_path "$ENV_CONFIG_PATH"
    --state-norm-stats-path ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth
    --checkpoint ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt
    --total_timesteps 100000000
    --max_time 301
    --learning_rate 3e-5
    --eval_freq 1
    --max-sparsity 0.8
    --num_envs 256
    --num_eval_envs 32
    --num_minibatches 16
    --gamma 0.9
    --reinforce_coef 5e-2
    --success_calibration_episodes 64
    --tag cl-critic-progress
    --no-capture-video
)

if [ "$LAUNCH_DIRECT" = "1" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    exec "${PYTHON_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" nohup "${PYTHON_CMD[@]}" > "$LOG_FILE" 2>&1 &

    TRAIN_PID=$!
    echo "TRAIN_PID=$TRAIN_PID"
    echo "RUN_DIR=$RUN_DIR"
    echo "LOG_FILE=$LOG_FILE"
    echo "training pid: $TRAIN_PID"
    echo "run name: $RUN_NAME"
    echo "plot output: $PLOT_OUTPUT"

    if [ "$ENABLE_SELF_CURVE_WATCHER" = "1" ]; then
        (
            while kill -0 "$TRAIN_PID" 2>/dev/null; do
                python -u -m train.octo.self_improv.draw_online_rl_acc \
                    --run-dir "$RUN_DIR" \
                    --output "$PLOT_OUTPUT" \
                    --title "Self-Improvement Success Curve" || true
                sleep 300
            done
            python -u -m train.octo.self_improv.draw_online_rl_acc \
                --run-dir "$RUN_DIR" \
                --output "$PLOT_OUTPUT" \
                --title "Self-Improvement Success Curve" || true
        ) >/dev/null 2>&1 &
    fi

    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
