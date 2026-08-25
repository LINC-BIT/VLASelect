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
ENV_ID=${ENV_ID_OVERRIDE:-PickCubeObjectScaleUp1p2-v1}
ENVS_ID=${ENVS_ID_OVERRIDE:-"['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']"}
ENV_CHANGE_TIME_POINTS=${ENV_CHANGE_TIME_POINTS_OVERRIDE:-"[31,62,96,131,151,163,207,247,271,300]"}
ENV_CONFIG_PATH=${ENV_CONFIG_PATH_OVERRIDE:-datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json}
STATE_NORM_STATS_PATH=${STATE_NORM_STATS_PATH_OVERRIDE:-ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth}
CHECKPOINT_PATH=${CHECKPOINT_PATH_OVERRIDE:-ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS_OVERRIDE:-100000000}
MAX_TIME=${MAX_TIME_OVERRIDE:-301}
NUM_ENVS=${NUM_ENVS:-64}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS_OVERRIDE:-32}
NUM_STEPS=${NUM_STEPS_OVERRIDE:-50}
NUM_EVAL_STEPS=${NUM_EVAL_STEPS_OVERRIDE:-50}
NUM_MINIBATCHES=${NUM_MINIBATCHES_OVERRIDE:-16}
EVAL_FREQ=${EVAL_FREQ_OVERRIDE:-1}

mkdir -p "$LOG_DIR"

PYTHON_CMD=(
    python -u -m train.octo.self_improv.online_rl
    --exp-name "$RUN_NAME"
    --env-id "$ENV_ID"
    --envs-id "$ENVS_ID"
    --env-change-time-points "$ENV_CHANGE_TIME_POINTS"
    --env_config_path "$ENV_CONFIG_PATH"
    --state-norm-stats-path "$STATE_NORM_STATS_PATH"
    --checkpoint "$CHECKPOINT_PATH"
    --total_timesteps "$TOTAL_TIMESTEPS"
    --max_time "$MAX_TIME"
    --learning_rate 3e-5
    --eval_freq "$EVAL_FREQ"
    --max-sparsity 0.8
    --num_envs "$NUM_ENVS"
    --num_eval_envs "$NUM_EVAL_ENVS"
    --num_steps "$NUM_STEPS"
    --num_eval_steps "$NUM_EVAL_STEPS"
    --num_minibatches "$NUM_MINIBATCHES"
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
