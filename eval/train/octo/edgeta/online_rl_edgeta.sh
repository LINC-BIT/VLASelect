#!/usr/bin/env bash

export WANDB_API_KEY=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

if [ "${WANDB_AUTO_LOGIN:-0}" = "1" ]; then
    wandb login --relogin "$WANDB_API_KEY" || true
fi

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/octo/edgeta/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME.log"
mkdir -p "$LOG_DIR"

CUDA_DEVICES=${CUDA_DEVICES:-1}
EXP_NAME=${EXP_NAME:-}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
ENV_ID=${ENV_ID_OVERRIDE:-PickCubeObjectScaleUp1p2-v1}
ENVS_ID=${ENVS_ID_OVERRIDE:-"['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']"}
ENV_CHANGE_TIME_POINTS=${ENV_CHANGE_TIME_POINTS_OVERRIDE:-"[31,62,96,131,151,163,207,247,271,300]"}
ENV_CONFIG_PATH=${ENV_CONFIG_PATH_OVERRIDE:-datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json}
STATE_NORM_STATS_PATH=${STATE_NORM_STATS_PATH_OVERRIDE:-ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth}
CHECKPOINT_PATH=${CHECKPOINT_PATH_OVERRIDE:-ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS_OVERRIDE:-100000000}
NUM_ENVS=${NUM_ENVS_OVERRIDE:-256}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS_OVERRIDE:-32}
NUM_STEPS=${NUM_STEPS_OVERRIDE:-50}
NUM_EVAL_STEPS=${NUM_EVAL_STEPS_OVERRIDE:-50}
NUM_MINIBATCHES=${NUM_MINIBATCHES_OVERRIDE:-16}
UPDATE_EPOCHS=${UPDATE_EPOCHS_OVERRIDE:-2}
EVAL_FREQ=${EVAL_FREQ_OVERRIDE:-1}
MAX_TIME=${MAX_TIME_OVERRIDE:-301}

EXTRA_ARGS=()
RUN_DIR=""
if [ -n "$EXP_NAME" ]; then
    EXTRA_ARGS+=(--exp-name "$EXP_NAME")
    RUN_DIR="ckpt/$EXP_NAME/[agent]"
fi

PYTHON_CMD=(
    python -u -m train.octo.ours_single_agent.online_rl_cl
    --env-id "$ENV_ID"
    --envs-id "$ENVS_ID"
    --env-change-time-points "$ENV_CHANGE_TIME_POINTS"
    --env_config_path "$ENV_CONFIG_PATH"
    --state-norm-stats-path "$STATE_NORM_STATS_PATH"
    --checkpoint "$CHECKPOINT_PATH"
    --total_timesteps "$TOTAL_TIMESTEPS"
    --learning_rate 3e-5
    --eval_freq "$EVAL_FREQ"
    --track
    --max-sparsity 0.8
    --num_envs "$NUM_ENVS"
    --num_eval_envs "$NUM_EVAL_ENVS"
    --num_steps "$NUM_STEPS"
    --num_eval_steps "$NUM_EVAL_STEPS"
    --num_minibatches "$NUM_MINIBATCHES"
    --update_epochs "$UPDATE_EPOCHS"
    --small_model_generation_strategy target-single-traj
    --small_model_feedback_schedule before_per_rollout
    --small_model_regeneration_schedule before_per_rollout
    --small_model_feedback_alpha 0.6
    --small_model_regeneration_increment_ratio 1.0
    --reset_optimizer_after_regeneration
    --small_model_generation_policy small
    --tag edgeta-targetsingletraj-feedback_per_rollout-regen_per_rollout-feedback0.6-arch_update1.0-reset_optimizer
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
