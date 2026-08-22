#!/usr/bin/env bash

set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/octo/ours_single_agent/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/${TIME}_motivation_small_model.log"
mkdir -p "$LOG_DIR"

ENV_SEQUENCE="['PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PushCubeObjectPurple-v1','PushCubeObjectBlack-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PickCubeColorTempHigher50-v1','PickCubeColorTempLower50-v1','PickCubeObjectPurple-v1','PickCubeObjectBlack-v1']"
CHANGE_POINTS="[30,60,90,120,150,180,210,240,270,300]"

CUDA_VISIBLE_DEVICES=3 nohup python -u -m train.octo.ours_single_agent.motivation.small_model \
    --env-id PokeCubeLightWeaker50-v1 \
    --envs-id "$ENV_SEQUENCE" \
    --env-change-time-points "$CHANGE_POINTS" \
    --env_config_path datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json \
    --state-norm-stats-path ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth \
    --checkpoint ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt \
    --total_timesteps 100000000 \
    --learning_rate 3e-5 \
    --eval_freq 1 \
    --max-sparsity 0.9 \
    --no-track \
    --num_envs 256 \
    --num_eval_envs 32 \
    --num_minibatches 16 \
    --small_model_generation_strategy source \
    --tag motivation-small-model \
    > "$LOG_FILE" 2>&1 &

echo "Started static small model training on GPU 3"
echo "Log file: $LOG_FILE"
tail -f "$LOG_FILE"
