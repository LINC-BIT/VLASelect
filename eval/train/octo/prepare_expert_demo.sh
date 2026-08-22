#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH=${OUTPUT_PATH:-train/octo/smoke_inputs/expert_demo.h5}
ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json}
STATE_NORM_STATS_PATH=${STATE_NORM_STATS_PATH:-ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt}
TARGET_SUCCESS_TRAJECTORIES=${TARGET_SUCCESS_TRAJECTORIES:-2}
NUM_ENVS=${NUM_ENVS:-2}
MAX_STEPS=${MAX_STEPS:-200}
SEED=${SEED:-0}
LOG_PREFIX=${LOG_PREFIX:-octo-smoke-expert-demo}
ALLOW_RANDOM_FALLBACK=${ALLOW_RANDOM_FALLBACK:-1}

JSON_PATH="${OUTPUT_PATH%.h5}.json"
mkdir -p "$(dirname "$OUTPUT_PATH")"

if [[ -f "$OUTPUT_PATH" && -f "$JSON_PATH" ]]; then
    echo "[$LOG_PREFIX] reuse existing expert demo: $OUTPUT_PATH"
    exit 0
fi

if [[ ! -f "$CHECKPOINT_PATH" || ! -f "$ENV_CONFIG_PATH" || ! -f "$STATE_NORM_STATS_PATH" ]]; then
    if [[ "$ALLOW_RANDOM_FALLBACK" != "1" ]]; then
        echo "[$LOG_PREFIX] missing dependency for teacher rollout and random fallback is disabled" >&2
        exit 1
    fi
    echo "[$LOG_PREFIX] teacher rollout prerequisites missing; create smoke expert demo instead"
    python -u train/octo/prepare_smoke_assets.py --expert-demo-out "$OUTPUT_PATH"
    exit 0
fi

echo "[$LOG_PREFIX] generate expert demo with teacher checkpoint"
python -u train/octo/generate_teacher_demo.py     --output-path "$OUTPUT_PATH"     --env-config-path "$ENV_CONFIG_PATH"     --state-norm-stats-path "$STATE_NORM_STATS_PATH"     --checkpoint "$CHECKPOINT_PATH"     --target-success-trajectories "$TARGET_SUCCESS_TRAJECTORIES"     --num-envs "$NUM_ENVS"     --max-steps "$MAX_STEPS"     --seed "$SEED"     --reuse-if-exists     --log-prefix "$LOG_PREFIX"
