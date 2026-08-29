#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
EVAL_ROOT="$REPO_ROOT/eval"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0}"
cd "$SCRIPT_DIR"

CNN_CHECKPOINT=${CNN_SOURCE_CHECKPOINT:-$EVAL_ROOT/ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt}
MLP_CHECKPOINT=${MLP_SOURCE_CHECKPOINT:-$SCRIPT_DIR/ckpt/mlp_pretrain/best.pt}
CNN_OUTPUT=${PPO_GEN_CNN_MODEL:-$SCRIPT_DIR/ppo-gen-cnn-model}
MLP_OUTPUT=${PPO_GEN_MLP_MODEL:-$SCRIPT_DIR/ppo-gen-mlp-model}
ENV_CONFIG=${ENV_CONFIG_PATH_OVERRIDE:-$EVAL_ROOT/datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json}
STATE_STATS=${STATE_NORM_STATS_PATH_OVERRIDE:-$EVAL_ROOT/ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth}

[[ -f "$CNN_CHECKPOINT" ]] || { echo "missing CNN source checkpoint: $CNN_CHECKPOINT" >&2; exit 1; }
[[ -f "$MLP_CHECKPOINT" ]] || { echo "missing MLP source checkpoint: $MLP_CHECKPOINT" >&2; exit 1; }

COMMON=(
  --env-id PickCube-v1
  --env_config_path "$ENV_CONFIG"
  --state-norm-stats-path "$STATE_STATS"
  --num_envs 1 --num_eval_envs 1 --num_steps 1 --num_eval_steps 50
  --total_timesteps 1 --eval_freq 1 --no-track --no-capture-video
  --generate-only
)

MODEL_TYPE=cnn python -u -m train.octo.ours_single_agent.online_rl_cl \
  "${COMMON[@]}" --checkpoint "$CNN_CHECKPOINT" \
  --max-sparsity "${CNN_GENERATION_SPARSITY:-0.8}" \
  --generated-model-output "$CNN_OUTPUT"

MODEL_TYPE=mlp python -u -m train.octo.ours_single_agent.online_rl_cl \
  "${COMMON[@]}" --checkpoint "$MLP_CHECKPOINT" \
  --max-sparsity "${MLP_GENERATION_SPARSITY:-0.1}" \
  --generated-model-output "$MLP_OUTPUT"

echo "Generated models: $CNN_OUTPUT $MLP_OUTPUT"
