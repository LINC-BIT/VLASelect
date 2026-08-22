#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"

OUTPUT_DIR=${OUTPUT_DIR_OVERRIDE:-"$SCRIPT_DIR/outputs/layer_grained"}
ENV_ID=${ENV_ID_OVERRIDE:-HoldCubeInHandObjectScaleDown1p2-v1}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
export CUDA_VISIBLE_DEVICES

ARGS=(--env-id "$ENV_ID" --output-dir "$OUTPUT_DIR" --num-envs 256 --num-eval-envs 8 --num-steps 50 \
  --model-dir eval/ckpt/vla_adapter_new/LIBERO-Object \
  --large-agent-checkpoint eval/ckpt/vla_adapter_new/ours/outputs/20260502-112804/best_policy.pt \
  --small-model-scaling-strategy target-single-traj --small-model-scaling-policy small \
  --small-model-regeneration-schedule before_per_rollout --max-sparsity 0.8 --cuda-device "$CUDA_VISIBLE_DEVICES")
exec python -u "$SCRIPT_DIR/layer_grained.py" "${ARGS[@]}"
