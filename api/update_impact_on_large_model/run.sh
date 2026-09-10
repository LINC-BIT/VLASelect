#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"

MWE=${MWE:-0}
RUN_NAME=${RUN_NAME_OVERRIDE:-$(date +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR_OVERRIDE:-"$SCRIPT_DIR/outputs/impact_on_large_model"}
CUDA_DEVICE=${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
CHECKPOINT=${LARGE_AGENT_CHECKPOINT:-"$ROOT_DIR/eval/ckpt/vla_adapter_new/ours/outputs/20260502-112804/best_policy.pt"}

ARGS=(
  --env-id "${ENV_ID_OVERRIDE:-HoldCubeInHandObjectScaleDown1p2-v1}"
  --output-dir "$OUTPUT_DIR"
  --model-dir eval/ckpt/vla_adapter_new/LIBERO-Object
  --num-envs 256 --num-eval-envs 8 --num-steps 10
  --num-minibatches 16 --update-epochs 2
  --eval-episodes 50 --eval-every-updates 50
  --max-runtime-hours 5.1
  --large-agent-checkpoint "$CHECKPOINT"
  --small-model-scaling-strategy target-single-traj
  --small-model-scaling-policy small
  --small-model-feedback-alpha 0.1
  --small-model-regeneration-schedule once
  --max-sparsity 0.8
  --early-stop-zero-success-minutes 45000
  --cuda-device "$CUDA_DEVICE"
  --run-name "$RUN_NAME"
)

if [[ "$MWE" == "1" ]]; then
  export MWE_MAX_RUNTIME_MINUTES=${MWE_MAX_RUNTIME_MINUTES:-10}
  ARGS+=(--learning-rate 1e-7 --head-learning-rate 1e-7 --state-learning-rate 1e-7 --value-head-learning-rate 1e-7 --backbone-learning-rate 1e-7)
fi

python -u "$SCRIPT_DIR/vla_adapter_impl.py" "${ARGS[@]}"
