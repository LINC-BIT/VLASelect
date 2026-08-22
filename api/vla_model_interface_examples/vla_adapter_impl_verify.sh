#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"

# These task/environment values mirror eval/train/vla_adapter_new/ours/run_online_rl_cl.sh.
MWE=${MWE:-0}
RUN_NAME=${RUN_NAME_OVERRIDE:-}
ENV_ID=${ENV_ID_OVERRIDE:-HoldCubeInHandObjectScaleDown1p2-v1}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
export CUDA_VISIBLE_DEVICES

SCALING_METHOD=
KNOWLEDGE_EXCHANGE_GRANULARITY=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scaling-method)
      [[ $# -ge 2 ]] || { echo "--scaling-method requires a value" >&2; exit 2; }
      SCALING_METHOD=$2
      shift 2
      ;;
    --knowledge-exchange-granularity)
      [[ $# -ge 2 ]] || { echo "--knowledge-exchange-granularity requires a value" >&2; exit 2; }
      KNOWLEDGE_EXCHANGE_GRANULARITY=$2
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$SCALING_METHOD" && -n "$KNOWLEDGE_EXCHANGE_GRANULARITY" ]]; then
  echo "--scaling-method and --knowledge-exchange-granularity are mutually exclusive" >&2
  exit 2
fi
for value in "$SCALING_METHOD" "$KNOWLEDGE_EXCHANGE_GRANULARITY"; do
  if [[ -n "$value" && ! "$value" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "invalid method/granularity name: $value" >&2
    exit 2
  fi
done

if [[ -n "${OUTPUT_DIR_OVERRIDE:-}" ]]; then
  OUTPUT_DIR=$OUTPUT_DIR_OVERRIDE
elif [[ -n "$SCALING_METHOD" ]]; then
  OUTPUT_DIR="$ROOT_DIR/api/results/scaling_methods/$SCALING_METHOD"
elif [[ -n "$KNOWLEDGE_EXCHANGE_GRANULARITY" ]]; then
  OUTPUT_DIR="$ROOT_DIR/api/results/knowledge_exchange/$KNOWLEDGE_EXCHANGE_GRANULARITY"
else
  OUTPUT_DIR="$SCRIPT_DIR/outputs/vla_adapter_online_rl_cl"
fi

# Shorter rollouts emit training accuracy/success metrics more frequently.
ARGS=(--env-id "$ENV_ID" --output-dir "$OUTPUT_DIR" \
  --envs-id "['HoldCubeInHandObjectScaleDown1p2-v1','HoldHammerInHandObjectScaleDown1p6-v1','HoldWrenchInHandObjectScaleUp1p2-v1','HoldWoodBlockInHandObjectScaleDown1p6-v1','HoldHammerInHandObjectScaleUp1p6-v1','HoldHammerInHandObjectScaleDown1p4-v1','HoldWrenchInHandObjectScaleUp1p6-v1','HoldHammerInHandObjectScaleDown1p2-v1','HoldHammerInHandObjectScaleUp1p4-v1','HoldWrenchInHandObjectScaleDown1p6-v1']" \
  --env-change-time-points "[31,62,96,131,151,163,207,247,271,300]" \
  --control-mode pd_joint_delta_pos --reward-mode normalized_dense --obs-mode rgb+state_dict \
  --model-dir eval/ckpt/vla_adapter_new/LIBERO-Object \
  --num-envs 256 --num-eval-envs 8 --num-steps 10 --num-minibatches 16 --update-epochs 2 \
  --learning-rate 3e-5 --head-learning-rate 3e-5 --state-learning-rate 3e-5 --value-head-learning-rate 3e-5 --backbone-learning-rate 3e-5 \
  --weight-decay 1e-6 --gamma 0.8 --gae-lambda 0.9 --clip-coef 0.2 --ent-coef 0.0 --vf-coef 0.5 --max-grad-norm 0.5 --target-kl 0.2 --minibatch-target-kl-factor 1.0 \
  --eval-episodes 50 --eval-every-updates 50 --max-runtime-hours 5.1 --rollout-micro-batch-size 256 --eval-micro-batch-size 256 --update-micro-batch-size 32 \
  --freeze-vla-backbone false --backbone-warmup-updates 0 --save-video false --action-dim 16 --state-dim 105 \
  --large-agent-checkpoint eval/ckpt/vla_adapter_new/ours/outputs/20260502-112804/best_policy.pt \
  --small-model-scaling-strategy target-single-traj --small-model-scaling-policy small \
  --small-model-feedback-schedule before_per_rollout_if_success_improv_is_larger_than_0.2 \
  --small-model-regeneration-schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters \
  --small-model-feedback-alpha 0.1 --small-model-regeneration-increment-ratio 0.05 --reset-optimizer-after-regeneration true --max-sparsity 0.8 \
  --early-stop-zero-success-minutes 45000 --cuda-device "$CUDA_VISIBLE_DEVICES")
if [[ -n "$SCALING_METHOD" ]]; then ARGS+=(--scaling-method "$SCALING_METHOD"); fi
if [[ -n "$KNOWLEDGE_EXCHANGE_GRANULARITY" ]]; then
  ARGS+=(--knowledge-exchange-granularity "$KNOWLEDGE_EXCHANGE_GRANULARITY")
fi
if [[ -n "$RUN_NAME" ]]; then ARGS+=(--run-name "$RUN_NAME"); fi
exec python -u "$SCRIPT_DIR/vla_adapter_impl.py" "${ARGS[@]}"
