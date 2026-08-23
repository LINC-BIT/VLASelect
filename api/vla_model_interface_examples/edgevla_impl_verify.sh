#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"

RUN_NAME=${RUN_NAME_OVERRIDE:-}
ENV_ID=${ENV_ID_OVERRIDE:-UnitreeG1LiftApple-v1}
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
  OUTPUT_DIR="$ROOT_DIR/api/results/edgevla/scaling_methods/$SCALING_METHOD"
elif [[ -n "$KNOWLEDGE_EXCHANGE_GRANULARITY" ]]; then
  OUTPUT_DIR="$ROOT_DIR/api/results/edgevla/knowledge_exchange/$KNOWLEDGE_EXCHANGE_GRANULARITY"
else
  OUTPUT_DIR="$SCRIPT_DIR/outputs/edgevla_online_rl_cl"
fi

# These values mirror eval/train/edgevla/ours/run_online_rl_cl.sh.
ARGS=(--env-id "$ENV_ID" --output-dir "$OUTPUT_DIR" \
  --envs-id "['UnitreeG1LiftCubeObjectScaleDown1p3-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeObjectPurple-v1','UnitreeG1LiftSphereLightStronger50-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectScaleDown1p1-v1','UnitreeG1LiftSphereObjectScaleDown1p3-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectPurple-v1']" \
  --env-change-time-points "[31,62,96,131,151,163,207,247,271,300]" \
  --control-mode pd_joint_delta_pos --reward-mode normalized_dense --obs-mode rgb+state_dict \
  --model-dir eval/ckpt/vla_adapter_new/LIBERO-Object \
  --num-envs 128 --num-eval-envs 8 --num-steps 64 --num-minibatches 16 --update-epochs 2 \
  --learning-rate 6e-5 --head-learning-rate 6e-5 --state-learning-rate 6e-5 --value-head-learning-rate 6e-5 --backbone-learning-rate 6e-5 \
  --weight-decay 1e-6 --gamma 0.99 --gae-lambda 0.95 --clip-coef 0.2 --ent-coef 1e-3 --vf-coef 0.5 --max-grad-norm 0.5 --target-kl 0.02 --minibatch-target-kl-factor 1.0 \
  --eval-episodes 16 --eval-every-updates 1 --max-runtime-hours 400 --rollout-micro-batch-size 256 --eval-micro-batch-size 256 --update-micro-batch-size 32 \
  --rollout-progress-log-interval 10 --freeze-vla-backbone false --backbone-warmup-updates 0 --save-video false \
  --action-dim 12 --state-dim 73 --env-action-dim 25 --controlled-action-indices "(2,4,6,8,10,14,15,16,20,21,22,24)" \
  --large-agent-checkpoint eval/ckpt/edgevla/ours/outputs/bc_unitree_g1_lift_apple_fbs/20260511-171959/best_policy.pt \
  --small-model-scaling-strategy target-single-traj --small-model-scaling-policy small \
  --small-model-feedback-schedule before_per_rollout_if_success_improv_is_larger_than_0.2 \
  --small-model-regeneration-schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters \
  --small-model-feedback-alpha 0.1 --small-model-regeneration-increment-ratio 0.05 --reset-optimizer-after-regeneration true --max-sparsity 0.8 \
  --early-stop-zero-success-minutes 45000 --cuda-device "$CUDA_VISIBLE_DEVICES")
if [[ -n "$SCALING_METHOD" ]]; then ARGS+=(--scaling-method "$SCALING_METHOD"); fi
if [[ -n "$KNOWLEDGE_EXCHANGE_GRANULARITY" ]]; then ARGS+=(--knowledge-exchange-granularity "$KNOWLEDGE_EXCHANGE_GRANULARITY"); fi
if [[ -n "$RUN_NAME" ]]; then ARGS+=(--run-name "$RUN_NAME"); fi
exec python -u "$SCRIPT_DIR/edgevla_impl.py" "${ARGS[@]}"
