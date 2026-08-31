#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$ROOT_DIR/eval/common/resource_summary.sh"
vlaselect_resource_summary_start "$(basename "${BASH_SOURCE[0]}")"
trap 'vlaselect_resource_summary_finalize "$?"' EXIT
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$ROOT_DIR/eval/datasets}"
cd "$ROOT_DIR"

MWE=${MWE:-0}
if [[ "$MWE" == "1" ]]; then
  export MWE_MAX_RUNTIME_MINUTES=2
fi
OUTPUT_DIR=${OUTPUT_DIR_OVERRIDE:-"$SCRIPT_DIR/outputs/tinyvla_online_rl_cl"}
RUN_NAME=${RUN_NAME_OVERRIDE:-$(date +%Y%m%d-%H%M%S)}
ENV_ID=${ENV_ID_OVERRIDE:-OpenCabinetDrawerCabinet1021Default-v1}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
export CUDA_VISIBLE_DEVICES

SCALING_METHOD=
KNOWLEDGE_EXCHANGE_GRANULARITY=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scaling-method)
      [[ $# -ge 2 ]] || { echo "--scaling-method requires a value" >&2; exit 2; }
      SCALING_METHOD=$2; shift 2 ;;
    --knowledge-exchange-granularity)
      [[ $# -ge 2 ]] || { echo "--knowledge-exchange-granularity requires a value" >&2; exit 2; }
      KNOWLEDGE_EXCHANGE_GRANULARITY=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
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
LARGE_AGENT_CHECKPOINT="$ROOT_DIR/eval/ckpt/tinyvla/ours/outputs/bc_open_cabinet_drawer_fbs/20260508-032529/best_policy.pt"
if [[ "$MWE" == "1" && ( -n "$SCALING_METHOD" || -n "$KNOWLEDGE_EXCHANGE_GRANULARITY" ) ]]; then
  LARGE_AGENT_CHECKPOINT="${LARGE_AGENT_CHECKPOINT}.base"
  [[ -f "$LARGE_AGENT_CHECKPOINT" ]] || {
    echo "missing MWE base checkpoint: $LARGE_AGENT_CHECKPOINT" >&2
    exit 2
  }
fi
if [[ -n "${OUTPUT_DIR_OVERRIDE:-}" ]]; then
  OUTPUT_DIR=$OUTPUT_DIR_OVERRIDE
elif [[ -n "$SCALING_METHOD" ]]; then
  OUTPUT_DIR="$ROOT_DIR/api/results/tinyvla/scaling_methods/$SCALING_METHOD"
elif [[ -n "$KNOWLEDGE_EXCHANGE_GRANULARITY" ]]; then
  OUTPUT_DIR="$ROOT_DIR/api/results/tinyvla/knowledge_exchange/$KNOWLEDGE_EXCHANGE_GRANULARITY"
fi

ARGS=(--env-id "$ENV_ID" --output-dir "$OUTPUT_DIR" \
  --envs-id "['OpenCabinetDrawerCabinet1021Default-v1','OpenCabinetDrawerCabinet1016ScaleUp1p3-v1','OpenCabinetDrawerCabinet1027Default-v1','OpenCabinetDrawerCabinet1016ScaleUp1p3-v1','OpenCabinetDrawerCabinet1032Default-v1','OpenCabinetDrawerCabinet1033ScaleUp1p3-v1','OpenCabinetDrawerCabinet1027Default-v1','OpenCabinetDrawerCabinet1021Default-v1','OpenCabinetDrawerCabinet1032Default-v1','OpenCabinetDrawerCabinet1033ScaleUp1p3-v1']" \
  --env-change-time-points "[31,62,96,131,151,163,207,247,271,300]" \
  --control-mode pd_joint_delta_pos --reward-mode normalized_dense --obs-mode rgb+state_dict \
  --model-dir eval/ckpt/vla_adapter_new/LIBERO-Object \
  --num-envs 256 --num-eval-envs 8 --num-steps 50 --num-minibatches 16 --update-epochs 2 \
  --learning-rate 3e-5 --head-learning-rate 3e-5 --state-learning-rate 3e-5 --value-head-learning-rate 3e-5 --backbone-learning-rate 3e-5 \
  --weight-decay 1e-6 --gamma 0.8 --gae-lambda 0.9 --clip-coef 0.2 --ent-coef 0.0 --vf-coef 0.5 --max-grad-norm 0.5 --target-kl 0.2 --minibatch-target-kl-factor 1.0 \
  --eval-episodes 50 --eval-every-updates 50 --max-runtime-hours 5.1 --rollout-micro-batch-size 256 --eval-micro-batch-size 256 --update-micro-batch-size 32 \
  --freeze-vla-backbone false --backbone-warmup-updates 0 --save-video false --action-dim 8 --state-dim 44 --env-action-dim 13 --controlled-action-indices "(0,1,2,3,4,5,6,7)" \
  --large-agent-checkpoint "$LARGE_AGENT_CHECKPOINT" \
  --small-model-scaling-strategy target-single-traj --small-model-scaling-policy small \
  --small-model-feedback-schedule before_per_rollout_if_success_improv_is_larger_than_0.2 \
  --small-model-regeneration-schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters \
  --small-model-feedback-alpha 0.1 --small-model-regeneration-increment-ratio 0.05 --reset-optimizer-after-regeneration true --max-sparsity 0.8 \
  --early-stop-zero-success-minutes 45000 --cuda-device "$CUDA_VISIBLE_DEVICES")
if [[ -n "$SCALING_METHOD" ]]; then ARGS+=(--scaling-method "$SCALING_METHOD"); fi
if [[ -n "$KNOWLEDGE_EXCHANGE_GRANULARITY" ]]; then ARGS+=(--knowledge-exchange-granularity "$KNOWLEDGE_EXCHANGE_GRANULARITY"); fi
ARGS+=(--run-name "$RUN_NAME")
python -u "$SCRIPT_DIR/tinyvla_impl.py" "${ARGS[@]}"

RUN_DIR="$OUTPUT_DIR/$RUN_NAME"
if [[ -n "$SCALING_METHOD" ]]; then
  python "$ROOT_DIR/api/plot_scaling_methods.py" \
    --results-dir "$RUN_DIR" \
    --output "$RUN_DIR/training_accuracy_curve.png"
elif [[ -n "$KNOWLEDGE_EXCHANGE_GRANULARITY" ]]; then
  python "$ROOT_DIR/api/plot_knowledge_exchange.py" \
    --results-dir "$RUN_DIR" \
    --output "$RUN_DIR/training_accuracy_curve.png"
else
  python "$ROOT_DIR/api/plot_scaling_methods.py" \
    --results-dir "$RUN_DIR" \
    --output "$RUN_DIR/training_accuracy_curve.png"
fi
echo "[plot] output=$RUN_DIR/training_accuracy_curve.png"
