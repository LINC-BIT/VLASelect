#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
export MS_ASSET_DIR="${MS_ASSET_DIR:-$ROOT_DIR/eval/datasets}"
VERIFY_SCRIPT="$SCRIPT_DIR/tinyvla_impl_verify.sh"
RESULTS_DIR="$ROOT_DIR/api/results/tinyvla/knowledge_exchange"
CUDA_DEVICE=${CUDA_DEVICE_OVERRIDE:-${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}}
CUDA_DEVICE=${CUDA_DEVICE%%,*}
RUN_TAG=${RUN_TAG_OVERRIDE:-$(date +%Y%m%d-%H%M%S)}

KNOWLEDGE_EXCHANGE_GRANULARITIES=(default layer block attention_head)

for granularity in "${KNOWLEDGE_EXCHANGE_GRANULARITIES[@]}"; do
  output_dir="$RESULTS_DIR/$granularity"
  run_name="${RUN_TAG}-${granularity}"
  echo "[granularity] start granularity=$granularity gpu=$CUDA_DEVICE output=$output_dir"
  command=(bash "$VERIFY_SCRIPT")
  if [[ "$granularity" != default ]]; then
    command+=(--knowledge-exchange-granularity "$granularity")
  fi
  CUDA_DEVICES="$CUDA_DEVICE" OUTPUT_DIR_OVERRIDE="$output_dir" RUN_NAME_OVERRIDE="$run_name" "${command[@]}"
  echo "[granularity] finished granularity=$granularity"
done

python "$ROOT_DIR/api/plot_knowledge_exchange.py" \
  --results-dir "$RESULTS_DIR" \
  --output "$RESULTS_DIR/training_accuracy_curve.png"
echo "[granularity] plot=$RESULTS_DIR/training_accuracy_curve.png"
