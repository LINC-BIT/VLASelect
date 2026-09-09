#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$ROOT_DIR/eval/common/resource_summary.sh"
vlaselect_resource_summary_start "$(basename "${BASH_SOURCE[0]}")"
trap 'vlaselect_resource_summary_finalize "$?"' EXIT
VERIFY_SCRIPT="$SCRIPT_DIR/vla_adapter_impl_verify.sh"
RESULTS_DIR="$ROOT_DIR/api/results/vla_adapter/knowledge_exchange"

# Use one GPU for the whole serial run. CUDA_DEVICE_OVERRIDE takes precedence;
# if a multi-GPU value is supplied, use its first device for this serial run.
CUDA_DEVICE=${CUDA_DEVICE_OVERRIDE:-${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}}
CUDA_DEVICE=${CUDA_DEVICE%%,*}
RUN_TAG=${RUN_TAG_OVERRIDE:-$(date +%Y%m%d-%H%M%S)}

KNOWLEDGE_EXCHANGE_GRANULARITIES=(
  default
  layer
  block
  attention_head
)

run_granularity() {
  local granularity=$1
  local output_dir="$RESULTS_DIR/$granularity"
  local run_name="${RUN_TAG}-${granularity}"

  echo "[granularity] start granularity=$granularity gpu=$CUDA_DEVICE output=$output_dir"
  if [[ "$granularity" == "default" ]]; then
    CUDA_DEVICES="$CUDA_DEVICE" \
      OUTPUT_DIR_OVERRIDE="$output_dir" \
      RUN_NAME_OVERRIDE="$run_name" \
      bash "$VERIFY_SCRIPT"
  else
    CUDA_DEVICES="$CUDA_DEVICE" \
      OUTPUT_DIR_OVERRIDE="$output_dir" \
      RUN_NAME_OVERRIDE="$run_name" \
      bash "$VERIFY_SCRIPT" --knowledge-exchange-granularity "$granularity"
  fi
  echo "[granularity] finished granularity=$granularity"
}

for granularity in "${KNOWLEDGE_EXCHANGE_GRANULARITIES[@]}"; do
  run_granularity "$granularity"
done

python "$ROOT_DIR/api/plot_knowledge_exchange.py" \
  --results-dir "$RESULTS_DIR" \
  --output "$RESULTS_DIR/training_accuracy_curve.png"
echo "[granularity] plot=$RESULTS_DIR/training_accuracy_curve.png"
