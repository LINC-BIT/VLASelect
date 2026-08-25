#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
export MS_ASSET_DIR="${MS_ASSET_DIR:-$ROOT_DIR/eval/datasets}"
VERIFY_SCRIPT="$SCRIPT_DIR/tinyvla_impl_verify.sh"
RESULTS_DIR="$ROOT_DIR/api/results/tinyvla/scaling_methods_only_4"
CUDA_DEVICE=${CUDA_DEVICE_OVERRIDE:-${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}}
CUDA_DEVICE=${CUDA_DEVICE%%,*}
RUN_TAG=${RUN_TAG_OVERRIDE:-$(date +%Y%m%d-%H%M%S)}

SCALING_METHODS=(
  default
  attn_distillation
  llm_in_a_flash
  edgeta
)

for method in "${SCALING_METHODS[@]}"; do
  output_dir="$RESULTS_DIR/$method"
  run_name="${RUN_TAG}-${method}"
  echo "[scaling] start method=$method gpu=$CUDA_DEVICE output=$output_dir"
  command=(bash "$VERIFY_SCRIPT")
  if [[ "$method" != default ]]; then command+=(--scaling-method "$method"); fi
  CUDA_DEVICES="$CUDA_DEVICE" OUTPUT_DIR_OVERRIDE="$output_dir" RUN_NAME_OVERRIDE="$run_name" "${command[@]}"
  echo "[scaling] finished method=$method"
done

PLOT_OUTPUT="$RESULTS_DIR/training_accuracy_curve.png"
python "$ROOT_DIR/api/plot_scaling_methods.py" --results-dir "$RESULTS_DIR" --output "$PLOT_OUTPUT"
echo "[scaling] plot=$PLOT_OUTPUT"
