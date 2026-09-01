#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$ROOT_DIR/eval/common/resource_summary.sh"
vlaselect_resource_summary_start "$(basename "${BASH_SOURCE[0]}")"
trap 'vlaselect_resource_summary_finalize "$?"' EXIT
VERIFY_SCRIPT="$SCRIPT_DIR/vla_adapter_impl_verify.sh"
RESULTS_DIR="$ROOT_DIR/api/results/vla_adapter/scaling_methods"

# Use one GPU for the whole serial run. CUDA_DEVICE_OVERRIDE takes precedence;
# if a multi-GPU value is supplied, use its first device for this serial run.
CUDA_DEVICE=${CUDA_DEVICE_OVERRIDE:-${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}}
CUDA_DEVICE=${CUDA_DEVICE%%,*}
RUN_TAG=${RUN_TAG_OVERRIDE:-$(date +%Y%m%d-%H%M%S)}

SCALING_METHODS=(
  default
  attn_distillation
  data_distillation
  distillm
  edgeta
  feature_distillation
  llm_in_a_flash
  llm_pruner
  logit_distillation
  minillm
  powerinfer
)

run_method() {
  local method=$1
  local output_dir="$RESULTS_DIR/$method"
  local run_name="${RUN_TAG}-${method}"

  echo "[scaling] start method=$method gpu=$CUDA_DEVICE output=$output_dir"
  if [[ "$method" == "default" ]]; then
    CUDA_DEVICES="$CUDA_DEVICE" \
      OUTPUT_DIR_OVERRIDE="$output_dir" \
      RUN_NAME_OVERRIDE="$run_name" \
      bash "$VERIFY_SCRIPT"
  else
    CUDA_DEVICES="$CUDA_DEVICE" \
      OUTPUT_DIR_OVERRIDE="$output_dir" \
      RUN_NAME_OVERRIDE="$run_name" \
      bash "$VERIFY_SCRIPT" --scaling-method "$method"
  fi
  echo "[scaling] finished method=$method"
}

for method in "${SCALING_METHODS[@]}"; do
  run_method "$method"
done

PLOT_OUTPUT="$RESULTS_DIR/training_accuracy_curve.png"
python "$ROOT_DIR/api/plot_scaling_methods.py" \
  --results-dir "$RESULTS_DIR" \
  --output "$PLOT_OUTPUT"
echo "[scaling] plot=$PLOT_OUTPUT"
