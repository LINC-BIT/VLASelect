#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$ROOT_DIR/eval/common/resource_summary.sh"
vlaselect_resource_summary_start "$(basename "${BASH_SOURCE[0]}")"
trap 'vlaselect_resource_summary_finalize "$?"' EXIT
MODE=${1:-${MWE:-0}}
if [[ "$MODE" != "0" && "$MODE" != "1" ]]; then
  echo "usage: $0 [0|1]" >&2
  exit 2
fi
export MWE="$MODE"
export TAIL_LOG=${TAIL_LOG:-1}
export RUN_NAME_OVERRIDE="results/cnn/compare-${MODE}"
bash "$SCRIPT_DIR/cnn.sh"
export RUN_NAME_OVERRIDE="results/mlp/compare-${MODE}"
bash "$SCRIPT_DIR/mlp.sh"
export RUN_NAME_OVERRIDE="results/cnn-conrft/compare-${MODE}"
bash "$SCRIPT_DIR/cnn-ppo-gen.sh"
export RUN_NAME_OVERRIDE="results/mlp-conrft/compare-${MODE}"
bash "$SCRIPT_DIR/mlp-ppo-gen.sh"
python "$SCRIPT_DIR/plot_ppo_compare.py" \
  --cnn-vlaselect "$SCRIPT_DIR/ckpt/results/cnn/compare-${MODE}/[agent]" \
  --cnn-conrft "$SCRIPT_DIR/ckpt/results/cnn-conrft/compare-${MODE}/[agent]" \
  --mlp-vlaselect "$SCRIPT_DIR/ckpt/results/mlp/compare-${MODE}/[agent]" \
  --mlp-conrft "$SCRIPT_DIR/ckpt/results/mlp-conrft/compare-${MODE}/[agent]"
