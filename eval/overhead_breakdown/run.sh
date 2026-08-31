#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$ROOT_DIR/eval/common/resource_summary.sh"
vlaselect_resource_summary_start "$(basename "${BASH_SOURCE[0]}")"
trap 'vlaselect_resource_summary_finalize "$?"' EXIT
cd "$ROOT_DIR"

OUTPUT=${OUTPUT_OVERRIDE:-"$SCRIPT_DIR/overhead_breakdown.json"}
PYTHON_BIN=${PYTHON_BIN:-python}

"$PYTHON_BIN" "$SCRIPT_DIR/benchmark.py" \
  --output "$OUTPUT" \
  "${@}"
