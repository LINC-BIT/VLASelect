#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

OUTPUT=${OUTPUT_OVERRIDE:-"$SCRIPT_DIR/overhead_breakdown.json"}
PYTHON_BIN=${PYTHON_BIN:-python}

exec "$PYTHON_BIN" "$SCRIPT_DIR/benchmark.py" \
  --output "$OUTPUT" \
  "${@}"
