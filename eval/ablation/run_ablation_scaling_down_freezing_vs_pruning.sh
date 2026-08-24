#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ABLATION_SELECTION="scaling_down_freezing_vs_pruning:pruning,scaling_down_freezing_vs_pruning:freezing"
ABLATION_SELECTION="${ABLATION_SELECTION:-$DEFAULT_ABLATION_SELECTION}"

exec env ABLATION_SELECTION="$ABLATION_SELECTION" bash "$SCRIPT_DIR/run_ablation.sh" "$@"
