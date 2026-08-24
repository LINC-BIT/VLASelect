#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ABLATION_SELECTION="neuron_grained_scaling_up:random,neuron_grained_scaling_up:inverse,neuron_grained_scaling_up:neuron_grained"
ABLATION_SELECTION="${ABLATION_SELECTION:-$DEFAULT_ABLATION_SELECTION}"

exec env ABLATION_SELECTION="$ABLATION_SELECTION" bash "$SCRIPT_DIR/run_ablation.sh" "$@"
