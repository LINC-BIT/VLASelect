#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ABLATION_SELECTION="knowledge_accumulation:no_accumulation,knowledge_accumulation:accumulate_every_rollout,knowledge_accumulation:selective_accumulation"
ABLATION_SELECTION="${ABLATION_SELECTION:-$DEFAULT_ABLATION_SELECTION}"

exec env ABLATION_SELECTION="$ABLATION_SELECTION" bash "$SCRIPT_DIR/run_ablation.sh" "$@"
