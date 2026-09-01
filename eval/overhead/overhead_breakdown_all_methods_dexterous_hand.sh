#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SELECTION="${MODEL_SELECTION:-vla_adapter_new}"
ENV_ID_ORDER="${ENV_ID_ORDER:-}"
exec env MODEL_SELECTION="$MODEL_SELECTION" ENV_ID_ORDER="$ENV_ID_ORDER" bash "$SCRIPT_DIR/overhead_breakdown_all_methods.sh" "$@"
