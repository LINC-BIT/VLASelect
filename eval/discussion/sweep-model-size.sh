#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_SELECTION="${MODEL_SELECTION:-}"
exec bash "${SCRIPT_DIR}/sweep_model_size.sh"
