#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PRE_GENERATED_MODEL="${RLVLA_CNN_MODEL:-$SCRIPT_DIR/RLVLA-cnn-model}"
export UPDATE_EPOCHS_OVERRIDE=1
export MAX_SPARSITY_OVERRIDE=0
exec bash "$SCRIPT_DIR/_run_model_type.sh" cnn "$@"
