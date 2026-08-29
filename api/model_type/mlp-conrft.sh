#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PRE_GENERATED_MODEL="${PPO_GEN_MLP_MODEL:-$SCRIPT_DIR/ppo-gen-mlp-model}"
export UPDATE_EPOCHS_OVERRIDE=1
export MAX_SPARSITY_OVERRIDE=0
exec bash "$SCRIPT_DIR/_run_model_type.sh" mlp "$@"
