#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec env   MWE="${MWE:-1}"   METHODS="${METHODS:-self_improv,vla_rft,world_env,vlaselect}"   SAME_ACC_ACCURACY_COMPAT="${SAME_ACC_ACCURACY_COMPAT:-1}"   SAME_ACC_BREAKDOWN_COMPAT="${SAME_ACC_BREAKDOWN_COMPAT:-1}"   MODEL_SELECTION="${MODEL_SELECTION:-vla_adapter_new}"   bash "$SCRIPT_DIR/overhead_same_acc_dexterous_hand.sh" "$@"
