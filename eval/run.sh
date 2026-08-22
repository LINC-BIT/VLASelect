#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR"

MWE="${MWE:-0}"

: "${MWE_RUNTIME_LIMIT_SECONDS:=300}"
export MWE_RUNTIME_LIMIT_SECONDS
if [[ "$MWE" == "1" && "${MWE_TIMEOUT_APPLIED:-0}" != "1" ]]; then
    if command -v timeout >/dev/null 2>&1; then
        export MWE_TIMEOUT_APPLIED=1
        exec timeout --preserve-status -k 10s "${MWE_RUNTIME_LIMIT_SECONDS}s" bash "$SCRIPT_PATH" "$@"
    fi
    echo "[warn] timeout command not found; MWE runtime is not hard-capped" >&2
fi
AUTO_POSTPROCESS="${AUTO_POSTPROCESS:-1}"
RUN_ACC_TASK_ENV="${RUN_ACC_TASK_ENV:-1}"
RUN_ACC_RES_CHANGE="${RUN_ACC_RES_CHANGE:-1}"
RUN_OVERHEAD_SAME_ACC="${RUN_OVERHEAD_SAME_ACC:-1}"
RUN_BREAKDOWN_ALL="${RUN_BREAKDOWN_ALL:-1}"
RUN_BREAKDOWN_MODULES="${RUN_BREAKDOWN_MODULES:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"

log() {
    echo "[run.sh] $*"
}

run_step() {
    local label="$1"
    shift
    log "starting: ${label}"
    "$@"
    log "finished: ${label}"
}

if [[ "$RUN_ACC_TASK_ENV" == "1" ]]; then
    run_step "Figure 7: task/environment accuracy" env MWE="$MWE" bash acc_comparison/run_acc_task_env_change.sh
    if [[ "$AUTO_POSTPROCESS" == "1" ]]; then
        run_step "Figure 7 postprocess" python acc_comparison/plot_acc_task_env.py
    fi
fi

if [[ "$RUN_ACC_RES_CHANGE" == "1" ]]; then
    run_step "Figure 8: resource-change accuracy" env MWE="$MWE" bash acc_comparison/run_acc_res_change.sh
    if [[ "$AUTO_POSTPROCESS" == "1" ]]; then
        run_step "Figure 8 postprocess" python acc_comparison/plot_acc_res_change.py
    fi
fi

if [[ "$RUN_OVERHEAD_SAME_ACC" == "1" ]]; then
    run_step "Figure 9 and Tables 2-3: same-accuracy overhead" env MWE="$MWE" bash overhead/overhead_same_acc.sh
    if [[ "$AUTO_POSTPROCESS" == "1" ]]; then
        run_step "Figure 9 postprocess" python overhead/plot_overhead.py
    fi
fi

if [[ "$RUN_BREAKDOWN_ALL" == "1" ]]; then
    run_step "Figure 10: breakdown for all methods" env MWE="$MWE" bash overhead/overhead_breakdown_all_methods.sh
    if [[ "$AUTO_POSTPROCESS" == "1" ]]; then
        run_step "Figure 10 postprocess" python overhead/plot_breakdown_all_methods.py
    fi
fi

if [[ "$RUN_BREAKDOWN_MODULES" == "1" ]]; then
    run_step "Figure 11: breakdown for VLASelect modules" env MWE="$MWE" bash overhead/overhead_breakdown_modules.sh
    if [[ "$AUTO_POSTPROCESS" == "1" ]]; then
        run_step "Figure 11 postprocess" python overhead/plot_breakdown_modules.py
    fi
fi

if [[ "$RUN_ABLATION" == "1" ]]; then
    run_step "Figure 12: ablation" env MWE="$MWE" bash ablation/run_ablation.sh
    if [[ "$AUTO_POSTPROCESS" == "1" ]]; then
        run_step "Figure 12 postprocess" python ablation/plot_ablation.py
    fi
fi

cat <<'MSG'
[run.sh] primary evaluation scripts finished.
[run.sh] Notes:
[run.sh] - The postprocess scripts now generate panel images first and then stitch them into final figures close to the paper layout.
[run.sh] - To run the short sanity-check mode instead, use `MWE=1 bash run.sh`.
MSG
