#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "$ROOT_DIR/common/resource_summary.sh"
vlaselect_resource_summary_start "$(basename "${BASH_SOURCE[0]}")"
trap 'vlaselect_resource_summary_finalize "$?"' EXIT
cd "$SCRIPT_DIR"

MWE="${MWE:-0}"

: "${MWE_RUNTIME_LIMIT_SECONDS:=300}"
export MWE_RUNTIME_LIMIT_SECONDS
# Do not wrap the whole suite in one timeout. Each child experiment enforces
# its own MWE cap, and some of them launch detached workers that must manage
# cleanup and result writing independently.
AUTO_POSTPROCESS="${AUTO_POSTPROCESS:-1}"
RUN_ACC_TASK_ENV="${RUN_ACC_TASK_ENV:-1}"
RUN_ACC_RES_CHANGE="${RUN_ACC_RES_CHANGE:-1}"
RUN_OVERHEAD_SAME_ACC="${RUN_OVERHEAD_SAME_ACC:-1}"
RUN_BREAKDOWN_ALL="${RUN_BREAKDOWN_ALL:-1}"
RUN_BREAKDOWN_MODULES="${RUN_BREAKDOWN_MODULES:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
METHODS="${METHODS:-${RUN_METHODS:-}}"
RUN_DISCUSSION_ICL="${RUN_DISCUSSION_ICL:-1}"
RUN_DISCUSSION_MODEL_SIZE="${RUN_DISCUSSION_MODEL_SIZE:-1}"
RUN_DISCUSSION_MULTI_AGENT="${RUN_DISCUSSION_MULTI_AGENT:-1}"
RUN_DISCUSSION_ALT_SCALING="${RUN_DISCUSSION_ALT_SCALING:-1}"
RUN_DISCUSSION_GRANULARITY="${RUN_DISCUSSION_GRANULARITY:-1}"
RUN_DISCUSSION_FORGETTING="${RUN_DISCUSSION_FORGETTING:-1}"
RUN_DISCUSSION_MLP_CNN="${RUN_DISCUSSION_MLP_CNN:-1}"
RUN_DISCUSSION_SIM_TO_REAL="${RUN_DISCUSSION_SIM_TO_REAL:-0}"
DISCUSSION_MODEL_SIZE_FAMILY="${DISCUSSION_MODEL_SIZE_FAMILY:-tinyvla}"

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
    run_step "Figure 7: task/environment accuracy" env MWE="$MWE" METHODS="$METHODS" bash acc_comparison/run_acc_task_env_change.sh
    if [[ "$AUTO_POSTPROCESS" == "1" ]]; then
        run_step "Figure 7 postprocess" python acc_comparison/plot_acc_task_env.py
    fi
fi

if [[ "$RUN_ACC_RES_CHANGE" == "1" ]]; then
    run_step "Figure 8: resource-change accuracy" env MWE="$MWE" METHODS="$METHODS" bash acc_comparison/run_acc_res_change.sh
    if [[ "$AUTO_POSTPROCESS" == "1" ]]; then
        run_step "Figure 8 postprocess" python acc_comparison/plot_acc_res_change.py
    fi
fi

if [[ "$RUN_OVERHEAD_SAME_ACC" == "1" ]]; then
    run_step "Figure 9 and Tables 2-3: same-accuracy overhead" env MWE="$MWE" METHODS="$METHODS" bash overhead/overhead_same_acc.sh
    if [[ "$AUTO_POSTPROCESS" == "1" ]]; then
        run_step "Figure 9 postprocess" python overhead/plot_overhead.py
    fi
fi

if [[ "$RUN_BREAKDOWN_ALL" == "1" ]]; then
    run_step "Figure 10: breakdown for all methods" env MWE="$MWE" METHODS="$METHODS" bash overhead_breakdown.sh
    if [[ "$AUTO_POSTPROCESS" == "1" ]]; then
        run_step "Figure 10 postprocess" python overhead_breakdown/benchmark.py
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

if [[ "$RUN_DISCUSSION_SIM_TO_REAL" == "1" ]]; then
    run_step "Discussion 1: sim-to-real transfer" bash discussion/run_sim_to_real.sh
fi

if [[ "$RUN_DISCUSSION_ICL" == "1" ]]; then
    run_step "Discussion 2: ICL" env MWE="$MWE" bash discussion/compare_icl.sh
fi

if [[ "$RUN_DISCUSSION_MODEL_SIZE" == "1" ]]; then
    run_step "Discussion 3: maximum supported model size" env MODEL_SIZE_LIMIT_FAMILY="$DISCUSSION_MODEL_SIZE_FAMILY" bash discussion/sweep_model_size.sh
fi

if [[ "$RUN_DISCUSSION_MULTI_AGENT" == "1" ]]; then
    run_step "Discussion 4: multi-agent scenarios" env MWE="$MWE" bash discussion/run_multi_agent.sh
fi

if [[ "$RUN_DISCUSSION_ALT_SCALING" == "1" ]]; then
    if [[ "$MWE" == "1" ]]; then
        run_step "Discussion 5: alternative scaling techniques (3 representative methods)" bash "$ROOT_DIR/api/vla_model_interface_examples/vla_adapter_impl_verify-all_scaling_methods-only4.sh"
    else
        run_step "Discussion 5: alternative scaling techniques" bash "$ROOT_DIR/api/vla_model_interface_examples/vla_adapter_impl_verify-all_scaling_methods.sh"
    fi
fi

if [[ "$RUN_DISCUSSION_GRANULARITY" == "1" ]]; then
    run_step "Discussion 6: knowledge exchange granularities" env MWE="$MWE" bash "$ROOT_DIR/api/vla_model_interface_examples/vla_adapter_impl_verify-all_granularities.sh"
fi

if [[ "$RUN_DISCUSSION_FORGETTING" == "1" ]]; then
    run_step "Discussion 7: forgetting on previous tasks" env MWE="$MWE" bash forgetting/measure_forgetting.sh
fi

if [[ "$RUN_DISCUSSION_MLP_CNN" == "1" ]]; then
    run_step "Discussion 8: MLP/CNN applicability" bash "$ROOT_DIR/api/model_type/run.sh" "$MWE"
fi

cat <<'MSG'
[run.sh] primary evaluation scripts finished.
[run.sh] Notes:
[run.sh] - The postprocess scripts now generate panel images first and then stitch them into final figures close to the paper layout.
[run.sh] - Each experiment entry script performs its own preflight sanity check before launch.
[run.sh] - Set `METHODS=self_improv,vla_rft,world_env,vlaselect` to restrict the main multi-method experiments (Figures 7-10 except the VLASelect-only module breakdown); ablation and discussion ignore this filter.
[run.sh] - Main experiment figures are written under `eval/acc_comparison/`, `eval/overhead/`, and `eval/ablation/`, including `FIG_ACC_TASK_ENV.*`, `FIG_ACC_RESOURCE.*`, `FIG_MEMORY_FOOTPOINT.*`, `FIG_BREAKDOWN_ALL_METHODS.*`, `FIG_BREAKDOWN_MODULES.*`, and `FIG_ABLATION.*`.
[run.sh] - Discussion outputs are written under `eval/discussion/results/` plus fixed summary plots such as `eval/discussion/FIG_VLA_APPLICABILITY.*`; model-scaling discussion outputs are under `api/results/vla_adapter/`; MLP/CNN applicability outputs are under `api/model_type/`.
[run.sh] - Raw checkpoints, metrics histories, and intermediate manifests remain under `eval/ckpt/`, `eval/overhead/*table/`, `eval/ablation/ablation_table/`, `eval/forgetting/results/`, and `eval/discussion/results/`.
[run.sh] - Discussion 1 (sim-to-real) is skipped by default because it requires physical robot hardware; run `RUN_DISCUSSION_SIM_TO_REAL=1 bash run.sh` only on the supported setup.
[run.sh] - To run the short sanity-check mode instead, use `MWE=1 bash run.sh`.
MSG
