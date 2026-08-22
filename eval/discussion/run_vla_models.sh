#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
EVAL_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "$EVAL_ROOT"
source "${EVAL_ROOT}/common/interrupt_cleanup.sh"

STAMP=${VLA_APPLICABILITY_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}
SHORT_TRAIN=${VLA_APPLICABILITY_SHORT_TRAIN:-${VLA_APPLICABILITY_SMOKE:-1}}
WAIT_FOR_COMPLETION=${VLA_APPLICABILITY_WAIT:-1}
MWE=${MWE:-0}

: "${MWE_RUNTIME_LIMIT_SECONDS:=300}"
export MWE_RUNTIME_LIMIT_SECONDS
if [[ "$MWE" == "1" && "${MWE_TIMEOUT_APPLIED:-0}" != "1" ]]; then
    if command -v timeout >/dev/null 2>&1; then
        export MWE_TIMEOUT_APPLIED=1
        exec timeout --preserve-status -k 10s "${MWE_RUNTIME_LIMIT_SECONDS}s" bash "$SCRIPT_PATH" "$@"
    fi
    echo "[warn] timeout command not found; MWE runtime is not hard-capped" >&2
fi
MODEL_SELECTION="${MODEL_SELECTION:-${MODEL_FAMILY:-}}"
if [[ "$MWE" == "1" ]]; then
    SHORT_TRAIN=1
    WAIT_FOR_COMPLETION=0
fi
vlaselect_install_cleanup_trap
TAIL_LOG=${TAIL_LOG:-1}
CUDA_DEVICES=${CUDA_DEVICES:-0}
MANIFEST_PATH="${SCRIPT_DIR}/vla_applicability_${STAMP}.json"
vlaselect_register_cleanup_manifest "$MANIFEST_PATH"

SHORT_TOTAL_TIMESTEPS=${VLA_APPLICABILITY_TOTAL_TIMESTEPS:-64}
SHORT_NUM_ENVS=${VLA_APPLICABILITY_NUM_ENVS:-2}
SHORT_NUM_EVAL_ENVS=${VLA_APPLICABILITY_NUM_EVAL_ENVS:-1}
SHORT_NUM_STEPS=${VLA_APPLICABILITY_NUM_STEPS:-2}
SHORT_NUM_MINIBATCHES=${VLA_APPLICABILITY_NUM_MINIBATCHES:-1}
SHORT_UPDATE_EPOCHS=${VLA_APPLICABILITY_UPDATE_EPOCHS:-1}
SHORT_EVAL_EVERY_UPDATES=${VLA_APPLICABILITY_EVAL_EVERY_UPDATES:-1}
SHORT_EVAL_EPISODES=${VLA_APPLICABILITY_EVAL_EPISODES:-2}
SHORT_MAX_RUNTIME_HOURS=${VLA_APPLICABILITY_MAX_RUNTIME_HOURS:-0.25}
SHORT_ROLLOUT_MICRO_BATCH_SIZE=${VLA_APPLICABILITY_ROLLOUT_MICRO_BATCH_SIZE:-2}
SHORT_EVAL_MICRO_BATCH_SIZE=${VLA_APPLICABILITY_EVAL_MICRO_BATCH_SIZE:-2}
SHORT_UPDATE_MICRO_BATCH_SIZE=${VLA_APPLICABILITY_UPDATE_MICRO_BATCH_SIZE:-2}
SHORT_TRAIN_VIDEO_NUM_ENVS=${VLA_APPLICABILITY_TRAIN_VIDEO_NUM_ENVS:-1}
SHORT_TEST_VIDEO_NUM_ENVS=${VLA_APPLICABILITY_TEST_VIDEO_NUM_ENVS:-1}
SHORT_TEST_VIDEO_EPISODES=${VLA_APPLICABILITY_TEST_VIDEO_EPISODES:-1}
SHORT_EARLY_STOP_ZERO_SUCCESS_MINUTES=${VLA_APPLICABILITY_EARLY_STOP_ZERO_SUCCESS_MINUTES:-15}

mkdir -p "${SCRIPT_DIR}"

select_models() {
    local raw_selection="$1"
    if [[ -z "$raw_selection" ]]; then
        printf "%s\n" octo vla_adapter_new tinyvla edgevla
        return
    fi
    printf "%s" "$raw_selection" | tr ',' '\n' | awk 'NF {gsub(/^[ \t]+|[ \t]+$/, ""); print}'
}

declare -a RUNS=()
declare -A SHOULD_RUN_MODEL=(
    [octo]=0
    [vla_adapter_new]=0
    [tinyvla]=0
    [edgevla]=0
)

SELECTED_MODEL_COUNT=0
while IFS= read -r family; do
    [[ -z "$family" ]] && continue
    if [[ -z "${SHOULD_RUN_MODEL[$family]+x}" ]]; then
        echo "[error] unknown model in MODEL_SELECTION: $family" >&2
        exit 1
    fi
    if [[ "${SHOULD_RUN_MODEL[$family]}" == "0" ]]; then
        SHOULD_RUN_MODEL["$family"]=1
        SELECTED_MODEL_COUNT=$((SELECTED_MODEL_COUNT + 1))
    fi
done < <(select_models "$MODEL_SELECTION")

if [[ "$SELECTED_MODEL_COUNT" -eq 0 ]]; then
    echo "[error] No models selected." >&2
    exit 1
fi


append_run() {
    local kind="$1"
    local family="$2"
    local pid="$3"
    local run_dir="$4"
    local log_file="$5"
    RUNS+=("${kind}|${family}|${pid}|${run_dir}|${log_file}")
}

launch_family() {
    local family="$1"
    local script_path="$2"
    local output_dir_base="$3"
    local run_name="$4"

    local cmd_output
    if [ "$SHORT_TRAIN" = "1" ]; then
        cmd_output=$(env             CUDA_DEVICES="$CUDA_DEVICES"             OUTPUT_DIR_BASE_OVERRIDE="$output_dir_base"             RUN_NAME_OVERRIDE="$run_name"             TAIL_LOG="$TAIL_LOG"             SAVE_VIDEO_OVERRIDE=false             NUM_ENVS_OVERRIDE="$SHORT_NUM_ENVS"             NUM_EVAL_ENVS_OVERRIDE="$SHORT_NUM_EVAL_ENVS"             NUM_STEPS_OVERRIDE="$SHORT_NUM_STEPS"             TOTAL_TIMESTEPS_OVERRIDE="$SHORT_TOTAL_TIMESTEPS"             NUM_MINIBATCHES_OVERRIDE="$SHORT_NUM_MINIBATCHES"             UPDATE_EPOCHS_OVERRIDE="$SHORT_UPDATE_EPOCHS"             EVAL_EVERY_UPDATES_OVERRIDE="$SHORT_EVAL_EVERY_UPDATES"             EVAL_EPISODES_OVERRIDE="$SHORT_EVAL_EPISODES"             MAX_RUNTIME_HOURS_OVERRIDE="$SHORT_MAX_RUNTIME_HOURS"             ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE="$SHORT_ROLLOUT_MICRO_BATCH_SIZE"             EVAL_MICRO_BATCH_SIZE_OVERRIDE="$SHORT_EVAL_MICRO_BATCH_SIZE"             UPDATE_MICRO_BATCH_SIZE_OVERRIDE="$SHORT_UPDATE_MICRO_BATCH_SIZE"             TRAIN_VIDEO_NUM_ENVS_OVERRIDE="$SHORT_TRAIN_VIDEO_NUM_ENVS"             TEST_VIDEO_NUM_ENVS_OVERRIDE="$SHORT_TEST_VIDEO_NUM_ENVS"             TEST_VIDEO_EPISODES_OVERRIDE="$SHORT_TEST_VIDEO_EPISODES"             EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE="$SHORT_EARLY_STOP_ZERO_SUCCESS_MINUTES"             RUN_SETUP_SMOKE_OVERRIDE=false             bash "$script_path")
    else
        cmd_output=$(env             CUDA_DEVICES="$CUDA_DEVICES"             OUTPUT_DIR_BASE_OVERRIDE="$output_dir_base"             RUN_NAME_OVERRIDE="$run_name"             TAIL_LOG="$TAIL_LOG"             SAVE_VIDEO_OVERRIDE=false             bash "$script_path")
    fi

    local train_pid
    local log_file
    train_pid=$(printf "%s
" "$cmd_output" | awk -F= '/^TRAIN_PID=/{print $2}')
    log_file=$(printf "%s
" "$cmd_output" | awk -F= '/^LOG_FILE=/{print $2}')
    local run_dir="${output_dir_base}/${run_name}"

    append_run "json_metrics" "$family" "${train_pid:-0}" "$run_dir" "$log_file"
    vlaselect_register_cleanup_pid "${train_pid:-0}"
    if [[ "$TAIL_LOG" == "1" ]]; then
        vlaselect_start_file_log_tail "$log_file" "$family"
    fi
    printf "[launch] family=%s pid=%s run_dir=%s
" "$family" "${train_pid:-unknown}" "$run_dir"
}

launch_octo() {
    local suite_name="vla-applicability-${STAMP}"
    local suite_dir="train/octo/ours_single_agent/workload_verify/results/${suite_name}"
    local log_file="${suite_dir}/launcher.log"
    mkdir -p "$suite_dir"

    if [ "$SHORT_TRAIN" = "1" ]; then
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" nohup             python -u -m train.octo.ours_single_agent.workload_verify.run_suite             --suite-name "$suite_name"             --gpu-ids "$CUDA_DEVICES"             --limit 1             --poll-seconds 5             --env-change-minute 0.25             --max-time-minutes 5             > "$log_file" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" nohup             python -u -m train.octo.ours_single_agent.workload_verify.run_suite             --suite-name "$suite_name"             --gpu-ids "$CUDA_DEVICES"             --limit 1             --poll-seconds 20             > "$log_file" 2>&1 &
    fi

    local train_pid=$!
    append_run "octo_suite" "octo" "$train_pid" "$suite_dir" "$log_file"
    vlaselect_register_cleanup_pid "$train_pid"
    if [[ "$TAIL_LOG" == "1" ]]; then
        vlaselect_start_file_log_tail "$log_file" "octo"
    fi
    printf "[launch] family=%s pid=%s run_dir=%s
" "octo" "$train_pid" "$suite_dir"
}

if [[ "${SHOULD_RUN_MODEL[octo]}" == "1" ]]; then
    launch_octo
fi
if [[ "${SHOULD_RUN_MODEL[vla_adapter_new]}" == "1" ]]; then
    launch_family "vla_adapter_new" "train/vla_adapter_new/ours/run_online_rl_cl.sh" "train/vla_adapter_new/ours/outputs/vla_applicability/${STAMP}" "run"
fi
if [[ "${SHOULD_RUN_MODEL[tinyvla]}" == "1" ]]; then
    launch_family "tinyvla" "train/tinyvla/ours/run_online_rl_cl.sh" "train/tinyvla/ours/outputs/vla_applicability/${STAMP}" "run"
fi
if [[ "${SHOULD_RUN_MODEL[edgevla]}" == "1" ]]; then
    launch_family "edgevla" "train/edgevla/ours/run_online_rl_cl.sh" "train/edgevla/ours/outputs/vla_applicability/${STAMP}" "run"
fi

RUNS_SERIALIZED=$(printf "%s
" "${RUNS[@]}")
RUNS_SERIALIZED="$RUNS_SERIALIZED" MANIFEST_PATH="$MANIFEST_PATH" STAMP="$STAMP" SHORT_TRAIN="$SHORT_TRAIN" python - <<'PY_MANIFEST'
import json
import os

runs = []
for line in os.environ.get("RUNS_SERIALIZED", "").splitlines():
    if not line:
        continue
    kind, family, pid, run_dir, log_file = line.split("|", 4)
    runs.append({
        "kind": kind,
        "family": family,
        "pid": int(pid) if pid else 0,
        "run_dir": run_dir,
        "log_file": log_file,
    })

manifest = {
    "stamp": os.environ["STAMP"],
    "short_train": int(os.environ["SHORT_TRAIN"]),
    "runs": runs,
}
with open(os.environ["MANIFEST_PATH"], "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("
")
PY_MANIFEST

printf "[summary] manifest=%s
" "$MANIFEST_PATH"
if [ "$WAIT_FOR_COMPLETION" = "1" ]; then
    python "$SCRIPT_DIR/summarize_vla_applicability.py" --manifest "$MANIFEST_PATH" --wait --poll-seconds 10
else
    python "$SCRIPT_DIR/summarize_vla_applicability.py" --manifest "$MANIFEST_PATH"
fi
