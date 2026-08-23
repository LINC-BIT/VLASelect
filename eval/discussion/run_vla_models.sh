#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
EVAL_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "$EVAL_ROOT"
source "${EVAL_ROOT}/common/interrupt_cleanup.sh"
source "${EVAL_ROOT}/common/sanity_check.sh"

STAMP=${VLA_APPLICABILITY_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}
SHORT_TRAIN=${VLA_APPLICABILITY_SHORT_TRAIN:-${VLA_APPLICABILITY_SMOKE:-1}}
WAIT_FOR_COMPLETION=${VLA_APPLICABILITY_WAIT:-1}
MWE=${MWE:-0}
MODEL_SELECTION="${MODEL_SELECTION:-${MODEL_FAMILY:-}}"
TAIL_LOG=${TAIL_LOG:-1}
CUDA_DEVICES=${CUDA_DEVICES:-}
GPU_BY_MODEL_OVERRIDE="${GPU_BY_MODEL_OVERRIDE:-}"
GPU_QUEUE_POLL_SECONDS="${GPU_QUEUE_POLL_SECONDS:-30}"
MANIFEST_PATH="${SCRIPT_DIR}/vla_applicability_${STAMP}.json"

: "${MWE_TOTAL_RUNTIME_LIMIT_SECONDS:=300}"
export MWE_TOTAL_RUNTIME_LIMIT_SECONDS
if [[ "$MWE" == "1" ]]; then
    SHORT_TRAIN=1
    WAIT_FOR_COMPLETION=1
fi
vlaselect_install_cleanup_trap
vlaselect_run_sanity_check "run_vla_models.sh" "$EVAL_ROOT" "$MWE" "16" "8"
vlaselect_register_cleanup_manifest "$MANIFEST_PATH"

SHORT_TOTAL_TIMESTEPS=${VLA_APPLICABILITY_TOTAL_TIMESTEPS:-1024}
SHORT_NUM_ENVS=${VLA_APPLICABILITY_NUM_ENVS:-8}
SHORT_NUM_EVAL_ENVS=${VLA_APPLICABILITY_NUM_EVAL_ENVS:-2}
SHORT_NUM_STEPS=${VLA_APPLICABILITY_NUM_STEPS:-16}
SHORT_NUM_MINIBATCHES=${VLA_APPLICABILITY_NUM_MINIBATCHES:-2}
SHORT_UPDATE_EPOCHS=${VLA_APPLICABILITY_UPDATE_EPOCHS:-1}
SHORT_EVAL_EVERY_UPDATES=${VLA_APPLICABILITY_EVAL_EVERY_UPDATES:-1}
SHORT_EVAL_EPISODES=${VLA_APPLICABILITY_EVAL_EPISODES:-4}
SHORT_MAX_RUNTIME_HOURS=${VLA_APPLICABILITY_MAX_RUNTIME_HOURS:-0.0084}
SHORT_ROLLOUT_MICRO_BATCH_SIZE=${VLA_APPLICABILITY_ROLLOUT_MICRO_BATCH_SIZE:-8}
SHORT_EVAL_MICRO_BATCH_SIZE=${VLA_APPLICABILITY_EVAL_MICRO_BATCH_SIZE:-8}
SHORT_UPDATE_MICRO_BATCH_SIZE=${VLA_APPLICABILITY_UPDATE_MICRO_BATCH_SIZE:-4}
SHORT_TRAIN_VIDEO_NUM_ENVS=${VLA_APPLICABILITY_TRAIN_VIDEO_NUM_ENVS:-1}
SHORT_TEST_VIDEO_NUM_ENVS=${VLA_APPLICABILITY_TEST_VIDEO_NUM_ENVS:-1}
SHORT_TEST_VIDEO_EPISODES=${VLA_APPLICABILITY_TEST_VIDEO_EPISODES:-1}
SHORT_EARLY_STOP_ZERO_SUCCESS_MINUTES=${VLA_APPLICABILITY_EARLY_STOP_ZERO_SUCCESS_MINUTES:-20}

mkdir -p "$SCRIPT_DIR"

select_models() {
    local raw_selection="$1"
    if [[ -z "$raw_selection" ]]; then
        printf "%s\n" octo vla_adapter_new tinyvla edgevla
        return
    fi
    printf "%s" "$raw_selection" | tr ',' '\n' | awk 'NF {gsub(/^[ 	]+|[ 	]+$/, ""); print}'
}

resolve_model_gpu_map() {
    local requested_gpus=()
    local default_map=""
    local method_order=(octo vla_adapter_new tinyvla edgevla)

    if [[ -n "$CUDA_DEVICES" ]]; then
        while IFS= read -r gpu; do
            [[ -n "$gpu" ]] && requested_gpus+=("$gpu")
        done < <(printf '%s' "$CUDA_DEVICES" | tr ',' '\n' | awk 'NF {gsub(/^[ 	]+|[ 	]+$/, ""); print}')
    fi
    if [[ "${#requested_gpus[@]}" -eq 0 ]]; then
        requested_gpus=(0 1 2 3)
    fi

    local idx=0
    for method in "${method_order[@]}"; do
        local gpu="${requested_gpus[$((idx % ${#requested_gpus[@]}))]}"
        if [[ -n "$default_map" ]]; then
            default_map+=","
        fi
        default_map+="${method}=${gpu}"
        idx=$((idx + 1))
    done

    python3 -m train.common.gpu_auto_select resolve-method-map         --method-order "$(IFS=,; echo "${method_order[*]}")"         --default-map "$default_map"         --override-map "$GPU_BY_MODEL_OVERRIDE"
}

declare -a RUNS=()
declare -A SHOULD_RUN_MODEL=(
    [octo]=0
    [vla_adapter_new]=0
    [tinyvla]=0
    [edgevla]=0
)
declare -a SELECTED_MODELS=()
declare -A RESOLVED_GPU_BY_MODEL=()
declare -A LAST_PID_BY_GPU=()

while IFS= read -r family; do
    [[ -z "$family" ]] && continue
    if [[ -z "${SHOULD_RUN_MODEL[$family]+x}" ]]; then
        echo "[error] unknown model in MODEL_SELECTION: $family" >&2
        exit 1
    fi
    if [[ "${SHOULD_RUN_MODEL[$family]}" == "0" ]]; then
        SHOULD_RUN_MODEL["$family"]=1
        SELECTED_MODELS+=("$family")
    fi
done < <(select_models "$MODEL_SELECTION")

if [[ "${#SELECTED_MODELS[@]}" -eq 0 ]]; then
    echo "[error] No models selected." >&2
    exit 1
fi

if [[ "$MWE" == "1" ]]; then
    mwe_selected_model_count="${#SELECTED_MODELS[@]}"
    if [[ "$mwe_selected_model_count" -lt 1 ]]; then
        mwe_selected_model_count=1
    fi
    MWE_PER_MODEL_RUNTIME_SECONDS=$((MWE_TOTAL_RUNTIME_LIMIT_SECONDS / mwe_selected_model_count))
    if [[ "$MWE_PER_MODEL_RUNTIME_SECONDS" -lt 1 ]]; then
        MWE_PER_MODEL_RUNTIME_SECONDS=1
    fi
    SHORT_MAX_RUNTIME_HOURS="$(awk -v sec="$MWE_PER_MODEL_RUNTIME_SECONDS" 'BEGIN { printf "%.6f", sec / 3600 }')"
    OCTO_SHORT_MAX_TIME_MINUTES="$(awk -v sec="$MWE_PER_MODEL_RUNTIME_SECONDS" 'BEGIN { printf "%.6f", sec / 60 }')"
else
    MWE_PER_MODEL_RUNTIME_SECONDS=""
    OCTO_SHORT_MAX_TIME_MINUTES="5"
fi

while IFS=$'	' read -r family gpu; do
    [[ -z "$family" ]] && continue
    RESOLVED_GPU_BY_MODEL["$family"]="$gpu"
done < <(resolve_model_gpu_map)

append_run() {
    local kind="$1"
    local family="$2"
    local pid="$3"
    local run_dir="$4"
    local log_file="$5"
    local gpu="$6"
    RUNS+=("${kind}|${family}|${pid}|${run_dir}|${log_file}|${gpu}")
}

wait_for_gpu_slot() {
    local gpu="$1"
    local wait_for_pid="${LAST_PID_BY_GPU[$gpu]:-}"
    if [[ -z "$wait_for_pid" ]]; then
        return
    fi
    echo "[queue] waiting for gpu=${gpu} pid=${wait_for_pid}"
    while kill -0 "$wait_for_pid" 2>/dev/null; do
        sleep "$GPU_QUEUE_POLL_SECONDS"
    done
}

launch_family() {
    local family="$1"
    local gpu="$2"
    local script_path="$3"
    local output_dir_base="$4"
    local run_name="$5"
    local cmd_output
    local train_pid
    local log_file
    local run_dir="${output_dir_base}/${run_name}"

    wait_for_gpu_slot "$gpu"

    if [[ "$SHORT_TRAIN" == "1" ]]; then
        cmd_output=$(env             CUDA_DEVICES="$gpu"             OUTPUT_DIR_BASE_OVERRIDE="$output_dir_base"             RUN_NAME_OVERRIDE="$run_name"             TAIL_LOG="$TAIL_LOG"             SAVE_VIDEO_OVERRIDE=false             NUM_ENVS_OVERRIDE="$SHORT_NUM_ENVS"             NUM_EVAL_ENVS_OVERRIDE="$SHORT_NUM_EVAL_ENVS"             NUM_STEPS_OVERRIDE="$SHORT_NUM_STEPS"             TOTAL_TIMESTEPS_OVERRIDE="$SHORT_TOTAL_TIMESTEPS"             NUM_MINIBATCHES_OVERRIDE="$SHORT_NUM_MINIBATCHES"             UPDATE_EPOCHS_OVERRIDE="$SHORT_UPDATE_EPOCHS"             EVAL_EVERY_UPDATES_OVERRIDE="$SHORT_EVAL_EVERY_UPDATES"             EVAL_EPISODES_OVERRIDE="$SHORT_EVAL_EPISODES"             MAX_RUNTIME_HOURS_OVERRIDE="$SHORT_MAX_RUNTIME_HOURS"             ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE="$SHORT_ROLLOUT_MICRO_BATCH_SIZE"             EVAL_MICRO_BATCH_SIZE_OVERRIDE="$SHORT_EVAL_MICRO_BATCH_SIZE"             UPDATE_MICRO_BATCH_SIZE_OVERRIDE="$SHORT_UPDATE_MICRO_BATCH_SIZE"             TRAIN_VIDEO_NUM_ENVS_OVERRIDE="$SHORT_TRAIN_VIDEO_NUM_ENVS"             TEST_VIDEO_NUM_ENVS_OVERRIDE="$SHORT_TEST_VIDEO_NUM_ENVS"             TEST_VIDEO_EPISODES_OVERRIDE="$SHORT_TEST_VIDEO_EPISODES"             EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE="$SHORT_EARLY_STOP_ZERO_SUCCESS_MINUTES"             RUN_SETUP_SMOKE_OVERRIDE=false             bash "$script_path")
    else
        cmd_output=$(env             CUDA_DEVICES="$gpu"             OUTPUT_DIR_BASE_OVERRIDE="$output_dir_base"             RUN_NAME_OVERRIDE="$run_name"             TAIL_LOG="$TAIL_LOG"             SAVE_VIDEO_OVERRIDE=false             bash "$script_path")
    fi

    train_pid=$(printf '%s\n' "$cmd_output" | awk -F= '/^TRAIN_PID=/{print $2}')
    log_file=$(printf '%s\n' "$cmd_output" | awk -F= '/^LOG_FILE=/{print $2}')
    if [[ -z "$train_pid" ]]; then
        echo "[error] failed to launch ${family}; launcher output did not contain TRAIN_PID" >&2
        printf '%s\n' "$cmd_output" >&2
        return 1
    fi

    LAST_PID_BY_GPU["$gpu"]="$train_pid"
    append_run "json_metrics" "$family" "$train_pid" "$run_dir" "$log_file" "$gpu"
    vlaselect_register_cleanup_pid "$train_pid"
    if [[ "$TAIL_LOG" == "1" && -n "$log_file" ]]; then
        vlaselect_start_file_log_tail "$log_file" "$family"
    fi
    printf '[launch] family=%s gpu=%s pid=%s run_dir=%s\n' "$family" "$gpu" "$train_pid" "$run_dir"
}

launch_octo() {
    local gpu="$1"
    local suite_name="vla-applicability-${STAMP}"
    local suite_dir="train/octo/ours_single_agent/workload_verify/results/${suite_name}"
    local log_file="${suite_dir}/launcher.log"

    wait_for_gpu_slot "$gpu"
    mkdir -p "$suite_dir"

    if [[ "$SHORT_TRAIN" == "1" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" nohup             python -u -m train.octo.ours_single_agent.workload_verify.run_suite             --suite-name "$suite_name"             --gpu-ids "$gpu"             --limit 1             --poll-seconds 5             --env-change-minute 0.25             --max-time-minutes "$OCTO_SHORT_MAX_TIME_MINUTES"             > "$log_file" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$gpu" nohup             python -u -m train.octo.ours_single_agent.workload_verify.run_suite             --suite-name "$suite_name"             --gpu-ids "$gpu"             --limit 1             --poll-seconds 20             > "$log_file" 2>&1 &
    fi

    local train_pid=$!
    LAST_PID_BY_GPU["$gpu"]="$train_pid"
    append_run "octo_suite" "octo" "$train_pid" "$suite_dir" "$log_file" "$gpu"
    vlaselect_register_cleanup_pid "$train_pid"
    if [[ "$TAIL_LOG" == "1" ]]; then
        vlaselect_start_file_log_tail "$log_file" "octo"
    fi
    printf '[launch] family=%s gpu=%s pid=%s run_dir=%s\n' 'octo' "$gpu" "$train_pid" "$suite_dir"
}

for family in "${SELECTED_MODELS[@]}"; do
    gpu="${RESOLVED_GPU_BY_MODEL[$family]}"
    case "$family" in
        octo)
            launch_octo "$gpu"
            ;;
        vla_adapter_new)
            launch_family "$family" "$gpu" 'train/vla_adapter_new/ours/run_online_rl_cl.sh' "train/vla_adapter_new/ours/outputs/vla_applicability/${STAMP}" 'run'
            ;;
        tinyvla)
            launch_family "$family" "$gpu" 'train/tinyvla/ours/run_online_rl_cl.sh' "train/tinyvla/ours/outputs/vla_applicability/${STAMP}" 'run'
            ;;
        edgevla)
            launch_family "$family" "$gpu" 'train/edgevla/ours/run_online_rl_cl.sh' "train/edgevla/ours/outputs/vla_applicability/${STAMP}" 'run'
            ;;
    esac
done

RUNS_SERIALIZED=$(printf '%s\n' "${RUNS[@]}")
RUNS_SERIALIZED="$RUNS_SERIALIZED" MANIFEST_PATH="$MANIFEST_PATH" STAMP="$STAMP" SHORT_TRAIN="$SHORT_TRAIN" python - <<'PY_MANIFEST'
import json
import os

runs = []
for line in os.environ.get('RUNS_SERIALIZED', '').splitlines():
    if not line:
        continue
    kind, family, pid, run_dir, log_file, gpu = line.split('|', 5)
    runs.append({
        'kind': kind,
        'family': family,
        'pid': int(pid) if pid else 0,
        'run_dir': run_dir,
        'log_file': log_file,
        'gpu': int(gpu) if gpu else None,
    })

manifest = {
    'stamp': os.environ['STAMP'],
    'short_train': int(os.environ['SHORT_TRAIN']),
    'runs': runs,
}
with open(os.environ['MANIFEST_PATH'], 'w', encoding='utf-8') as handle:
    json.dump(manifest, handle, indent=2)
    handle.write(chr(10))
PY_MANIFEST

printf '[summary] manifest=%s\n' "$MANIFEST_PATH"
if [[ "$WAIT_FOR_COMPLETION" == '1' ]]; then
    python "$SCRIPT_DIR/summarize_vla_applicability.py" --manifest "$MANIFEST_PATH" --wait --poll-seconds 10
else
    python "$SCRIPT_DIR/summarize_vla_applicability.py" --manifest "$MANIFEST_PATH"
fi
