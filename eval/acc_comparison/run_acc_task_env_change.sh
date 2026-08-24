#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$EVAL_ROOT"
source "${EVAL_ROOT}/common/interrupt_cleanup.sh"
source "${EVAL_ROOT}/common/sanity_check.sh"
source "${EVAL_ROOT}/common/env_order.sh"
source "${EVAL_ROOT}/common/mwe_time.sh"

SUITE_STAMP="${SUITE_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}"
TABLE_ROOT="${TABLE_ROOT_OVERRIDE:-acc_comparison/acc_comparison_task_env_table}"
RUN_ROOT="${TABLE_ROOT}/${SUITE_STAMP}"
LAUNCH_LOG_DIR="${RUN_ROOT}/launch_logs"
PANELS_JSONL="${RUN_ROOT}/panels.jsonl"
MANIFEST_JSON="${RUN_ROOT}/manifest.json"
LATEST_POINTER="${TABLE_ROOT}/latest.txt"

TAIL_LOG="${TAIL_LOG:-1}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-30}"
PLOT_INTERVAL_SECONDS="${PLOT_INTERVAL_SECONDS:-60}"
SUITE_WAIT_POLL_SECONDS="${SUITE_WAIT_POLL_SECONDS:-30}"
EDGEVLA_QUEUED_PER_GPU="${EDGEVLA_QUEUED_PER_GPU:-1}"
EDGEVLA_SMOKE="${EDGEVLA_SMOKE:-0}"
ENABLE_SELF_CURVE_WATCHER="${ENABLE_SELF_CURVE_WATCHER:-0}"
MODEL_SELECTION="${MODEL_SELECTION:-}"
FAMILY_SELECTION="${FAMILY_SELECTION:-${MODEL_SELECTION:-}}"
MWE="${MWE:-0}"

: "${MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS:=300}"
export MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS
if [[ "$MWE" == "1" ]]; then
    EDGEVLA_SMOKE="1"
    MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-5}"
    PLOT_INTERVAL_SECONDS="${PLOT_INTERVAL_SECONDS:-5}"
    SUITE_WAIT_POLL_SECONDS="${SUITE_WAIT_POLL_SECONDS:-5}"
fi
vlaselect_install_cleanup_trap
vlaselect_run_sanity_check "run_acc_task_env_change.sh" "$EVAL_ROOT" "$MWE" "16" "8"

EDGEVLA_ENVS_ID="${EDGEVLA_ENVS_ID:-['UnitreeG1LiftCubeObjectScaleDown1p3-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeObjectPurple-v1','UnitreeG1LiftSphereLightStronger50-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectScaleDown1p1-v1','UnitreeG1LiftSphereObjectScaleDown1p3-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectPurple-v1']}"
TINYVLA_ENVS_ID="${TINYVLA_ENVS_ID:-['OpenCabinetDrawerCabinet1021Default-v1','OpenCabinetDrawerCabinet1016ScaleUp1p3-v1','OpenCabinetDrawerCabinet1027Default-v1','OpenCabinetDrawerCabinet1016ScaleUp1p3-v1','OpenCabinetDrawerCabinet1032Default-v1','OpenCabinetDrawerCabinet1033ScaleUp1p3-v1','OpenCabinetDrawerCabinet1027Default-v1','OpenCabinetDrawerCabinet1021Default-v1','OpenCabinetDrawerCabinet1032Default-v1','OpenCabinetDrawerCabinet1033ScaleUp1p3-v1']}"
VLA_ADAPTER_NEW_ENVS_ID="${VLA_ADAPTER_NEW_ENVS_ID:-['HoldHammerInHandObjectScaleDown1p6-v1','HoldWrenchInHandObjectScaleUp1p2-v1','HoldWoodBlockInHandObjectScaleDown1p6-v1','HoldHammerInHandObjectScaleUp1p6-v1','HoldHammerInHandObjectScaleDown1p4-v1','HoldWrenchInHandObjectScaleUp1p6-v1','HoldWrenchInHandObjectScaleUp1p4-v1','HoldHammerInHandObjectScaleDown1p2-v1','HoldHammerInHandObjectScaleUp1p4-v1','HoldWrenchInHandObjectScaleDown1p6-v1']}"
OCTO_ENVS_ID="${OCTO_ENVS_ID:-['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']}"
ENV_CHANGE_TIME_POINTS="${ENV_CHANGE_TIME_POINTS:-[31,62,96,131,151,163,207,247,271,300]}"

vlaselect_apply_env_id_order OCTO_ENVS_ID ENV_CHANGE_TIME_POINTS
vlaselect_apply_env_id_order VLA_ADAPTER_NEW_ENVS_ID ENV_CHANGE_TIME_POINTS
vlaselect_apply_env_id_order TINYVLA_ENVS_ID ENV_CHANGE_TIME_POINTS
vlaselect_apply_env_id_order EDGEVLA_ENVS_ID ENV_CHANGE_TIME_POINTS

EFFECTIVE_ENV_CHANGE_TIME_POINTS="$ENV_CHANGE_TIME_POINTS"
if [[ "$MWE" == "1" ]]; then
    EFFECTIVE_ENV_CHANGE_TIME_POINTS="$(vlaselect_convert_mwe_schedule_seconds_to_minutes "$ENV_CHANGE_TIME_POINTS")"
fi

declare -a PAPER_FAMILY_ORDER=(
    octo
    vla_adapter_new
    tinyvla
    edgevla
)

declare -A PANEL_LABEL_BY_FAMILY=(
    [octo]=a
    [vla_adapter_new]=b
    [tinyvla]=c
    [edgevla]=d
)

declare -A WORKLOAD_NAME_BY_FAMILY=(
    [octo]="Single-arm robot"
    [vla_adapter_new]="Dexterous hand"
    [tinyvla]="Mobile manipulator"
    [edgevla]="Humanoid robot"
)

declare -A DISPLAY_NAME_BY_FAMILY=(
    [octo]="Octo"
    [vla_adapter_new]="VLA-Adapter"
    [tinyvla]="TinyVLA"
    [edgevla]="EdgeVLA"
)

declare -A LAUNCH_SCRIPT_BY_FAMILY=(
    [octo]="train/octo/launch_cl_suite.sh"
    [vla_adapter_new]="train/vla_adapter_new/launch_cl_suite.sh"
    [tinyvla]="train/tinyvla/launch_cl_suite.sh"
    [edgevla]="train/edgevla/launch_cl_suite.sh"
)

declare -A SUITE_MANIFEST_BY_FAMILY=(
    [octo]="ckpt/cl_suite/${SUITE_STAMP}/manifest.json"
    [vla_adapter_new]="ckpt/vla_adapter_new/cl_suite/${SUITE_STAMP}/manifest.json"
    [tinyvla]="ckpt/tinyvla/cl_suite/${SUITE_STAMP}/manifest.json"
    [edgevla]="train/edgevla/cl_suite/${SUITE_STAMP}/manifest.json"
)

declare -A ENVS_ID_BY_FAMILY=(
    [octo]="$OCTO_ENVS_ID"
    [vla_adapter_new]="$VLA_ADAPTER_NEW_ENVS_ID"
    [tinyvla]="$TINYVLA_ENVS_ID"
    [edgevla]="$EDGEVLA_ENVS_ID"
)

select_families() {
    local raw_selection="$1"
    if [[ -z "$raw_selection" ]]; then
        printf "%s\n" "${PAPER_FAMILY_ORDER[@]}"
        return
    fi
    printf "%s" "$raw_selection" | tr ',' '\n' | awk 'NF {gsub(/^[ \t]+|[ \t]+$/, ""); print}'
}

mkdir -p "$RUN_ROOT" "$LAUNCH_LOG_DIR"
: > "$PANELS_JSONL"
printf "%s\n" "$SUITE_STAMP" > "$LATEST_POINTER"

declare -A SHOULD_RUN_FAMILY=(
    [octo]=0
    [vla_adapter_new]=0
    [tinyvla]=0
    [edgevla]=0
)

SELECTED_FAMILY_COUNT=0
while IFS= read -r family; do
    [[ -z "$family" ]] && continue
    if [[ -z "${SHOULD_RUN_FAMILY[$family]+x}" ]]; then
        echo "Unknown family in FAMILY_SELECTION: $family" >&2
        exit 1
    fi
    if [[ "${SHOULD_RUN_FAMILY[$family]}" == "0" ]]; then
        SHOULD_RUN_FAMILY["$family"]=1
        SELECTED_FAMILY_COUNT=$((SELECTED_FAMILY_COUNT + 1))
    fi
done < <(select_families "$FAMILY_SELECTION")

if [[ "$SELECTED_FAMILY_COUNT" -eq 0 ]]; then
    echo "No families selected." >&2
    exit 1
fi

log() {
    echo "[fig7] $*"
}

print_log_excerpt() {
    vlaselect_print_log_excerpt "$1" "${2:-20}" "fig7"
}

detect_host_cpu_total() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
        return
    fi
    if command -v getconf >/dev/null 2>&1; then
        getconf _NPROCESSORS_ONLN
        return
    fi
    echo 4
}

compute_cpu_thread_limit() {
    local cpu_total="$1"
    if [[ "$cpu_total" -le 4 ]]; then
        echo 1
    elif [[ "$cpu_total" -le 8 ]]; then
        echo 2
    elif [[ "$cpu_total" -le 16 ]]; then
        echo 4
    else
        echo 8
    fi
}

CPU_TOTAL="${CPU_TOTAL_OVERRIDE:-$(detect_host_cpu_total)}"
CPU_THROTTLE_ENABLED="${CPU_THROTTLE_ENABLED:-1}"
CPU_THREAD_LIMIT="${CPU_THREAD_LIMIT:-}"

if [[ "$CPU_THROTTLE_ENABLED" == "1" ]]; then
    if [[ -z "$CPU_THREAD_LIMIT" ]]; then
        CPU_THREAD_LIMIT="$(compute_cpu_thread_limit "$CPU_TOTAL")"
    fi
    if [[ ! "$CPU_THREAD_LIMIT" =~ ^[0-9]+$ ]] || [[ "$CPU_THREAD_LIMIT" -lt 1 ]]; then
        echo "[fig7] invalid CPU_THREAD_LIMIT: $CPU_THREAD_LIMIT" >&2
        exit 1
    fi

    export OMP_NUM_THREADS="$CPU_THREAD_LIMIT"
    export MKL_NUM_THREADS="$CPU_THREAD_LIMIT"
    export OPENBLAS_NUM_THREADS="$CPU_THREAD_LIMIT"
    export NUMEXPR_NUM_THREADS="$CPU_THREAD_LIMIT"
    export VECLIB_MAXIMUM_THREADS="$CPU_THREAD_LIMIT"
    export BLIS_NUM_THREADS="$CPU_THREAD_LIMIT"
    export TOKENIZERS_PARALLELISM=false

    log "detected ${CPU_TOTAL} CPU cores; applying thread cap ${CPU_THREAD_LIMIT} to downstream jobs"
    if [[ "$CPU_TOTAL" -le 8 && "$MWE" != "1" ]]; then
        log "low-core host detected; consider 'MWE=1 bash run_acc_task_env_change.sh' for a lighter validation run"
    fi
else
    log "CPU throttling disabled"
fi

run_launch_command() {
    local family="$1"
    local launch_log="$2"
    shift 2

    if [[ "$TAIL_LOG" == "1" ]]; then
        vlaselect_start_file_log_tail "$launch_log" "${family}-launch"
    fi

    set +e
    "$@" > "$launch_log" 2>&1
    local rc=$?
    set -e

    if [[ "$rc" -ne 0 ]]; then
        vlaselect_report_command_failure "fig7" "launch failed for ${family}" "$launch_log" "" "$rc"
        print_log_excerpt "$launch_log"
        return "$rc"
    fi

    log "launch command finished for ${family}; log: ${launch_log}"
}

refresh_top_manifest() {
    python - <<'PY' "$PANELS_JSONL" "$MANIFEST_JSON" "$SUITE_STAMP" "$TABLE_ROOT"
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
suite_stamp = sys.argv[3]
table_root = sys.argv[4]
panels = []
if jsonl_path.exists():
    for raw_line in jsonl_path.read_text(encoding='utf-8').splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        panels.append(json.loads(raw_line))
payload = {
    'suite_stamp': suite_stamp,
    'table_root': table_root,
    'figure_output': 'acc_comparison/FIG_ACC_TASK_ENV.pdf',
    'panels': panels,
    'families': panels,
}
manifest_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
PY
}

append_panel_entry() {
    local family="$1"
    local suite_manifest="$2"
    local launch_log="$3"
    python - <<'PY' "$PANELS_JSONL" "$family" "$suite_manifest" "$launch_log" "$SUITE_STAMP" "$ENV_CHANGE_TIME_POINTS" "${PANEL_LABEL_BY_FAMILY[$family]}" "${WORKLOAD_NAME_BY_FAMILY[$family]}" "${DISPLAY_NAME_BY_FAMILY[$family]}" "${ENVS_ID_BY_FAMILY[$family]}"
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
entry = {
    'family': sys.argv[2],
    'suite_manifest': sys.argv[3],
    'suite_root': str(Path(sys.argv[3]).parent),
    'launch_log': sys.argv[4],
    'suite_stamp': sys.argv[5],
    'env_change_time_points': sys.argv[6],
    'panel_label': sys.argv[7],
    'workload_name': sys.argv[8],
    'display_name': sys.argv[9],
    'envs_id': sys.argv[10],
}
with jsonl_path.open('a', encoding='utf-8') as handle:
    handle.write(json.dumps(entry, ensure_ascii=True) + '\n')
PY
    refresh_top_manifest
}

check_suite_state() {
    python - <<'PY' "$1"
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.exists() or manifest_path.stat().st_size == 0:
    print(json.dumps({'state': 'waiting_for_manifest', 'active_labels': []}))
    raise SystemExit(2)

manifest = json.loads(manifest_path.read_text(encoding='utf-8'))


def is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    proc_stat = Path(f'/proc/{pid}/stat')
    if proc_stat.exists():
        try:
            fields = proc_stat.read_text(encoding='utf-8').split()
        except OSError:
            fields = []
        if len(fields) >= 3 and fields[2] == 'Z':
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

active_labels = []
tracked = []
for key in ('scheduler_pid', 'plotter_pid'):
    pid = manifest.get(key)
    if pid is not None:
        tracked.append((key, pid))

for method in manifest.get('methods', []):
    for key in ('pid', 'monitor_pid'):
        pid = method.get(key)
        if pid is None:
            continue
        tracked.append((f"{method.get('name', 'unknown')}:{key}", pid))

for label, pid in tracked:
    if is_alive(pid):
        active_labels.append(f'{label}={pid}')

payload = {
    'state': 'running' if active_labels else 'finished',
    'active_labels': active_labels,
    'tracked_pid_count': len(tracked),
}
print(json.dumps(payload, ensure_ascii=True))
raise SystemExit(10 if active_labels else 0)
PY
}

wait_for_suite_completion() {
    local family="$1"
    local suite_manifest="$2"
    local launch_log="$3"
    local status=""

    log "waiting for ${family}: ${suite_manifest}"
    while true; do
        if check_suite_state "$suite_manifest" > /dev/null; then
            status="finished"
        else
            rc=$?
            if [[ "$rc" -eq 10 ]]; then
                status="running"
            elif [[ "$rc" -eq 2 ]]; then
                status="waiting_for_manifest"
            else
                vlaselect_report_command_failure "fig7" "failed to inspect suite state for ${family}: ${suite_manifest}" "$launch_log"
                print_log_excerpt "$launch_log"
                return "$rc"
            fi
        fi

        if [[ "$status" == "finished" ]]; then
            log "completed ${family}: ${suite_manifest}"
            return 0
        fi

        log "${family} still running"
        sleep "$SUITE_WAIT_POLL_SECONDS"
    done
}

launch_family_suite() {
    local family="$1"
    local launch_script="${LAUNCH_SCRIPT_BY_FAMILY[$family]}"
    local suite_manifest="${SUITE_MANIFEST_BY_FAMILY[$family]}"
    local launch_log="${LAUNCH_LOG_DIR}/${family}.log"
    local envs_id="${ENVS_ID_BY_FAMILY[$family]}"

    vlaselect_register_cleanup_manifest "$suite_manifest"

    log "launching ${family} (${WORKLOAD_NAME_BY_FAMILY[$family]})"
    log "launch log: ${launch_log}"

    case "$family" in
        octo)
            run_launch_command "$family" "$launch_log" \
                env \
                SUITE_STAMP="$SUITE_STAMP" \
                TAIL_LOG="$TAIL_LOG" \
                MONITOR_INTERVAL_SECONDS="$MONITOR_INTERVAL_SECONDS" \
                PLOT_INTERVAL_SECONDS="$PLOT_INTERVAL_SECONDS" \
                ENABLE_SELF_CURVE_WATCHER="$ENABLE_SELF_CURVE_WATCHER" \
                SMOKE="$MWE" \
                MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS="$MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS" \
                ENVS_ID_OVERRIDE="$envs_id" \
                ENV_CHANGE_TIME_POINTS_OVERRIDE="$EFFECTIVE_ENV_CHANGE_TIME_POINTS" \
                bash "$launch_script"
            ;;
        vla_adapter_new)
            run_launch_command "$family" "$launch_log" \
                env \
                SUITE_STAMP="$SUITE_STAMP" \
                TAIL_LOG="$TAIL_LOG" \
                MONITOR_INTERVAL_SECONDS="$MONITOR_INTERVAL_SECONDS" \
                PLOT_INTERVAL_SECONDS="$PLOT_INTERVAL_SECONDS" \
                SMOKE="$MWE" \
                MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS="$MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS" \
                ENVS_ID_OVERRIDE="$envs_id" \
                ENV_IDS_OVERRIDE="$envs_id" \
                ENV_CHANGE_TIME_POINTS_OVERRIDE="$EFFECTIVE_ENV_CHANGE_TIME_POINTS" \
                bash "$launch_script"
            ;;
        tinyvla)
            run_launch_command "$family" "$launch_log" \
                env \
                SUITE_STAMP="$SUITE_STAMP" \
                TAIL_LOG="$TAIL_LOG" \
                MONITOR_INTERVAL_SECONDS="$MONITOR_INTERVAL_SECONDS" \
                PLOT_INTERVAL_SECONDS="$PLOT_INTERVAL_SECONDS" \
                SMOKE="$MWE" \
                MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS="$MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS" \
                ENVS_ID_OVERRIDE="$envs_id" \
                ENV_IDS_OVERRIDE="$envs_id" \
                ENV_CHANGE_TIME_POINTS_OVERRIDE="$EFFECTIVE_ENV_CHANGE_TIME_POINTS" \
                bash "$launch_script"
            ;;
        edgevla)
            run_launch_command "$family" "$launch_log" \
                env \
                SUITE_STAMP="$SUITE_STAMP" \
                TAIL_LOG="$TAIL_LOG" \
                MONITOR_INTERVAL_SECONDS="$MONITOR_INTERVAL_SECONDS" \
                PLOT_INTERVAL_SECONDS="$PLOT_INTERVAL_SECONDS" \
                QUEUED_PER_GPU="$EDGEVLA_QUEUED_PER_GPU" \
                SMOKE="$EDGEVLA_SMOKE" \
                MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS="$MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS" \
                SUITE_ENVS_ID="$envs_id" \
                SUITE_ENV_CHANGE_TIME_POINTS="$EFFECTIVE_ENV_CHANGE_TIME_POINTS" \
                ENVS_ID_OVERRIDE="$envs_id" \
                ENV_IDS_OVERRIDE="$envs_id" \
                ENV_CHANGE_TIME_POINTS_OVERRIDE="$EFFECTIVE_ENV_CHANGE_TIME_POINTS" \
                bash "$launch_script"
            ;;
        *)
            echo "Unhandled family: $family" >&2
            return 1
            ;;
    esac

    if [[ ! -f "$suite_manifest" ]]; then
        vlaselect_report_command_failure "fig7" "suite manifest not found for ${family}: ${suite_manifest}" "$launch_log"
        print_log_excerpt "$launch_log"
        return 1
    fi

    vlaselect_print_suite_training_logs "$suite_manifest" "fig7" "$family"
    append_panel_entry "$family" "$suite_manifest" "$launch_log"
    if [[ "$TAIL_LOG" == "1" ]]; then
        vlaselect_start_manifest_log_tail "$suite_manifest" "$family"
    fi
    wait_for_suite_completion "$family" "$suite_manifest" "$launch_log" || return $?
    if ! vlaselect_report_manifest_failures "$suite_manifest" "fig7" "$family" "$launch_log"; then
        return 1
    fi
}

refresh_top_manifest

for family in "${PAPER_FAMILY_ORDER[@]}"; do
    if [[ "${SHOULD_RUN_FAMILY[$family]}" != "1" ]]; then
        continue
    fi
    launch_family_suite "$family"
done

refresh_top_manifest

log "Figure 7 suites finished: ${SUITE_STAMP}"
log "Top-level manifest: ${MANIFEST_JSON}"
log "Output table root: ${RUN_ROOT}"
log "Plot script: ${SCRIPT_DIR}/plot_acc_task_env.py"
log "Expected figure: ${SCRIPT_DIR}/FIG_ACC_TASK_ENV.pdf"
