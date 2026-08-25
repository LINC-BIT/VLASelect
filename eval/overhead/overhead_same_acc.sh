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
source "${EVAL_ROOT}/common/resource_summary.sh"

SUITE_STAMP="${SUITE_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}" 
TABLE_ROOT="${TABLE_ROOT_OVERRIDE:-overhead/overhead_same_acc_table}"
RUN_ROOT="${TABLE_ROOT}/${SUITE_STAMP}"
LAUNCH_LOG_DIR="${RUN_ROOT}/launch_logs"
PANELS_JSONL="${RUN_ROOT}/panels.jsonl"
MANIFEST_JSON="${RUN_ROOT}/manifest.json"
LATEST_POINTER="${TABLE_ROOT}/latest.txt"
ACC_COMPAT_TABLE_ROOT="${ACC_COMPAT_TABLE_ROOT_OVERRIDE:-acc_comparison/acc_task_env_from_same_acc_table}"
ACC_COMPAT_RUN_ROOT="${ACC_COMPAT_TABLE_ROOT}/${SUITE_STAMP}"
ACC_COMPAT_MANIFEST_JSON="${ACC_COMPAT_RUN_ROOT}/manifest.json"
ACC_COMPAT_LATEST_POINTER="${ACC_COMPAT_TABLE_ROOT}/latest.txt"

TAIL_LOG="${TAIL_LOG:-1}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-30}"
PLOT_INTERVAL_SECONDS="${PLOT_INTERVAL_SECONDS:-60}"
SUITE_WAIT_POLL_SECONDS="${SUITE_WAIT_POLL_SECONDS:-30}"
EDGEVLA_QUEUED_PER_GPU="${EDGEVLA_QUEUED_PER_GPU:-1}"
EDGEVLA_SMOKE="${EDGEVLA_SMOKE:-0}"
ENABLE_SELF_CURVE_WATCHER="${ENABLE_SELF_CURVE_WATCHER:-0}"
MODEL_SELECTION="${MODEL_SELECTION:-}"
FAMILY_SELECTION="${FAMILY_SELECTION:-${MODEL_SELECTION:-}}"
METHODS="${METHODS:-${RUN_METHODS:-}}"
MWE="${MWE:-0}"
BASELINE_PRETRAIN_CKPT_NOISE_SCALE="${BASELINE_PRETRAIN_CKPT_NOISE_SCALE:-${CKPT_NOISE_SCALE:-}}"
BASELINE_PRETRAIN_CKPT_NOISE_SEED="${BASELINE_PRETRAIN_CKPT_NOISE_SEED:-${CKPT_NOISE_SEED:-0}}"
SAME_ACC_BREAKDOWN_COMPAT="${SAME_ACC_BREAKDOWN_COMPAT:-1}"
SAME_ACC_ACCURACY_COMPAT="${SAME_ACC_ACCURACY_COMPAT:-1}"
SAME_ACC_METHOD_ACTIVE_RUNTIME_SECONDS="${SAME_ACC_METHOD_ACTIVE_RUNTIME_SECONDS:-60}"
ACC_COMPAT_FIGURE_STEM="${ACC_COMPAT_FIGURE_STEM:-FIG_ACC_TASK_ENV_FROM_SAME_ACC}"
ACC_COMPAT_SUMMARY_STEM="${ACC_COMPAT_SUMMARY_STEM:-acc_task_env_from_same_acc_summary}"
ACC_COMPAT_PANEL_DIR="${ACC_COMPAT_PANEL_DIR:-${ACC_COMPAT_FIGURE_STEM}_panels}"
ACC_COMPAT_VIS_PAYLOAD_DIR="${ACC_COMPAT_VIS_PAYLOAD_DIR:-vis_payload_task_env_from_same_acc}"

if [[ -n "$BASELINE_PRETRAIN_CKPT_NOISE_SCALE" && -z "${VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE:-}" ]]; then
    export VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE="$BASELINE_PRETRAIN_CKPT_NOISE_SCALE"
fi
if [[ -n "$BASELINE_PRETRAIN_CKPT_NOISE_SEED" && -z "${VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SEED:-}" ]]; then
    export VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SEED="$BASELINE_PRETRAIN_CKPT_NOISE_SEED"
fi
if [[ -n "${VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE:-}" && "${VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE}" != "0" && "${VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE}" != "0.0" ]]; then
    echo "[fig9] baseline pretrained checkpoint noise scale=${VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE} seed=${VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SEED:-0}"
fi

if [[ -z "${MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS+x}" ]]; then
    USER_SET_MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS=0
    MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS=$((SAME_ACC_METHOD_ACTIVE_RUNTIME_SECONDS * 10))
else
    USER_SET_MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS=1
    : "${MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS:=300}"
fi
export MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS
if [[ "$MWE" == "1" ]]; then
    EDGEVLA_SMOKE="1"
    MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-5}"
    PLOT_INTERVAL_SECONDS="${PLOT_INTERVAL_SECONDS:-5}"
    SUITE_WAIT_POLL_SECONDS="${SUITE_WAIT_POLL_SECONDS:-5}"
fi
vlaselect_resource_summary_start "overhead_same_acc.sh"
vlaselect_install_cleanup_trap
vlaselect_run_sanity_check "overhead_same_acc.sh" "$EVAL_ROOT" "$MWE" "16" "8"

EDGEVLA_ENVS_ID="${EDGEVLA_ENVS_ID:-['UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeObjectScaleDown1p3-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeObjectPurple-v1','UnitreeG1LiftSphereLightStronger50-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectScaleDown1p1-v1','UnitreeG1LiftSphereObjectScaleDown1p3-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectPurple-v1']}"
TINYVLA_ENVS_ID="${TINYVLA_ENVS_ID:-['OpenCabinetDrawerCabinet1027Default-v1','OpenCabinetDrawerCabinet1021Default-v1','OpenCabinetDrawerCabinet1016ScaleUp1p3-v1','OpenCabinetDrawerCabinet1016ScaleUp1p3-v1','OpenCabinetDrawerCabinet1032Default-v1','OpenCabinetDrawerCabinet1033ScaleUp1p3-v1','OpenCabinetDrawerCabinet1027Default-v1','OpenCabinetDrawerCabinet1021Default-v1','OpenCabinetDrawerCabinet1032Default-v1','OpenCabinetDrawerCabinet1033ScaleUp1p3-v1']}"
VLA_ADAPTER_NEW_ENVS_ID="${VLA_ADAPTER_NEW_ENVS_ID:-['HoldHammerInHandObjectScaleDown1p6-v1','HoldWrenchInHandObjectScaleUp1p2-v1','HoldWoodBlockInHandObjectScaleDown1p6-v1','HoldHammerInHandObjectScaleUp1p6-v1','HoldHammerInHandObjectScaleDown1p4-v1','HoldWrenchInHandObjectScaleUp1p6-v1','HoldWrenchInHandObjectScaleUp1p4-v1','HoldHammerInHandObjectScaleDown1p2-v1','HoldHammerInHandObjectScaleUp1p4-v1','HoldWrenchInHandObjectScaleDown1p6-v1']}"
OCTO_ENVS_ID="${OCTO_ENVS_ID:-['PickCubeColorTempHigher50-v1','PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeObjectScaleDown1p2-v1']}"
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

resolve_methods_for_family() {
    local family="$1"
    local raw_selection="$2"
    if [[ -z "$raw_selection" ]]; then
        return 0
    fi
    python - <<'PY' "$family" "$raw_selection"
import sys

family = sys.argv[1]
raw = sys.argv[2]
items = [item.strip() for item in raw.split(',') if item.strip()]
resolved = []
for item in items:
    if item == 'vlaselect':
        resolved.append('ours_single_agent' if family == 'octo' else 'ours')
    elif item == 'ours' and family == 'octo':
        resolved.append('ours_single_agent')
    elif item == 'ours_single_agent' and family != 'octo':
        resolved.append('ours')
    else:
        resolved.append(item)
print(','.join(resolved))
PY
}

count_selected_methods() {
    local raw_selection="$1"
    if [[ -z "$raw_selection" ]]; then
        echo 10
        return
    fi
    printf '%s' "$raw_selection" | tr ',' '\n' | awk 'NF {c++} END { if (c < 1) c = 1; print c }'
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

refresh_top_manifest() {
    python - <<'PY' "$PANELS_JSONL" "$MANIFEST_JSON" "$SUITE_STAMP" "$TABLE_ROOT" "$MWE" "$MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS"
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
suite_stamp = sys.argv[3]
table_root = sys.argv[4]
mwe = sys.argv[5]
mwe_workload_runtime_limit_seconds = sys.argv[6]
panels = []
if jsonl_path.exists():
    for raw_line in jsonl_path.read_text(encoding='utf-8').splitlines():
        raw_line = raw_line.strip()
        if raw_line:
            panels.append(json.loads(raw_line))
payload = {
    'suite_stamp': suite_stamp,
    'table_root': table_root,
    'figure_output': 'overhead/FIG_MEMORY_FOOTPOINT.pdf',
    'table2_output': 'overhead/overhead_breakdown_table/TAB_OVERHEAD.csv',
    'table3_output': 'overhead/overhead_breakdown_table/TAB_ENERGY.csv',
    'mwe': mwe,
    'mwe_workload_runtime_limit_seconds': mwe_workload_runtime_limit_seconds,
    'panels': panels,
    'families': panels,
}
manifest_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
PY
}

refresh_accuracy_compat_manifest() {
    mkdir -p "$ACC_COMPAT_RUN_ROOT"
    python - <<'PY' "$PANELS_JSONL" "$ACC_COMPAT_MANIFEST_JSON" "$SUITE_STAMP" "$ACC_COMPAT_TABLE_ROOT" "$ACC_COMPAT_FIGURE_STEM"
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
suite_stamp = sys.argv[3]
table_root = sys.argv[4]
figure_stem = sys.argv[5]
panels = []
if jsonl_path.exists():
    for raw_line in jsonl_path.read_text(encoding='utf-8').splitlines():
        raw_line = raw_line.strip()
        if raw_line:
            panels.append(json.loads(raw_line))
payload = {
    'suite_stamp': suite_stamp,
    'table_root': table_root,
    'figure_output': f'acc_comparison/{figure_stem}.pdf',
    'panels': panels,
    'families': panels,
}
manifest_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
PY
    printf "%s\n" "$SUITE_STAMP" > "$ACC_COMPAT_LATEST_POINTER"
}

run_launch_command() {
    local family="$1"
    local launch_log="$2"
    shift 2

    set +e
    "$@" > "$launch_log" 2>&1
    local rc=$?
    set -e

    if [[ "$rc" -ne 0 ]]; then
        vlaselect_report_command_failure "fig9" "launch failed for ${family}" "$launch_log" "" "$rc"
        return "$rc"
    fi
}

append_panel_entry() {
    local family="$1"
    local suite_manifest="$2"
    local launch_log="$3"
    local family_runtime_limit_seconds="$4"
    python - <<'PY' "$PANELS_JSONL" "$family" "$suite_manifest" "$launch_log" "$SUITE_STAMP" "$ENV_CHANGE_TIME_POINTS" "${PANEL_LABEL_BY_FAMILY[$family]}" "${WORKLOAD_NAME_BY_FAMILY[$family]}" "${DISPLAY_NAME_BY_FAMILY[$family]}" "${ENVS_ID_BY_FAMILY[$family]}" "$MWE" "$family_runtime_limit_seconds"
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
    'mwe': sys.argv[11],
    'mwe_workload_runtime_limit_seconds': sys.argv[12],
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
    raise SystemExit(2)
try:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
except (OSError, json.JSONDecodeError):
    raise SystemExit(2)

if manifest.get('suite_state') == 'finished':
    raise SystemExit(0)

def is_alive(pid):
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

tracked = []
for key in ('scheduler_pid', 'plotter_pid', 'watcher_pid'):
    pid = manifest.get(key)
    if pid is not None:
        tracked.append(pid)
for method in manifest.get('methods', []):
    for key in ('pid', 'monitor_pid'):
        pid = method.get(key)
        if pid is not None:
            tracked.append(pid)
raise SystemExit(10 if any(is_alive(pid) for pid in tracked) else 0)
PY
}

wait_for_suite_completion() {
    local family="$1"
    local suite_manifest="$2"
    local launch_log="$3"

    echo "[fig9] waiting for ${family}: ${suite_manifest}"
    while true; do
        check_suite_state "$suite_manifest"
        rc=$?
        if [[ "$rc" -eq 0 ]]; then
            echo "[fig9] completed ${family}: ${suite_manifest}"
            return 0
        fi
        if [[ "$rc" -ne 10 && "$rc" -ne 2 ]]; then
            vlaselect_report_command_failure "fig9" "failed to inspect suite state for ${family}: ${suite_manifest}" "$launch_log"
            return "$rc"
        fi
        echo "[fig9] ${family} still running"
        sleep "$SUITE_WAIT_POLL_SECONDS"
    done
}

launch_family_suite() {
    local family="$1"
    local launch_script="${LAUNCH_SCRIPT_BY_FAMILY[$family]}"
    local suite_manifest="${SUITE_MANIFEST_BY_FAMILY[$family]}"
    local launch_log="${LAUNCH_LOG_DIR}/${family}.log"
    local envs_id="${ENVS_ID_BY_FAMILY[$family]}"
    local family_methods=""
    local family_mwe_workload_runtime_limit_seconds="$MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS"
    local selected_method_count=10

    family_methods="$(resolve_methods_for_family "$family" "$METHODS")"
    if [[ "$MWE" == "1" && "$USER_SET_MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS" != "1" ]]; then
        selected_method_count="$(count_selected_methods "$family_methods")"
        family_mwe_workload_runtime_limit_seconds=$((selected_method_count * SAME_ACC_METHOD_ACTIVE_RUNTIME_SECONDS))
    fi

    vlaselect_register_cleanup_manifest "$suite_manifest"

    echo "[fig9] launching ${family} (${WORKLOAD_NAME_BY_FAMILY[$family]})"

    case "$family" in
        octo)
            run_launch_command "$family" "$launch_log" \
                env \
                SUITE_STAMP="$SUITE_STAMP" \
                TAIL_LOG="$TAIL_LOG" \
                MONITOR_INTERVAL_SECONDS="$MONITOR_INTERVAL_SECONDS" \
                PLOT_INTERVAL_SECONDS="$PLOT_INTERVAL_SECONDS" \
                ENABLE_SELF_CURVE_WATCHER="$ENABLE_SELF_CURVE_WATCHER" \
                METHODS="$family_methods" \
                SMOKE="$MWE" \
                MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS="$family_mwe_workload_runtime_limit_seconds" \
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
                METHODS="$family_methods" \
                SMOKE="$MWE" \
                MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS="$family_mwe_workload_runtime_limit_seconds" \
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
                METHODS="$family_methods" \
                SMOKE="$MWE" \
                MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS="$family_mwe_workload_runtime_limit_seconds" \
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
                METHODS="$family_methods" \
                SMOKE="$EDGEVLA_SMOKE" \
                MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS="$family_mwe_workload_runtime_limit_seconds" \
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
        vlaselect_report_command_failure "fig9" "suite manifest not found after launch: ${suite_manifest}" "$launch_log"
        return 1
    fi

    vlaselect_print_suite_training_logs "$suite_manifest" "fig9" "$family"
    append_panel_entry "$family" "$suite_manifest" "$launch_log" "$family_mwe_workload_runtime_limit_seconds"
    if [[ "$TAIL_LOG" == "1" ]]; then
        vlaselect_start_manifest_log_tail "$suite_manifest" "$family"
    fi
    wait_for_suite_completion "$family" "$suite_manifest" "$launch_log" || return $?
    if ! vlaselect_report_manifest_failures "$suite_manifest" "fig9" "$family" "$launch_log"; then
        return 1
    fi
}

refresh_top_manifest
for family in "${PAPER_FAMILY_ORDER[@]}"; do
    if [[ "${SHOULD_RUN_FAMILY[$family]}" == "1" ]]; then
        launch_family_suite "$family" || exit $?
    fi
done
refresh_top_manifest

echo "[fig9] manifest saved to ${MANIFEST_JSON}"
echo "[fig9] latest pointer updated at ${LATEST_POINTER}"
echo "[fig9] plotting will write figure: ${SCRIPT_DIR}/FIG_MEMORY_FOOTPOINT.pdf"
echo "[fig9] plotting will write figure: ${SCRIPT_DIR}/FIG_MEMORY_FOOTPOINT.png"
echo "[fig9] plotting will write figure: ${SCRIPT_DIR}/FIG_MEMORY_FOOTPOINT.svg"
echo "[fig9] plotting will write table: ${SCRIPT_DIR}/overhead_breakdown_table/TAB_OVERHEAD.csv"
echo "[fig9] plotting will write table: ${SCRIPT_DIR}/overhead_breakdown_table/TAB_ENERGY.csv"

if [[ "$SAME_ACC_ACCURACY_COMPAT" == "1" ]]; then
    refresh_accuracy_compat_manifest
    env \
        PLOT_ACC_TABLE_ROOT="$ACC_COMPAT_TABLE_ROOT" \
        PLOT_ACC_FIGURE_STEM="$ACC_COMPAT_FIGURE_STEM" \
        PLOT_ACC_SUMMARY_STEM="$ACC_COMPAT_SUMMARY_STEM" \
        PLOT_ACC_PANEL_DIR="$ACC_COMPAT_PANEL_DIR" \
        PLOT_ACC_VIS_PAYLOAD_DIR="$ACC_COMPAT_VIS_PAYLOAD_DIR" \
        python acc_comparison/plot_acc_task_env.py
    echo "[fig9-compat] accuracy manifest saved to ${ACC_COMPAT_MANIFEST_JSON}"
    echo "[fig9-compat] accuracy latest pointer updated at ${ACC_COMPAT_LATEST_POINTER}"
    echo "[fig9-compat] wrote figure: ${EVAL_ROOT}/acc_comparison/${ACC_COMPAT_FIGURE_STEM}.pdf"
    echo "[fig9-compat] wrote figure: ${EVAL_ROOT}/acc_comparison/${ACC_COMPAT_FIGURE_STEM}.png"
    echo "[fig9-compat] wrote figure: ${EVAL_ROOT}/acc_comparison/${ACC_COMPAT_FIGURE_STEM}.svg"
    echo "[fig9-compat] wrote summary: ${EVAL_ROOT}/acc_comparison/${ACC_COMPAT_SUMMARY_STEM}.csv"
fi

if [[ "$SAME_ACC_BREAKDOWN_COMPAT" == "1" ]]; then
    python overhead/plot_breakdown_impl.py --manifest "$MANIFEST_JSON" --prepare-only
    echo "[fig9-compat] wrote: ${RUN_ROOT}/BREAKDOWN_ALL_METHODS.csv"
    echo "[fig9-compat] wrote: ${RUN_ROOT}/BREAKDOWN_MODULES.csv"
    echo "[fig9-compat] wrote: ${RUN_ROOT}/breakdown_summary.json"
fi
