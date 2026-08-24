#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$EVAL_ROOT"
source "${EVAL_ROOT}/common/interrupt_cleanup.sh"
source "${EVAL_ROOT}/common/sanity_check.sh"
source "${EVAL_ROOT}/common/env_order.sh"

SUITE_STAMP="${SUITE_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}"
TABLE_ROOT="${TABLE_ROOT_OVERRIDE:-overhead/overhead_breakdown_modules_table}"
RUN_ROOT="${TABLE_ROOT}/${SUITE_STAMP}"
LAUNCH_LOG_DIR="${RUN_ROOT}/launch_logs"
PANELS_JSONL="${RUN_ROOT}/panels.jsonl"
MANIFEST_JSON="${RUN_ROOT}/manifest.json"
LATEST_POINTER="${TABLE_ROOT}/latest.txt"
COLLECT_ONLY="${COLLECT_ONLY:-0}"
SUITE_WAIT_POLL_SECONDS="${SUITE_WAIT_POLL_SECONDS:-30}"
TAIL_LOG="${TAIL_LOG:-1}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-30}"
PLOT_INTERVAL_SECONDS="${PLOT_INTERVAL_SECONDS:-60}"
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
vlaselect_run_sanity_check "overhead_breakdown_modules.sh" "$EVAL_ROOT" "$MWE" "50" "16"

EDGEVLA_ENVS_ID="${EDGEVLA_ENVS_ID:-['UnitreeG1LiftCubeObjectScaleDown1p3-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeObjectPurple-v1','UnitreeG1LiftSphereLightStronger50-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectScaleDown1p1-v1','UnitreeG1LiftSphereObjectScaleDown1p3-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectPurple-v1']}"
TINYVLA_ENVS_ID="${TINYVLA_ENVS_ID:-['OpenCabinetDrawerCabinet1021Default-v1','OpenCabinetDrawerCabinet1016ScaleUp1p3-v1','OpenCabinetDrawerCabinet1027Default-v1','OpenCabinetDrawerCabinet1016ScaleUp1p3-v1','OpenCabinetDrawerCabinet1032Default-v1','OpenCabinetDrawerCabinet1033ScaleUp1p3-v1','OpenCabinetDrawerCabinet1027Default-v1','OpenCabinetDrawerCabinet1021Default-v1','OpenCabinetDrawerCabinet1032Default-v1','OpenCabinetDrawerCabinet1033ScaleUp1p3-v1']}"
VLA_ADAPTER_NEW_ENVS_ID="${VLA_ADAPTER_NEW_ENVS_ID:-['HoldHammerInHandObjectScaleDown1p6-v1','HoldWrenchInHandObjectScaleUp1p2-v1','HoldWoodBlockInHandObjectScaleDown1p6-v1','HoldHammerInHandObjectScaleUp1p6-v1','HoldHammerInHandObjectScaleDown1p4-v1','HoldWrenchInHandObjectScaleUp1p6-v1','HoldWrenchInHandObjectScaleUp1p4-v1','HoldHammerInHandObjectScaleDown1p2-v1','HoldHammerInHandObjectScaleUp1p4-v1','HoldWrenchInHandObjectScaleDown1p6-v1']}"
OCTO_ENVS_ID="${OCTO_ENVS_ID:-['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']}"
ENV_CHANGE_TIME_POINTS="${ENV_CHANGE_TIME_POINTS:-[31,62,96,131,151,163,207,247,271,300]}"

vlaselect_apply_env_id_order OCTO_ENVS_ID ENV_CHANGE_TIME_POINTS
vlaselect_apply_env_id_order VLA_ADAPTER_NEW_ENVS_ID ENV_CHANGE_TIME_POINTS
vlaselect_apply_env_id_order TINYVLA_ENVS_ID ENV_CHANGE_TIME_POINTS
vlaselect_apply_env_id_order EDGEVLA_ENVS_ID ENV_CHANGE_TIME_POINTS


declare -a FAMILY_ORDER=(
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
        printf "%s\n" "${FAMILY_ORDER[@]}"
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
    for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if raw_line:
            panels.append(json.loads(raw_line))
payload = {
    "suite_stamp": suite_stamp,
    "table_root": table_root,
    "figure_all_methods_output": "overhead/FIG_BREAKDOWN_ALL_METHODS.pdf",
    "figure_modules_output": "overhead/FIG_BREAKDOWN_MODULES.pdf",
    "all_methods_csv": f"{table_root}/{suite_stamp}/BREAKDOWN_ALL_METHODS.csv",
    "modules_csv": f"{table_root}/{suite_stamp}/BREAKDOWN_MODULES.csv",
    "summary_output": f"{table_root}/{suite_stamp}/breakdown_summary.json",
    "panels": panels,
    "families": panels,
}
manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

append_panel_entry() {
    local family="$1"
    local suite_manifest="$2"
    local launch_log="$3"
    python - <<'PY' "$PANELS_JSONL" "$family" "$suite_manifest" "$launch_log" "$SUITE_STAMP" "${PANEL_LABEL_BY_FAMILY[$family]}" "${WORKLOAD_NAME_BY_FAMILY[$family]}" "${DISPLAY_NAME_BY_FAMILY[$family]}"
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
suite_manifest = sys.argv[3]
entry = {
    "family": sys.argv[2],
    "suite_manifest": suite_manifest,
    "suite_root": str(Path(suite_manifest).parent) if suite_manifest else "",
    "launch_log": sys.argv[4],
    "suite_stamp": sys.argv[5],
    "panel_label": sys.argv[6],
    "workload_name": sys.argv[7],
    "display_name": sys.argv[8],
}
with jsonl_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(2)

if manifest.get("suite_state") == "finished":
    raise SystemExit(0)

def is_alive(pid):
    if pid is None:
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            fields = proc_stat.read_text(encoding="utf-8").split()
        except OSError:
            fields = []
        if len(fields) >= 3 and fields[2] == "Z":
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

tracked = []
for key in ("scheduler_pid", "plotter_pid", "watcher_pid"):
    pid = manifest.get(key)
    if pid is not None:
        tracked.append(pid)
for method in manifest.get("methods", []):
    for key in ("pid", "monitor_pid"):
        pid = method.get(key)
        if pid is not None:
            tracked.append(pid)
raise SystemExit(10 if any(is_alive(pid) for pid in tracked) else 0)
PY
}

wait_for_suite_completion() {
    local family="$1"
    local suite_manifest="$2"

    echo "[breakdown] waiting for ${family}: ${suite_manifest}"
    while true; do
        if check_suite_state "$suite_manifest"; then
            rc=0
        else
            rc=$?
        fi
        if [[ "$rc" -eq 0 ]]; then
            echo "[breakdown] completed ${family}: ${suite_manifest}"
            return 0
        fi
        if [[ "$rc" -ne 10 && "$rc" -ne 2 ]]; then
            echo "[breakdown] failed to inspect suite state for ${family}: ${suite_manifest}" >&2
            return "$rc"
        fi
        echo "[breakdown] ${family} still running"
        sleep "$SUITE_WAIT_POLL_SECONDS"
    done
}

launch_family_suite() {
    local family="$1"
    local launch_script="${LAUNCH_SCRIPT_BY_FAMILY[$family]}"
    local suite_manifest="${SUITE_MANIFEST_BY_FAMILY[$family]}"
    local launch_log="${LAUNCH_LOG_DIR}/${family}.log"
    local envs_id="${ENVS_ID_BY_FAMILY[$family]}"
    local reuse_var_name
    local reuse_manifest=""

    reuse_var_name="$(printf '%s' "${family^^}_SUITE_MANIFEST_OVERRIDE")"
    reuse_manifest="${!reuse_var_name:-}"

    if [[ -n "$reuse_manifest" ]]; then
        vlaselect_register_cleanup_manifest "$reuse_manifest"
        echo "[breakdown] reusing ${family}: ${reuse_manifest}"
        vlaselect_print_suite_training_logs "$reuse_manifest" "breakdown" "$family"
        append_panel_entry "$family" "$reuse_manifest" "$launch_log"
        return 0
    fi

    if [[ "$COLLECT_ONLY" == "1" ]]; then
        echo "COLLECT_ONLY=1 requires ${reuse_var_name} for family ${family}" >&2
        exit 1
    fi

    echo "[breakdown] launching ${family} (${WORKLOAD_NAME_BY_FAMILY[$family]})"
    vlaselect_register_cleanup_manifest "$suite_manifest"

    case "$family" in
        octo)
            env \
                SUITE_STAMP="$SUITE_STAMP" \
                TAIL_LOG="$TAIL_LOG" \
                MONITOR_INTERVAL_SECONDS="$MONITOR_INTERVAL_SECONDS" \
                PLOT_INTERVAL_SECONDS="$PLOT_INTERVAL_SECONDS" \
                ENABLE_SELF_CURVE_WATCHER="$ENABLE_SELF_CURVE_WATCHER" \
                SMOKE="$MWE" \
                MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS="$MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS" \
                ENVS_ID_OVERRIDE="$envs_id" \
                ENV_CHANGE_TIME_POINTS_OVERRIDE="$ENV_CHANGE_TIME_POINTS" \
                bash "$launch_script" > "$launch_log" 2>&1
            ;;
        vla_adapter_new)
            env \
                SUITE_STAMP="$SUITE_STAMP" \
                TAIL_LOG="$TAIL_LOG" \
                MONITOR_INTERVAL_SECONDS="$MONITOR_INTERVAL_SECONDS" \
                PLOT_INTERVAL_SECONDS="$PLOT_INTERVAL_SECONDS" \
                SMOKE="$MWE" \
                MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS="$MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS" \
                ENVS_ID_OVERRIDE="$envs_id" \
                ENV_IDS_OVERRIDE="$envs_id" \
                ENV_CHANGE_TIME_POINTS_OVERRIDE="$ENV_CHANGE_TIME_POINTS" \
                bash "$launch_script" > "$launch_log" 2>&1
            ;;
        tinyvla)
            env \
                SUITE_STAMP="$SUITE_STAMP" \
                TAIL_LOG="$TAIL_LOG" \
                MONITOR_INTERVAL_SECONDS="$MONITOR_INTERVAL_SECONDS" \
                PLOT_INTERVAL_SECONDS="$PLOT_INTERVAL_SECONDS" \
                SMOKE="$MWE" \
                MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS="$MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS" \
                ENVS_ID_OVERRIDE="$envs_id" \
                ENV_IDS_OVERRIDE="$envs_id" \
                ENV_CHANGE_TIME_POINTS_OVERRIDE="$ENV_CHANGE_TIME_POINTS" \
                bash "$launch_script" > "$launch_log" 2>&1
            ;;
        edgevla)
            env \
                SUITE_STAMP="$SUITE_STAMP" \
                TAIL_LOG="$TAIL_LOG" \
                SMOKE="$EDGEVLA_SMOKE" \
                MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS="$MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS" \
                QUEUED_PER_GPU="$EDGEVLA_QUEUED_PER_GPU" \
                MONITOR_INTERVAL_SECONDS="$MONITOR_INTERVAL_SECONDS" \
                PLOT_INTERVAL_SECONDS="$PLOT_INTERVAL_SECONDS" \
                SUITE_ENVS_ID="$envs_id" \
                SUITE_ENV_CHANGE_TIME_POINTS="$ENV_CHANGE_TIME_POINTS" \
                ENVS_ID_OVERRIDE="$envs_id" \
                ENV_IDS_OVERRIDE="$envs_id" \
                ENV_CHANGE_TIME_POINTS_OVERRIDE="$ENV_CHANGE_TIME_POINTS" \
                bash "$launch_script" > "$launch_log" 2>&1
            ;;
        *)
            echo "Unsupported family: $family" >&2
            exit 1
            ;;
    esac

    vlaselect_print_suite_training_logs "$suite_manifest" "breakdown" "$family"
    append_panel_entry "$family" "$suite_manifest" "$launch_log"
    if [[ "$TAIL_LOG" == "1" ]]; then
        vlaselect_start_manifest_log_tail "$suite_manifest" "$family"
    fi
    wait_for_suite_completion "$family" "$suite_manifest"
}

for family in "${FAMILY_ORDER[@]}"; do
    if [[ "${SHOULD_RUN_FAMILY[$family]}" == "1" ]]; then
        launch_family_suite "$family"
    fi
done

python overhead/plot_breakdown_impl.py --manifest "$MANIFEST_JSON" --prepare-only

echo "Breakdown suite prepared: ${SUITE_STAMP}"
echo "Manifest: ${MANIFEST_JSON}"
echo "Prepared CSV: ${RUN_ROOT}/BREAKDOWN_ALL_METHODS.csv"
echo "Prepared CSV: ${RUN_ROOT}/BREAKDOWN_MODULES.csv"
