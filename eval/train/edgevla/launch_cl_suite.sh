#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "${ROOT_DIR}/common/interrupt_cleanup.sh"

SUITE_STAMP="${SUITE_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}"
SUITE_ROOT="${SUITE_ROOT_OVERRIDE:-train/edgevla/cl_suite/${SUITE_STAMP}}"
PLOTS_DIR="${SUITE_ROOT}/plots"
LAUNCH_LOG_DIR="${SUITE_ROOT}/launch_logs"
MANIFEST_TSV="${SUITE_ROOT}/methods.tsv"
MANIFEST_JSON="${SUITE_ROOT}/manifest.json"
PID_DIR="${SUITE_ROOT}/pids"

INHERIT_SUITE_FROM="${INHERIT_SUITE_FROM:-}"
RERUN_METHODS="${RERUN_METHODS:-}"
METHODS="${METHODS:-${RUN_METHODS:-}}"
SMOKE="${SMOKE:-0}"
TAIL_LOG="${TAIL_LOG:-1}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-30}"
PLOT_INTERVAL_SECONDS="${PLOT_INTERVAL_SECONDS:-60}"
QUEUED_PER_GPU="${QUEUED_PER_GPU:-1}"
vlaselect_install_cleanup_trap
if [[ -z "${VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE+x}" ]]; then
    if [[ "$SMOKE" == "1" ]]; then
        export VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE="0.35"
    else
        export VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE="0.0"
    fi
else
    export VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE
fi
export VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SEED="${VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SEED:-0}"
if [[ "$VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE" != "0" && "$VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE" != "0.0" ]]; then
    echo "[edgevla-suite] baseline pretrained checkpoint noise scale=$VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE seed=$VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SEED"
fi
GPU_BY_METHOD_OVERRIDE="${GPU_BY_METHOD_OVERRIDE:-}"

FULL_ENVS_ID="['UnitreeG1LiftCubeObjectScaleDown1p3-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeObjectPurple-v1','UnitreeG1LiftSphereLightStronger50-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectScaleDown1p1-v1','UnitreeG1LiftSphereObjectScaleDown1p3-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectPurple-v1']"
FULL_ENV_CHANGE_TIME_POINTS="[31,62,96,131,151,163,207,247,271,300]"

SUITE_ENVS_ID="${SUITE_ENVS_ID:-$FULL_ENVS_ID}"
SUITE_ENV_CHANGE_TIME_POINTS="${SUITE_ENV_CHANGE_TIME_POINTS:-$FULL_ENV_CHANGE_TIME_POINTS}"
RESOURCE_CHANGE_TIME_POINTS="${RESOURCE_CHANGE_TIME_POINTS_OVERRIDE:-}"
RESOURCE_CHANGE_DIRECTIONS="${RESOURCE_CHANGE_DIRECTIONS_OVERRIDE:-}"
RESOURCE_CHANGE_FACTORS="${RESOURCE_CHANGE_FACTORS_OVERRIDE:-}"

declare -a METHOD_ORDER=(
    ppo_gen
    ours
    conrft
    flare
    edgeta
    convertnet
    improv_vla
    self_improv
    vla_rft
    world_env
)

declare -A DISPLAY_NAMES=(
    [conrft]="ConRFT"
    [convertnet]="ConvertNet"
    [edgeta]="EdgeTA"
    [flare]="FLaRe"
    [improv_vla]="Improv-VLA"
    [ours]="Ours"
    [ppo_gen]="PPO-Gen"
    [self_improv]="Self-Improv"
    [vla_rft]="VLA-RFT"
    [world_env]="WorldEnv"
)

declare -A GPU_BY_METHOD=(
    [ppo_gen]=6
    [conrft]=1
    [convertnet]=1
    [edgeta]=6
    [flare]=6
    [improv_vla]=7
    [ours]=1
    [self_improv]=6
    [vla_rft]=7
    [world_env]=6
)

DEFAULT_GPU_BY_METHOD_RAW=""
for method in "${METHOD_ORDER[@]}"; do
    if [[ -n "$DEFAULT_GPU_BY_METHOD_RAW" ]]; then
        DEFAULT_GPU_BY_METHOD_RAW+=","
    fi
    DEFAULT_GPU_BY_METHOD_RAW+="${method}=${GPU_BY_METHOD[$method]}"
done

RESOLVED_GPU_TSV=$(python3 -m train.common.gpu_auto_select resolve-method-map \
    --method-order "$(IFS=,; echo "${METHOD_ORDER[*]}")" \
    --default-map "$DEFAULT_GPU_BY_METHOD_RAW" \
    --override-map "$GPU_BY_METHOD_OVERRIDE")
declare -A RESOLVED_GPU_BY_METHOD=()
while IFS=$'\t' read -r method gpu; do
    [[ -z "$method" ]] && continue
    RESOLVED_GPU_BY_METHOD["$method"]="$gpu"
done <<< "$RESOLVED_GPU_TSV"

declare -A SCRIPT_BY_METHOD=(
    [conrft]="train/edgevla/conrft/run_online_rl.sh"
    [convertnet]="train/edgevla/convertnet/run_online_rl.sh"
    [edgeta]="train/edgevla/edgeta/run_online_rl.sh"
    [flare]="train/edgevla/flare/run_online_rl.sh"
    [improv_vla]="train/edgevla/improv_vla/run_online_rl.sh"
    [ours]="train/edgevla/ours/run_online_rl_cl.sh"
    [ppo_gen]="train/edgevla/ppo_gen/run_online_rl.sh"
    [self_improv]="train/edgevla/self_improv/run_online_rl.sh"
    [vla_rft]="train/edgevla/vla_rft/run_online_rl.sh"
    [world_env]="train/edgevla/world_env/run_online_rl.sh"
)

resolve_suite_root() {
    local value="$1"
    local candidate=""
    if [[ -d "$value" ]]; then
        candidate="$value"
    elif [[ -d "$ROOT_DIR/$value" ]]; then
        candidate="$ROOT_DIR/$value"
    elif [[ -d "$ROOT_DIR/train/edgevla/cl_suite/$value" ]]; then
        candidate="$ROOT_DIR/train/edgevla/cl_suite/$value"
    else
        return 1
    fi
    (
        cd "$candidate"
        pwd
    )
}

select_methods() {
    local raw_selection="$1"
    if [[ -z "$raw_selection" ]]; then
        printf "%s\n" "${METHOD_ORDER[@]}"
        return
    fi
    printf "%s" "$raw_selection" | tr ',' '\n' | awk 'NF {gsub(/^[ \t]+|[ \t]+$/, ""); print}'
}

declare -a SELECTED_METHODS=()
while IFS= read -r method; do
    [[ -z "$method" ]] && continue
    if [[ -z "${SCRIPT_BY_METHOD[$method]+x}" ]]; then
        echo "Unknown method: $method" >&2
        exit 1
    fi
    SELECTED_METHODS+=("$method")
done < <(select_methods "$METHODS")

if [[ "${#SELECTED_METHODS[@]}" -eq 0 ]]; then
    echo "No methods selected." >&2
    exit 1
fi

declare -A SHOULD_RUN=()
if [[ -n "$INHERIT_SUITE_FROM" ]]; then
    for method in "${METHOD_ORDER[@]}"; do
        SHOULD_RUN["$method"]=0
    done
else
    for method in "${METHOD_ORDER[@]}"; do
        SHOULD_RUN["$method"]=0
    done
    for method in "${SELECTED_METHODS[@]}"; do
        SHOULD_RUN["$method"]=1
    done
fi

if [[ -z "$INHERIT_SUITE_FROM" && -n "$RERUN_METHODS" ]]; then
    echo "RERUN_METHODS requires INHERIT_SUITE_FROM so non-rerun methods keep valid results." >&2
    exit 1
fi

if [[ -n "$RERUN_METHODS" ]]; then
    rerun_method_count=0
    while IFS= read -r method; do
        [[ -z "$method" ]] && continue
        rerun_method_count=$((rerun_method_count + 1))
        if [[ -z "${SCRIPT_BY_METHOD[$method]+x}" ]]; then
            echo "Unknown method in RERUN_METHODS: $method" >&2
            exit 1
        fi
        SHOULD_RUN["$method"]=1
    done < <(select_methods "$RERUN_METHODS")
    if [[ "$rerun_method_count" -eq 0 ]]; then
        echo "RERUN_METHODS is set but empty after parsing: $RERUN_METHODS" >&2
        exit 1
    fi
fi

INHERITED_SUITE_ROOT=""
INHERITED_SUITE_LABEL=""
if [[ -n "$INHERIT_SUITE_FROM" ]]; then
    INHERITED_SUITE_ROOT="$(resolve_suite_root "$INHERIT_SUITE_FROM")" || {
        echo "Unable to resolve inherited suite: $INHERIT_SUITE_FROM" >&2
        exit 1
    }
    if [[ "$INHERITED_SUITE_ROOT" == "$ROOT_DIR/$SUITE_ROOT" ]]; then
        echo "Inherited suite root must differ from destination suite root: $INHERITED_SUITE_ROOT" >&2
        exit 1
    fi
    if [[ -e "$SUITE_ROOT" ]]; then
        echo "Destination suite already exists: $SUITE_ROOT" >&2
        exit 1
    fi
    mkdir -p "$SUITE_ROOT"
    cp -a "$INHERITED_SUITE_ROOT"/. "$SUITE_ROOT"/
    INHERITED_SUITE_LABEL="${INHERITED_SUITE_ROOT#$ROOT_DIR/}"
    if [[ "$INHERITED_SUITE_LABEL" == "$INHERITED_SUITE_ROOT" ]]; then
        INHERITED_SUITE_LABEL="$INHERITED_SUITE_ROOT"
    fi
fi

if [[ "$SMOKE" == "1" ]]; then
    : "${MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS:=300}"
    selected_method_count=0
    for method in "${METHOD_ORDER[@]}"; do
        if [[ "${SHOULD_RUN[$method]}" == "1" ]]; then
            selected_method_count=$((selected_method_count + 1))
        fi
    done
    if [[ "$selected_method_count" -lt 1 ]]; then
        selected_method_count=1
    fi
    MWE_PER_METHOD_RUNTIME_SECONDS=$((MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS / selected_method_count))
    if [[ "$MWE_PER_METHOD_RUNTIME_SECONDS" -lt 1 ]]; then
        MWE_PER_METHOD_RUNTIME_SECONDS=1
    fi
    MWE_PER_METHOD_RUNTIME_HOURS="$(awk -v sec="$MWE_PER_METHOD_RUNTIME_SECONDS" 'BEGIN { printf "%.6f", sec / 3600 }')"
    MWE_PER_METHOD_RUNTIME_MINUTES="$(awk -v sec="$MWE_PER_METHOD_RUNTIME_SECONDS" 'BEGIN { printf "%.6f", sec / 60 }')"
else
    MWE_PER_METHOD_RUNTIME_HOURS="0.0084"
    MWE_PER_METHOD_RUNTIME_MINUTES="0.500000"
fi

filter_method_smoke_overrides() {
    local method="$1"
    shift || true
    local -a raw_overrides=("$@")
    local -a filtered_overrides=()
    local entry key
    for entry in "${raw_overrides[@]}"; do
        key="${entry%%=*}"
        if [[ "$method" == "ours" ]]; then
            case "$key" in
                TOTAL_TIMESTEPS_OVERRIDE|NUM_ENVS_OVERRIDE|NUM_EVAL_ENVS_OVERRIDE|NUM_STEPS_OVERRIDE|NUM_MINIBATCHES_OVERRIDE|UPDATE_EPOCHS_OVERRIDE|ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE|EVAL_MICRO_BATCH_SIZE_OVERRIDE|UPDATE_MICRO_BATCH_SIZE_OVERRIDE|MAX_RUNTIME_HOURS_OVERRIDE)
                    continue
                    ;;
            esac
        fi
        filtered_overrides+=("$entry")
    done
    printf '%s
' "${filtered_overrides[@]}"
}

mkdir -p "$SUITE_ROOT" "$PLOTS_DIR" "$LAUNCH_LOG_DIR"
rm -rf "$PID_DIR"
mkdir -p "$PID_DIR"
rm -f "$MANIFEST_TSV" "$MANIFEST_JSON"

if [[ "$QUEUED_PER_GPU" == "1" ]]; then
    if [[ -n "$INHERIT_SUITE_FROM" || -n "$RERUN_METHODS" ]]; then
        echo "QUEUED_PER_GPU=1 does not support INHERIT_SUITE_FROM/RERUN_METHODS yet." >&2
        exit 1
    fi

    SCHEDULER_LOG="${LAUNCH_LOG_DIR}/scheduler.log"
    SCHEDULER_PID_FILE="${PID_DIR}/scheduler.pid"
    rm -f "$SCHEDULER_LOG" "$SCHEDULER_PID_FILE"
    scheduler_cmd=(
        python -u -m train.edgevla.run_cl_suite_queue
        --root-dir "$ROOT_DIR"
        --suite-root "$SUITE_ROOT"
        --manifest "$MANIFEST_JSON"
        --suite-stamp "$SUITE_STAMP"
        --envs-id "$SUITE_ENVS_ID"
        --env-change-time-points "$SUITE_ENV_CHANGE_TIME_POINTS"
        --monitor-interval-seconds "$MONITOR_INTERVAL_SECONDS"
        --plot-interval-seconds "$PLOT_INTERVAL_SECONDS"
        --tail-log "$TAIL_LOG"
        --resource-change-time-points "$RESOURCE_CHANGE_TIME_POINTS"
        --resource-change-directions "$RESOURCE_CHANGE_DIRECTIONS"
        --resource-change-factors "$RESOURCE_CHANGE_FACTORS"
        --gpu-by-method-override "$GPU_BY_METHOD_OVERRIDE"
        --methods "$METHODS"
        --smoke-max-runtime-hours "$MWE_PER_METHOD_RUNTIME_HOURS"
    )
    if [[ "$SMOKE" == "1" ]]; then
        scheduler_cmd+=(--smoke)
    fi

    python "$ROOT_DIR/train/octo/spawn_detached.py" \
        --pid-file "$SCHEDULER_PID_FILE" \
        --log-file "$SCHEDULER_LOG" \
        --cwd "$ROOT_DIR" \
        -- "${scheduler_cmd[@]}" \
        > /dev/null
    SCHEDULER_PID="$(cat "$SCHEDULER_PID_FILE")"

    for _ in $(seq 1 50); do
        if [[ -s "$MANIFEST_JSON" ]]; then
            break
        fi
        sleep 0.2
    done
    if [[ ! -s "$MANIFEST_JSON" ]]; then
        echo "Scheduler did not create manifest: $MANIFEST_JSON" >&2
        exit 1
    fi

    PLOTTER_LOG="${LAUNCH_LOG_DIR}/plotter.log"
    PLOTTER_PID_FILE="${PID_DIR}/plotter.pid"
    rm -f "$PLOTTER_LOG" "$PLOTTER_PID_FILE"
    python "$ROOT_DIR/train/octo/spawn_detached.py" \
        --pid-file "$PLOTTER_PID_FILE" \
        --log-file "$PLOTTER_LOG" \
        --cwd "$ROOT_DIR" \
        -- python -u -m train.edgevla.plot_cl_suite \
        --manifest "$MANIFEST_JSON" \
        --output-dir "$PLOTS_DIR" \
        --interval-seconds "$PLOT_INTERVAL_SECONDS" \
        > /dev/null
    PLOTTER_PID="$(cat "$PLOTTER_PID_FILE")"

    python - <<'PY' "$MANIFEST_JSON" "$PLOTTER_PID" "$PLOTTER_LOG"
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
plotter_pid = int(sys.argv[2])
plotter_log = sys.argv[3]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["plotter_pid"] = plotter_pid
manifest["plotter_log"] = plotter_log
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

    echo "Human CL suite launched: ${SUITE_STAMP}"
    echo "Smoke: ${SMOKE}"
    echo "Queued per GPU: 1"
    echo "GPU assignment override: ${GPU_BY_METHOD_OVERRIDE:-<default>}"
    echo "Manifest: ${MANIFEST_JSON}"
    echo "Plots: ${PLOTS_DIR}"
    echo "Scheduler PID: ${SCHEDULER_PID}"
    echo "Plotter PID: ${PLOTTER_PID}"
    exit 0
fi

printf "name\tdisplay_name\tgpu\tstatus\tinherited_from\tpid\tmonitor_pid\trun_dir\tlog_file\tscript_path\toutput_dir_base\trun_name\n" > "$MANIFEST_TSV"

for method in "${METHOD_ORDER[@]}"; do
    gpu="${RESOLVED_GPU_BY_METHOD[$method]}"
    script_path="${SCRIPT_BY_METHOD[$method]}"
    output_dir_base="${SUITE_ROOT}/${method}"
    run_name="run"
    run_dir="${output_dir_base}/${run_name}"
    launch_log="${LAUNCH_LOG_DIR}/${method}.launch.log"
    log_file="${LAUNCH_LOG_DIR}/${method}.train.log"
    pid_file="${PID_DIR}/${method}.pid"

    if [[ "${SHOULD_RUN[$method]}" != "1" ]]; then
        if [[ -n "$INHERITED_SUITE_LABEL" ]]; then
            method_status="inherited"
            inherited_from="$INHERITED_SUITE_LABEL"
        else
            method_status="skipped"
            inherited_from=""
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$method" "${DISPLAY_NAMES[$method]}" "$gpu" "$method_status" "$inherited_from" "" "" \
            "$run_dir" "" "$script_path" "$output_dir_base" "$run_name" >> "$MANIFEST_TSV"
        continue
    fi

    if [[ -n "$INHERITED_SUITE_ROOT" ]]; then
        rm -rf "$output_dir_base"
        rm -f "$launch_log" "$log_file" "$LAUNCH_LOG_DIR/${method}.gpu_monitor.log"
    fi

    cmd=(
        env
        CUDA_DEVICES="$gpu"
        OUTPUT_DIR_BASE_OVERRIDE="$output_dir_base"
        RUN_NAME_OVERRIDE="$run_name"
        LOG_FILE_OVERRIDE="$log_file"
        TAIL_LOG="$TAIL_LOG"
        LAUNCH_DIRECT=1
        SAVE_VIDEO_OVERRIDE=false
        ENVS_ID_OVERRIDE="$SUITE_ENVS_ID"
        ENV_CHANGE_TIME_POINTS_OVERRIDE="$SUITE_ENV_CHANGE_TIME_POINTS"
    )

    if [[ "$SMOKE" == "1" ]]; then
        mapfile -t method_smoke_overrides < <(filter_method_smoke_overrides "$method"             TOTAL_TIMESTEPS_OVERRIDE=1024             NUM_ENVS_OVERRIDE=2             NUM_EVAL_ENVS_OVERRIDE=8             NUM_STEPS_OVERRIDE=16             NUM_MINIBATCHES_OVERRIDE=2             UPDATE_EPOCHS_OVERRIDE=1             EVAL_EVERY_UPDATES_OVERRIDE=15             EVAL_EPISODES_OVERRIDE=8             MAX_RUNTIME_HOURS_OVERRIDE="$MWE_PER_METHOD_RUNTIME_HOURS"             EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE=45             ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE=2             EVAL_MICRO_BATCH_SIZE_OVERRIDE=3             UPDATE_MICRO_BATCH_SIZE_OVERRIDE=1             ROLLOUT_PROGRESS_LOG_INTERVAL_OVERRIDE=1             SUPERVISED_UPDATES_PER_ITER_OVERRIDE=1             SUPERVISED_BATCH_SIZE_OVERRIDE=2             ONLINE_BUFFER_CAPACITY_OVERRIDE=256             EXPERT_BUFFER_CAPACITY_OVERRIDE=256             EXPERT_TARGET_SUCCESS_TRAJECTORIES_OVERRIDE=0             EXPERT_COLLECT_NUM_ENVS_OVERRIDE=1             EXPERT_COLLECT_MAX_STEPS_OVERRIDE=128             MWE_ACTIVE_RUNTIME_ONLY=1)
        cmd+=("${method_smoke_overrides[@]}")
        if [[ "$method" == "ours" ]]; then
            cmd+=(
                MWE=1
                MWE_MAX_RUNTIME_MINUTES="$MWE_PER_METHOD_RUNTIME_MINUTES"
                EVAL_EVERY_UPDATES_OVERRIDE=32
            )
        fi
    fi

    cmd+=(bash "$script_path")

    python "$ROOT_DIR/train/octo/spawn_detached.py" \
        --pid-file "$pid_file" \
        --log-file "$log_file" \
        --cwd "$ROOT_DIR" \
        -- "${cmd[@]}" \
        > "$launch_log"
    train_pid="$(cat "$pid_file")"

    monitor_pid_file="${PID_DIR}/${method}.gpu_monitor.pid"
    monitor_log="${LAUNCH_LOG_DIR}/${method}.gpu_monitor.log"
    python "$ROOT_DIR/train/octo/spawn_detached.py" \
        --pid-file "$monitor_pid_file" \
        --log-file "$monitor_log" \
        --cwd "$ROOT_DIR" \
        -- python -u -m train.common.monitor_gpu_metrics \
        --pid "$train_pid" \
        --gpu-index "$gpu" \
        --run-dir "$run_dir" \
        --interval-seconds "$MONITOR_INTERVAL_SECONDS" \
        --label "$method" \
        > /dev/null
    monitor_pid="$(cat "$monitor_pid_file")"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$method" "${DISPLAY_NAMES[$method]}" "$gpu" "launched" "" "$train_pid" "$monitor_pid" \
        "$run_dir" "$log_file" "$script_path" "$output_dir_base" "$run_name" >> "$MANIFEST_TSV"
done

python - <<'PY' "$MANIFEST_TSV" "$MANIFEST_JSON" "$SUITE_STAMP" "$SMOKE" "$MONITOR_INTERVAL_SECONDS" "$PLOT_INTERVAL_SECONDS" "$SUITE_ENVS_ID" "$SUITE_ENV_CHANGE_TIME_POINTS" "$INHERITED_SUITE_LABEL" "$MWE_PER_METHOD_RUNTIME_HOURS"
import csv
import json
import sys
from pathlib import Path

tsv_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
suite_stamp = sys.argv[3]
smoke = sys.argv[4] == "1"
monitor_interval = float(sys.argv[5])
plot_interval = float(sys.argv[6])
envs_id = sys.argv[7]
env_change_time_points = sys.argv[8]
inherited_suite = sys.argv[9] or None
smoke_max_runtime_hours = float(sys.argv[10]) if sys.argv[10] else None

with tsv_path.open("r", encoding="utf-8") as f:
    methods = []
    for row in csv.DictReader(f, delimiter="\t"):
        methods.append(
            {
                "name": row["name"],
                "display_name": row["display_name"],
                "gpu": int(row["gpu"]),
                "status": row["status"],
                "inherited_from": row["inherited_from"] or None,
                "pid": int(row["pid"]) if row["pid"] else None,
                "monitor_pid": int(row["monitor_pid"]) if row["monitor_pid"] else None,
                "run_dir": row["run_dir"],
                "log_file": row["log_file"] or None,
                "script_path": row["script_path"],
                "output_dir_base": row["output_dir_base"],
                "run_name": row["run_name"],
            }
        )

manifest = {
    "suite_stamp": suite_stamp,
    "smoke": smoke,
    "monitor_interval_seconds": monitor_interval,
    "plot_interval_seconds": plot_interval,
    "envs_id": envs_id,
    "env_change_time_points": env_change_time_points,
    "inherited_suite": inherited_suite,
    "smoke_max_runtime_hours": smoke_max_runtime_hours if smoke else None,
    "methods": methods,
}
json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

PLOTTER_LOG="${LAUNCH_LOG_DIR}/plotter.log"
PLOTTER_PID_FILE="${PID_DIR}/plotter.pid"
rm -f "$PLOTTER_LOG" "$PLOTTER_PID_FILE"
python "$ROOT_DIR/train/octo/spawn_detached.py" \
    --pid-file "$PLOTTER_PID_FILE" \
    --log-file "$PLOTTER_LOG" \
    --cwd "$ROOT_DIR" \
    -- python -u -m train.edgevla.plot_cl_suite \
    --manifest "$MANIFEST_JSON" \
    --output-dir "$PLOTS_DIR" \
    --interval-seconds "$PLOT_INTERVAL_SECONDS" \
    > /dev/null
PLOTTER_PID="$(cat "$PLOTTER_PID_FILE")"

python - <<'PY' "$MANIFEST_JSON" "$PLOTTER_PID" "$PLOTTER_LOG"
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
plotter_pid = int(sys.argv[2])
plotter_log = sys.argv[3]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["plotter_pid"] = plotter_pid
manifest["plotter_log"] = plotter_log
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

echo "Human CL suite launched: ${SUITE_STAMP}"
echo "Smoke: ${SMOKE}"
echo "Manifest: ${MANIFEST_JSON}"
echo "Plots: ${PLOTS_DIR}"
echo "Plotter PID: ${PLOTTER_PID}"
