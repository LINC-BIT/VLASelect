#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "${ROOT_DIR}/common/interrupt_cleanup.sh"

SUITE_STAMP="${SUITE_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}"
SUITE_ROOT="ckpt/vla_adapter_new/cl_suite/${SUITE_STAMP}"
PLOTS_DIR="${SUITE_ROOT}/plots"
LAUNCH_LOG_DIR="${SUITE_ROOT}/launch_logs"
MANIFEST_TSV="${SUITE_ROOT}/methods.tsv"
MANIFEST_JSON="${SUITE_ROOT}/manifest.json"
PID_DIR="${SUITE_ROOT}/pids"

INHERIT_SUITE_FROM="${INHERIT_SUITE_FROM:-}"
RERUN_METHODS="${RERUN_METHODS:-}"
SMOKE="${SMOKE:-0}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-30}"
PLOT_INTERVAL_SECONDS="${PLOT_INTERVAL_SECONDS:-60}"
TAIL_LOG="${TAIL_LOG:-1}"
vlaselect_install_cleanup_trap
GPU_BY_METHOD_OVERRIDE="${GPU_BY_METHOD_OVERRIDE:-}"
GPU_QUEUE_POLL_SECONDS="${GPU_QUEUE_POLL_SECONDS:-30}"
RESOURCE_CHANGE_TIME_POINTS="${RESOURCE_CHANGE_TIME_POINTS_OVERRIDE:-}"
RESOURCE_CHANGE_DIRECTIONS="${RESOURCE_CHANGE_DIRECTIONS_OVERRIDE:-}"
RESOURCE_CHANGE_FACTORS="${RESOURCE_CHANGE_FACTORS_OVERRIDE:-}"

SMOKE_ENV_OVERRIDES=()
if [[ "$SMOKE" == "1" ]]; then
    SMOKE_ENV_OVERRIDES=(
        TOTAL_TIMESTEPS_OVERRIDE=200
        NUM_ENVS_OVERRIDE=1
        NUM_EVAL_ENVS_OVERRIDE=1
        NUM_STEPS_OVERRIDE=50
        NUM_MINIBATCHES_OVERRIDE=1
        UPDATE_EPOCHS_OVERRIDE=1
        MAX_RUNTIME_HOURS_OVERRIDE=0.08
        TRAIN_VIDEO_NUM_ENVS_OVERRIDE=1
        TEST_VIDEO_NUM_ENVS_OVERRIDE=1
        TEST_VIDEO_EPISODES_OVERRIDE=1
        ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE=1
        EVAL_MICRO_BATCH_SIZE_OVERRIDE=1
        UPDATE_MICRO_BATCH_SIZE_OVERRIDE=1
        EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE=45
    )
fi

declare -a METHOD_ORDER=(
    conrft
    flare
    improv_vla
    edgeta
    convertnet
    ours
    ppo_gen
    self_improv
    vla_rft
    world_env
)

declare -A DISPLAY_NAMES=(
    [conrft]="ConRFT"
    [flare]="FLaRe"
    [improv_vla]="Improv-VLA"
    [edgeta]="EdgeTA"
    [convertnet]="ConvertNet"
    [ppo_gen]="PPO-Gen"
    [ours]="VLASelect"
    [self_improv]="Self-Improv"
    [vla_rft]="VLA-RFT"
    [world_env]="WorldEnv"
)

declare -A GPU_BY_METHOD=(
    [conrft]=0
    [flare]=1
    [improv_vla]=2
    [edgeta]=5
    [convertnet]=6
    [ours]=1
    [ppo_gen]=3
    [self_improv]=2
    [vla_rft]=1
    [world_env]=2
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
    [conrft]="train/vla_adapter_new/conrft/run_online_rl.sh"
    [flare]="train/vla_adapter_new/flare/run_online_rl.sh"
    [improv_vla]="train/vla_adapter_new/improv_vla/run_online_rl.sh"
    [edgeta]="train/vla_adapter_new/edgeta/online_rl_edgeta.sh"
    [convertnet]="train/vla_adapter_new/convertnet/online_rl_convertnet.sh"
    [ours]="train/vla_adapter_new/ours/run_online_rl_cl.sh"
    [ppo_gen]="train/vla_adapter_new/ppo_gen/run_online_rl.sh"
    [self_improv]="train/vla_adapter_new/self_improv/run_online_rl.sh"
    [vla_rft]="train/vla_adapter_new/vla_rft/run_online_rl.sh"
    [world_env]="train/vla_adapter_new/world_env/run_online_rl.sh"
)

declare -A EXP_NAME_BY_METHOD=(
    [conrft]="vla_adapter_new/cl_suite/${SUITE_STAMP}/conrft"
    [flare]="vla_adapter_new/cl_suite/${SUITE_STAMP}/flare"
    [improv_vla]="vla_adapter_new/cl_suite/${SUITE_STAMP}/improv_vla"
    [edgeta]="vla_adapter_new/cl_suite/${SUITE_STAMP}/edgeta"
    [convertnet]="vla_adapter_new/cl_suite/${SUITE_STAMP}/convertnet"
    [ours]="vla_adapter_new/cl_suite/${SUITE_STAMP}/ours"
    [ppo_gen]="vla_adapter_new/cl_suite/${SUITE_STAMP}/ppo_gen"
    [self_improv]="vla_adapter_new/cl_suite/${SUITE_STAMP}/self_improv"
    [vla_rft]="vla_adapter_new/cl_suite/${SUITE_STAMP}/vla_rft"
    [world_env]="vla_adapter_new/cl_suite/${SUITE_STAMP}/world_env"
)

resolve_suite_root() {
    local value="$1"
    local candidate=""
    if [[ -d "$value" ]]; then
        candidate="$value"
    elif [[ -d "$ROOT_DIR/$value" ]]; then
        candidate="$ROOT_DIR/$value"
    elif [[ -d "$ROOT_DIR/ckpt/vla_adapter_new/cl_suite/$value" ]]; then
        candidate="$ROOT_DIR/ckpt/vla_adapter_new/cl_suite/$value"
    else
        return 1
    fi
    (
        cd "$candidate"
        pwd
    )
}

declare -A SHOULD_RUN=()
if [[ -n "$INHERIT_SUITE_FROM" ]]; then
    for method in "${METHOD_ORDER[@]}"; do
        SHOULD_RUN["$method"]=0
    done
else
    for method in "${METHOD_ORDER[@]}"; do
        SHOULD_RUN["$method"]=1
    done
fi

if [[ -z "$INHERIT_SUITE_FROM" && -n "$RERUN_METHODS" ]]; then
    echo "RERUN_METHODS requires INHERIT_SUITE_FROM so non-rerun methods keep valid results." >&2
    exit 1
fi

if [[ -n "$RERUN_METHODS" ]]; then
    method_selection="$(printf '%s' "$RERUN_METHODS" | tr ',' ' ')"
    read -r -a requested_methods <<< "$method_selection"
    if [[ "${#requested_methods[@]}" -eq 0 ]]; then
        echo "RERUN_METHODS is set but empty after parsing: $RERUN_METHODS" >&2
        exit 1
    fi
    for method in "${requested_methods[@]}"; do
        if [[ -z "${SCRIPT_BY_METHOD[$method]+x}" ]]; then
            echo "Unknown method in RERUN_METHODS: $method" >&2
            exit 1
        fi
        SHOULD_RUN["$method"]=1
    done
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

mkdir -p "$SUITE_ROOT" "$PLOTS_DIR" "$LAUNCH_LOG_DIR"
rm -rf "$PID_DIR"
mkdir -p "$PID_DIR"
rm -f "$MANIFEST_TSV" "$MANIFEST_JSON"
printf "name\tdisplay_name\tgpu\tstatus\tinherited_from\tpid\tmonitor_pid\trun_dir\tlog_file\tscript_path\texp_name\n" > "$MANIFEST_TSV"

declare -A LAST_PID_BY_GPU=()

for method in "${METHOD_ORDER[@]}"; do
    gpu="${RESOLVED_GPU_BY_METHOD[$method]}"
    script_path="${SCRIPT_BY_METHOD[$method]}"
    exp_name="${EXP_NAME_BY_METHOD[$method]}"
    launch_log="${LAUNCH_LOG_DIR}/${method}.launch.log"
    pid_file="${PID_DIR}/${method}.pid"
    run_dir="ckpt/${exp_name}"

    if [[ "${SHOULD_RUN[$method]}" != "1" ]]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$method" \
            "${DISPLAY_NAMES[$method]}" \
            "$gpu" \
            "inherited" \
            "$INHERITED_SUITE_LABEL" \
            "" \
            "" \
            "$run_dir" \
            "" \
            "$script_path" \
            "$exp_name" \
            >> "$MANIFEST_TSV"
        continue
    fi

    if [[ -n "$INHERITED_SUITE_ROOT" ]]; then
        rm -rf "$run_dir"
        rm -f "$LAUNCH_LOG_DIR/${method}.launch.log" "$LAUNCH_LOG_DIR/${method}.gpu_monitor.log"
    fi

    case "$method" in
        conrft)
            log_file="train/vla_adapter_new/conrft/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-cl_suite.log"
            ;;
        flare)
            log_file="train/vla_adapter_new/flare/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-cl_suite.log"
            ;;
        improv_vla)
            log_file="train/vla_adapter_new/improv_vla/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-cl_suite.log"
            ;;
        edgeta)
            log_file="train/vla_adapter_new/edgeta/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-cl_suite.log"
            ;;
        convertnet)
            log_file="train/vla_adapter_new/convertnet/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-cl_suite.log"
            ;;
        ours)
            log_file="train/vla_adapter_new/ours/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-cl_suite.log"
            ;;
        ppo_gen)
            log_file="train/vla_adapter_new/ppo_gen/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-cl_suite.log"
            ;;
        self_improv)
            log_file="train/vla_adapter_new/self_improv/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-cl_suite.log"
            ;;
        vla_rft)
            log_file="train/vla_adapter_new/vla_rft/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-cl_suite.log"
            ;;
        world_env)
            log_file="train/vla_adapter_new/world_env/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-cl_suite.log"
            ;;
        *)
            echo "Unhandled method: $method" >&2
            exit 1
            ;;
    esac

    cmd=(
        env
        CUDA_DEVICES="$gpu"
        EXP_NAME="$exp_name"
        TAIL_LOG="$TAIL_LOG"
        LAUNCH_DIRECT=1
        LOG_FILE_OVERRIDE="$log_file"
        RESOURCE_CHANGE_TIME_POINTS_OVERRIDE="$RESOURCE_CHANGE_TIME_POINTS"
        RESOURCE_CHANGE_DIRECTIONS_OVERRIDE="$RESOURCE_CHANGE_DIRECTIONS"
        RESOURCE_CHANGE_FACTORS_OVERRIDE="$RESOURCE_CHANGE_FACTORS"
        "${SMOKE_ENV_OVERRIDES[@]}"
        bash "$script_path"
    )

    wait_for_pid="${LAST_PID_BY_GPU[$gpu]:-}"
    cmd_escaped="$(printf '%q ' "${cmd[@]}")"
    cmd_escaped="${cmd_escaped% }"
    if [[ -n "$wait_for_pid" ]]; then
        launch_cmd=(
            bash
            -lc
            "while kill -0 ${wait_for_pid} 2>/dev/null; do sleep ${GPU_QUEUE_POLL_SECONDS}; done; exec ${cmd_escaped}"
        )
    else
        launch_cmd=("${cmd[@]}")
    fi

    python "$ROOT_DIR/train/octo/spawn_detached.py" \
        --pid-file "$pid_file" \
        --log-file "$log_file" \
        --cwd "$ROOT_DIR" \
        -- "${launch_cmd[@]}" \
        > "$launch_log"
    train_pid="$(cat "$pid_file")"
    LAST_PID_BY_GPU["$gpu"]="$train_pid"

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

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$method" \
        "${DISPLAY_NAMES[$method]}" \
        "$gpu" \
        "launched" \
        "" \
        "$train_pid" \
        "$monitor_pid" \
        "$run_dir" \
        "$log_file" \
        "$script_path" \
        "$exp_name" \
        >> "$MANIFEST_TSV"
done

python - <<'PY' "$MANIFEST_TSV" "$MANIFEST_JSON" "$SUITE_STAMP" "$MONITOR_INTERVAL_SECONDS" "$PLOT_INTERVAL_SECONDS" "$INHERITED_SUITE_LABEL" "$RESOURCE_CHANGE_TIME_POINTS" "$RESOURCE_CHANGE_DIRECTIONS" "$RESOURCE_CHANGE_FACTORS"
import csv
import json
import sys
from pathlib import Path

tsv_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
suite_stamp = sys.argv[3]
monitor_interval = float(sys.argv[4])
plot_interval = float(sys.argv[5])
inherited_suite = sys.argv[6] or None
resource_change_time_points = sys.argv[7] or None
resource_change_directions = sys.argv[8] or None
resource_change_factors = sys.argv[9] or None

with tsv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    methods = []
    for row in reader:
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
                "exp_name": row["exp_name"],
            }
        )

manifest = {
    "suite_stamp": suite_stamp,
    "monitor_interval_seconds": monitor_interval,
    "plot_interval_seconds": plot_interval,
    "inherited_suite": inherited_suite,
    "resource_change_time_points": resource_change_time_points,
    "resource_change_directions": resource_change_directions,
    "resource_change_factors": resource_change_factors,
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
    -- python -u -m train.vla_adapter_new.plot_cl_suite \
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

WATCHER_LOG="${LAUNCH_LOG_DIR}/watcher.log"
WATCHER_PID_FILE="${PID_DIR}/watcher.pid"
rm -f "$WATCHER_LOG" "$WATCHER_PID_FILE"
python "$ROOT_DIR/train/octo/spawn_detached.py"     --pid-file "$WATCHER_PID_FILE"     --log-file "$WATCHER_LOG"     --cwd "$ROOT_DIR"     -- python -u -m train.common.watch_suite_manifest     --manifest "$MANIFEST_JSON"     --interval-seconds "$MONITOR_INTERVAL_SECONDS"     > /dev/null
WATCHER_PID="$(cat "$WATCHER_PID_FILE")"

python - <<'PY' "$MANIFEST_JSON" "$WATCHER_PID" "$WATCHER_LOG"
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
watcher_pid = int(sys.argv[2])
watcher_log = sys.argv[3]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["watcher_pid"] = watcher_pid
manifest["watcher_log"] = watcher_log
manifest.setdefault("suite_state", "running")
manifest.setdefault("suite_status", "running")
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

echo "Suite launched: ${SUITE_STAMP}"
echo "Manifest: ${MANIFEST_JSON}"
echo "Plots: ${PLOTS_DIR}"
echo "Plotter PID: ${PLOTTER_PID}"
echo "Watcher PID: ${WATCHER_PID}"
