#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "${ROOT_DIR}/common/interrupt_cleanup.sh"

SUITE_STAMP="${SUITE_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}"
SUITE_ROOT="ckpt/cl_suite/${SUITE_STAMP}"
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
ENABLE_SELF_CURVE_WATCHER="${ENABLE_SELF_CURVE_WATCHER:-0}"

WORLD_ENV_WM_CKPT="${WORLD_ENV_WM_CKPT:-ckpt/PickCube-v1/baselines/world_env/world_model/20260425-032853-run032_e2/checkpoints/best_with_reference.pt}"
VLA_RFT_WM_CKPT="${VLA_RFT_WM_CKPT:-ckpt/PickCube-v1/baselines/vla_rft/world_model/20260424-160154-run32x2/checkpoints/best.pt}"
RESOURCE_CHANGE_TIME_POINTS="${RESOURCE_CHANGE_TIME_POINTS_OVERRIDE:-}"
RESOURCE_CHANGE_DIRECTIONS="${RESOURCE_CHANGE_DIRECTIONS_OVERRIDE:-}"
RESOURCE_CHANGE_FACTORS="${RESOURCE_CHANGE_FACTORS_OVERRIDE:-}"
DEFAULT_ENV_CONFIG_PATH="datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json"
FALLBACK_ENV_CONFIG_PATH="${SUITE_ROOT}/fallback_env_config.json"

SMOKE_ENV_OVERRIDES=()
if [[ "$SMOKE" == "1" ]]; then
    SMOKE_ENV_OVERRIDES=(
        TOTAL_TIMESTEPS_OVERRIDE=1024
        NUM_ENVS_OVERRIDE=8
        NUM_EVAL_ENVS_OVERRIDE=2
        NUM_STEPS_OVERRIDE=16
        NUM_EVAL_STEPS_OVERRIDE=16
        NUM_MINIBATCHES_OVERRIDE=2
        UPDATE_EPOCHS_OVERRIDE=1
        SUPERVISED_UPDATES_PER_ITER_OVERRIDE=1
        SUPERVISED_BATCH_SIZE_OVERRIDE=8
        WANDB_MODE=disabled
        WANDB_SILENT=true
    )
fi

SMOKE_SHARED_EXPERT_DEMO_PATH="${SMOKE_SHARED_EXPERT_DEMO_PATH:-train/octo/smoke_inputs/expert_demo.h5}"
SMOKE_SHARED_EXPERT_DEMO_JSON="${SMOKE_SHARED_EXPERT_DEMO_PATH%.h5}.json"

if [[ -n "${ENV_CONFIG_PATH_OVERRIDE:-}" ]]; then
    EFFECTIVE_ENV_CONFIG_PATH="$ENV_CONFIG_PATH_OVERRIDE"
elif [[ -f "$DEFAULT_ENV_CONFIG_PATH" ]]; then
    EFFECTIVE_ENV_CONFIG_PATH="$DEFAULT_ENV_CONFIG_PATH"
else
    mkdir -p "$(dirname "$FALLBACK_ENV_CONFIG_PATH")"
    cat > "$FALLBACK_ENV_CONFIG_PATH" <<'EOF'
{
  "env_info": {
    "env_kwargs": {
      "obs_mode": "rgb+depth+state_dict",
      "control_mode": "pd_ee_delta_pos",
      "reward_mode": "normalized_dense",
      "render_mode": "all",
      "num_envs": 1
    }
  }
}
EOF
    EFFECTIVE_ENV_CONFIG_PATH="$FALLBACK_ENV_CONFIG_PATH"
fi

prepare_smoke_inputs() {
    if [[ "$SMOKE" != "1" ]]; then
        return 0
    fi
    if [[ -f "$SMOKE_SHARED_EXPERT_DEMO_PATH" && -f "$SMOKE_SHARED_EXPERT_DEMO_JSON" ]]; then
        echo "[octo-smoke] reuse expert demo: $SMOKE_SHARED_EXPERT_DEMO_PATH"
        return 0
    fi
    echo "[octo-smoke] prepare expert demo before timed MWE run"
    OUTPUT_PATH="$SMOKE_SHARED_EXPERT_DEMO_PATH" \
    ENV_CONFIG_PATH="$EFFECTIVE_ENV_CONFIG_PATH" \
    STATE_NORM_STATS_PATH="ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth" \
    CHECKPOINT_PATH="ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt" \
    TARGET_SUCCESS_TRAJECTORIES="${SMOKE_SHARED_EXPERT_DEMO_TARGET_SUCCESS_TRAJECTORIES:-4}" \
    NUM_ENVS="${SMOKE_SHARED_EXPERT_DEMO_NUM_ENVS:-4}" \
    MAX_STEPS="${SMOKE_SHARED_EXPERT_DEMO_MAX_STEPS:-128}" \
    SEED="${SMOKE_SHARED_EXPERT_DEMO_SEED:-0}" \
    LOG_PREFIX="${SMOKE_SHARED_EXPERT_DEMO_LOG_PREFIX:-octo-smoke-expert-demo}" \
    bash train/octo/prepare_expert_demo.sh
}

declare -a METHOD_ORDER=(
    conrft
    flare
    improv_vla
    edgeta
    convertnet
    ours_single_agent
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
    [ours_single_agent]="VLASelect"
    [ppo_gen]="PPO-Gen"
    [self_improv]="Self-Improvement"
    [vla_rft]="VLA-RFT"
    [world_env]="WorldEnv"
)

declare -A GPU_BY_METHOD=(
    [conrft]=0
    [flare]=1
    [improv_vla]=2
    [edgeta]=0
    [convertnet]=1
    [ours_single_agent]=3
    [ppo_gen]=4
    [self_improv]=5
    [vla_rft]=6
    [world_env]=7
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
    [conrft]="train/octo/conrft/online_rl_conrft.sh"
    [flare]="train/octo/flare/online_rl_flare.sh"
    [improv_vla]="train/octo/improv_vla/online_rl_improv_vla.sh"
    [edgeta]="train/octo/edgeta/online_rl_edgeta.sh"
    [convertnet]="train/octo/convertnet/online_rl_convertnet.sh"
    [ours_single_agent]="train/octo/ours_single_agent/online_rl_ours_single_agent_cl.sh"
    [ppo_gen]="train/octo/ppo_gen/online_rl_ppo_gen.sh"
    [self_improv]="train/octo/self_improv/online_rl_self_improv.sh"
    [vla_rft]="train/octo/vla_rft/online_rl_vla_rft.sh"
    [world_env]="train/octo/world_env/online_rl_world_env.sh"
)

declare -A EXP_NAME_BY_METHOD=(
    [conrft]="cl_suite/${SUITE_STAMP}/conrft"
    [flare]="cl_suite/${SUITE_STAMP}/flare"
    [improv_vla]="cl_suite/${SUITE_STAMP}/improv_vla"
    [edgeta]="cl_suite/${SUITE_STAMP}/edgeta"
    [convertnet]="cl_suite/${SUITE_STAMP}/convertnet"
    [ours_single_agent]="cl_suite/${SUITE_STAMP}/ours_single_agent"
    [ppo_gen]="cl_suite/${SUITE_STAMP}/ppo_gen"
    [self_improv]="cl_suite/${SUITE_STAMP}/self_improv"
    [vla_rft]="cl_suite/${SUITE_STAMP}/vla_rft"
    [world_env]="cl_suite/${SUITE_STAMP}/world_env"
)

resolve_suite_root() {
    local value="$1"
    local candidate=""
    if [[ -d "$value" ]]; then
        candidate="$value"
    elif [[ -d "$ROOT_DIR/$value" ]]; then
        candidate="$ROOT_DIR/$value"
    elif [[ -d "$ROOT_DIR/ckpt/cl_suite/$value" ]]; then
        candidate="$ROOT_DIR/ckpt/cl_suite/$value"
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

prepare_smoke_inputs

: "${MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS:=300}"
if [[ "$SMOKE" == "1" ]]; then
    selected_method_count=0
    for method in "${METHOD_ORDER[@]}"; do
        if [[ "${SHOULD_RUN[$method]}" == "1" ]]; then
            selected_method_count=$((selected_method_count + 1))
        fi
    done
    if [[ "$selected_method_count" -lt 1 ]]; then
        selected_method_count=1
    fi
    mwe_per_method_runtime_seconds=$((MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS / selected_method_count))
    if [[ "$mwe_per_method_runtime_seconds" -lt 1 ]]; then
        mwe_per_method_runtime_seconds=1
    fi
    SMOKE_ENV_OVERRIDES+=(MAX_TIME_OVERRIDE="$mwe_per_method_runtime_seconds")
fi

declare -A LAST_PID_BY_GPU=()

for method in "${METHOD_ORDER[@]}"; do
    gpu="${RESOLVED_GPU_BY_METHOD[$method]}"
    script_path="${SCRIPT_BY_METHOD[$method]}"
    exp_name="${EXP_NAME_BY_METHOD[$method]}"
    launch_log="${LAUNCH_LOG_DIR}/${method}.launch.log"
    pid_file="${PID_DIR}/${method}.pid"

    if [[ "$method" == "ours_single_agent" || "$method" == "edgeta" || "$method" == "convertnet" ]]; then
        run_dir="ckpt/${exp_name}/[agent]"
    else
        run_dir="ckpt/${exp_name}"
    fi

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
        rm -rf "$SUITE_ROOT/$method"
        rm -f "$LAUNCH_LOG_DIR/${method}.launch.log" "$LAUNCH_LOG_DIR/${method}.gpu_monitor.log"
    fi

    if [[ "$method" == "conrft" ]]; then
        log_file="train/octo/conrft/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}.log"
    elif [[ "$method" == "flare" ]]; then
        log_file="train/octo/flare/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-flare_1h.log"
    elif [[ "$method" == "improv_vla" ]]; then
        log_file="train/octo/improv_vla/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}.log"
    elif [[ "$method" == "edgeta" ]]; then
        log_file="train/octo/edgeta/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}.log"
    elif [[ "$method" == "convertnet" ]]; then
        log_file="train/octo/convertnet/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}.log"
    elif [[ "$method" == "ours_single_agent" ]]; then
        log_file="train/octo/ours_single_agent/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}.log"
    elif [[ "$method" == "ppo_gen" ]]; then
        log_file="train/octo/ppo_gen/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}.log"
    elif [[ "$method" == "self_improv" ]]; then
        log_file="train/octo/self_improv/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}.log"
    elif [[ "$method" == "vla_rft" ]]; then
        log_file="train/octo/vla_rft/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-online_rl.log"
    else
        log_file="train/octo/world_env/nohup_out/$(date +"%Y-%m-%d")/${SUITE_STAMP}-online_rl.log"
    fi

    cmd=(
        env
        CUDA_DEVICES="$gpu"
        EXP_NAME="$exp_name"
        TAIL_LOG="$TAIL_LOG"
        LAUNCH_DIRECT=1
        SMOKE="$SMOKE"
        LOG_FILE_OVERRIDE="$log_file"
        ENABLE_SELF_CURVE_WATCHER="$ENABLE_SELF_CURVE_WATCHER"
        RESOURCE_CHANGE_TIME_POINTS_OVERRIDE="$RESOURCE_CHANGE_TIME_POINTS"
        RESOURCE_CHANGE_DIRECTIONS_OVERRIDE="$RESOURCE_CHANGE_DIRECTIONS"
        RESOURCE_CHANGE_FACTORS_OVERRIDE="$RESOURCE_CHANGE_FACTORS"
        ENV_CONFIG_PATH_OVERRIDE="$EFFECTIVE_ENV_CONFIG_PATH"
        "${SMOKE_ENV_OVERRIDES[@]}"
    )
    if [[ "$SMOKE" == "1" && ( "$method" == "conrft" || "$method" == "improv_vla" ) ]]; then
        cmd+=(
            EXPERT_DEMO_PATH_OVERRIDE="$SMOKE_SHARED_EXPERT_DEMO_PATH"
            AUTO_GENERATE_EXPERT_DEMO=0
        )
    fi
    if [[ "$method" == "vla_rft" ]]; then
        cmd+=(WORLD_MODEL_CKPT="$VLA_RFT_WM_CKPT")
    elif [[ "$method" == "world_env" ]]; then
        cmd+=(WORLD_MODEL_CKPT="$WORLD_ENV_WM_CKPT")
    fi
    cmd+=(bash "$script_path")

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

    python "$ROOT_DIR/train/octo/spawn_detached.py"         --pid-file "$pid_file"         --log-file "$log_file"         --cwd "$ROOT_DIR"         -- "${launch_cmd[@]}"         > "$launch_log"
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

python - <<'PY' "$MANIFEST_TSV" "$MANIFEST_JSON" "$SUITE_STAMP" "$MONITOR_INTERVAL_SECONDS" "$PLOT_INTERVAL_SECONDS" "$WORLD_ENV_WM_CKPT" "$VLA_RFT_WM_CKPT" "$INHERITED_SUITE_LABEL" "$RESOURCE_CHANGE_TIME_POINTS" "$RESOURCE_CHANGE_DIRECTIONS" "$RESOURCE_CHANGE_FACTORS"
import csv
import json
import sys
from pathlib import Path

tsv_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
suite_stamp = sys.argv[3]
monitor_interval = float(sys.argv[4])
plot_interval = float(sys.argv[5])
world_env_ckpt = sys.argv[6]
vla_rft_ckpt = sys.argv[7]
inherited_suite = sys.argv[8] or None
resource_change_time_points = sys.argv[9] or None
resource_change_directions = sys.argv[10] or None
resource_change_factors = sys.argv[11] or None

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
    "world_env_world_model_ckpt": world_env_ckpt,
    "vla_rft_world_model_ckpt": vla_rft_ckpt,
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
    -- python -u -m train.octo.plot_cl_suite \
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
