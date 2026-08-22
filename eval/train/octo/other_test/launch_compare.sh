#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

SUITE_STAMP="${SUITE_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}"
SUITE_ROOT="ckpt/octo/other_test/${SUITE_STAMP}"
PLOTS_DIR="${SUITE_ROOT}/plots"
LAUNCH_LOG_DIR="${SUITE_ROOT}/launch_logs"
PID_DIR="${SUITE_ROOT}/pids"
MANIFEST_JSON="${SUITE_ROOT}/manifest.json"

COMPRESSED_RUN_DIR="${COMPRESSED_RUN_DIR:-ckpt/cl_suite/20260428-ours-ue2-edgeta06-convertnet-rerun/ppo_gen}"
PLOT_INTERVAL_SECONDS="${PLOT_INTERVAL_SECONDS:-100}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-30}"
ORIGINAL_GPU="${ORIGINAL_GPU:-2}"
PEFT_GPU="${PEFT_GPU:-3}"

mkdir -p "$SUITE_ROOT" "$PLOTS_DIR" "$LAUNCH_LOG_DIR" "$PID_DIR"

declare -a METHODS=(
    original
    original_peft
)

declare -A DISPLAY_NAMES=(
    [compressed]="PPO-Gen compressed"
    [original]="PPO-Gen original"
    [original_peft]="PPO-Gen original+PEFT"
)

declare -A SCRIPT_BY_METHOD=(
    [original]="train/octo/other_test/original_model/online_rl_original_model.sh"
    [original_peft]="train/octo/other_test/original_model_peft/online_rl_original_model_peft.sh"
)

declare -A GPU_BY_METHOD=(
    [original]="$ORIGINAL_GPU"
    [original_peft]="$PEFT_GPU"
)

declare -A EXP_NAME_BY_METHOD=(
    [original]="octo/other_test/${SUITE_STAMP}/original_model"
    [original_peft]="octo/other_test/${SUITE_STAMP}/original_model_peft"
)

declare -a METHOD_ROWS=()

compressed_row="$(python - "$COMPRESSED_RUN_DIR" "${DISPLAY_NAMES[compressed]}" <<'PY'
import json
import sys
print(json.dumps({
    "name": "compressed",
    "display_name": sys.argv[2],
    "gpu": None,
    "pid": None,
    "monitor_pid": None,
    "run_dir": sys.argv[1],
    "log_file": None,
    "script_path": None,
    "exp_name": None,
    "status": "inherited",
}))
PY
)"
METHOD_ROWS+=("$compressed_row")

for method in "${METHODS[@]}"; do
    gpu="${GPU_BY_METHOD[$method]}"
    script_path="${SCRIPT_BY_METHOD[$method]}"
    exp_name="${EXP_NAME_BY_METHOD[$method]}"
    run_dir="ckpt/${exp_name}"
    train_log="${LAUNCH_LOG_DIR}/${method}.train.log"
    pid_file="${PID_DIR}/${method}.pid"

    python /home/Maniskill/train/octo/spawn_detached.py \
        --pid-file "$pid_file" \
        --log-file "$train_log" \
        --cwd "$ROOT_DIR" \
        -- env \
        CUDA_DEVICES="$gpu" \
        EXP_NAME="$exp_name" \
        TAIL_LOG=0 \
        LAUNCH_DIRECT=1 \
        bash "$script_path" \
        > /dev/null
    train_pid="$(cat "$pid_file")"

    monitor_pid_file="${PID_DIR}/${method}.gpu_monitor.pid"
    monitor_log="${LAUNCH_LOG_DIR}/${method}.gpu_monitor.log"
    python /home/Maniskill/train/octo/spawn_detached.py \
        --pid-file "$monitor_pid_file" \
        --log-file "$monitor_log" \
        --cwd "$ROOT_DIR" \
        -- python -u -m train.octo.monitor_gpu_metrics \
        --pid "$train_pid" \
        --gpu-index "$gpu" \
        --run-dir "$run_dir" \
        --interval-seconds "$MONITOR_INTERVAL_SECONDS" \
        --label "$method" \
        > /dev/null
    monitor_pid="$(cat "$monitor_pid_file")"

    method_row="$(python - "$method" "${DISPLAY_NAMES[$method]}" "$gpu" "$train_pid" "$monitor_pid" "$run_dir" "$train_log" "$script_path" "$exp_name" <<'PY'
import json
import sys
print(json.dumps({
    "name": sys.argv[1],
    "display_name": sys.argv[2],
    "gpu": int(sys.argv[3]),
    "pid": int(sys.argv[4]),
    "monitor_pid": int(sys.argv[5]),
    "run_dir": sys.argv[6],
    "log_file": sys.argv[7],
    "script_path": sys.argv[8],
    "exp_name": sys.argv[9],
    "status": "launched",
}))
PY
)"
    METHOD_ROWS+=("$method_row")
done

python - "$MANIFEST_JSON" "$SUITE_STAMP" "$PLOT_INTERVAL_SECONDS" "$MONITOR_INTERVAL_SECONDS" "${METHOD_ROWS[@]}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
suite_stamp = sys.argv[2]
plot_interval_seconds = float(sys.argv[3])
monitor_interval_seconds = float(sys.argv[4])
methods = [json.loads(item) for item in sys.argv[5:]]

manifest = {
    "suite_stamp": suite_stamp,
    "plot_interval_seconds": plot_interval_seconds,
    "monitor_interval_seconds": monitor_interval_seconds,
    "methods": methods,
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

PLOTTER_PID_FILE="${PID_DIR}/plotter.pid"
PLOTTER_LOG="${LAUNCH_LOG_DIR}/plotter.log"
python /home/Maniskill/train/octo/spawn_detached.py \
    --pid-file "$PLOTTER_PID_FILE" \
    --log-file "$PLOTTER_LOG" \
    --cwd "$ROOT_DIR" \
    -- python -u -m train.octo.other_test.plot_compare \
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
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["plotter_pid"] = int(sys.argv[2])
manifest["plotter_log"] = sys.argv[3]
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

echo "Suite launched: ${SUITE_STAMP}"
echo "Manifest: ${MANIFEST_JSON}"
echo "Plots: ${PLOTS_DIR}"
echo "Plotter PID: ${PLOTTER_PID}"
