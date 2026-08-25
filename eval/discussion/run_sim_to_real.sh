#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="${EVAL_ROOT}/sim_to_real"
source "${EVAL_ROOT}/common/interrupt_cleanup.sh"
source "${EVAL_ROOT}/common/resource_summary.sh"

SUITE_STAMP="${SUITE_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}"
RESULT_ROOT="${SIM_ROOT}/sim_to_real_results/${SUITE_STAMP}"
LAUNCH_LOG_DIR="${RESULT_ROOT}/launch_logs"
VIDEO_OUTPUT_DIR="${RESULT_ROOT}/videos"
SIM_TO_REAL_PLATFORM="${SIM_TO_REAL_PLATFORM:-dofbot_se}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_SELECTION="${MODEL_SELECTION:-}"
MWE="${MWE:-0}"

: "${MWE_RUNTIME_LIMIT_SECONDS:=300}"
export MWE_RUNTIME_LIMIT_SECONDS
if [[ "$MWE" == "1" && "${MWE_TIMEOUT_APPLIED:-0}" != "1" ]]; then
    if command -v timeout >/dev/null 2>&1; then
        export MWE_TIMEOUT_APPLIED=1
        exec timeout --preserve-status -k 10s "${MWE_RUNTIME_LIMIT_SECONDS}s" bash "$SCRIPT_PATH" "$@"
    fi
    echo "[warn] timeout command not found; MWE runtime is not hard-capped" >&2
fi
vlaselect_resource_summary_start "run_sim_to_real.sh"
vlaselect_install_cleanup_trap

case "${SIM_TO_REAL_PLATFORM}" in
    dofbot_se)
        PLATFORM_ROOT="${SIM_ROOT}/dofbot_se"
        ENTRY_DIR="${PLATFORM_ROOT}"
        ENTRY_SCRIPT="main.py"
        VIDEO_SOURCE_DIR="${PLATFORM_ROOT}/video"
        ;;
    amazinghand)
        PLATFORM_ROOT="${SIM_ROOT}/amazinghand"
        ENTRY_DIR="${PLATFORM_ROOT}/AmazingHand-main/PythonExample"
        ENTRY_SCRIPT="main.py"
        VIDEO_SOURCE_DIR="${PLATFORM_ROOT}/video"
        ;;
    *)
        echo "Unsupported SIM_TO_REAL_PLATFORM: ${SIM_TO_REAL_PLATFORM}" >&2
        echo "Supported values: dofbot_se, amazinghand" >&2
        exit 1
        ;;
esac

RUN_LOG="${LAUNCH_LOG_DIR}/${SIM_TO_REAL_PLATFORM}.log"

mkdir -p "${RESULT_ROOT}" "${LAUNCH_LOG_DIR}" "${VIDEO_OUTPUT_DIR}"
printf "%s\n" "${SUITE_STAMP}" > "${SIM_ROOT}/sim_to_real_results/latest.txt"

echo "[sim-to-real] output directory: ${RESULT_ROOT}"
echo "[sim-to-real] log file: ${RUN_LOG}"
echo "[sim-to-real] target platform: ${SIM_TO_REAL_PLATFORM}"
echo "[sim-to-real] entrypoint: ${ENTRY_DIR}/${ENTRY_SCRIPT}"
if [[ "${MWE}" == "1" ]]; then
    echo "[sim-to-real] MWE=1 uses the same real-robot dispatch path as the full run"
fi

if [[ ! -d "${ENTRY_DIR}" ]]; then
    echo "Sim-to-real entry directory not found: ${ENTRY_DIR}" >&2
    exit 1
fi

if [[ ! -f "${ENTRY_DIR}/${ENTRY_SCRIPT}" ]]; then
    echo "Sim-to-real entrypoint not found: ${ENTRY_DIR}/${ENTRY_SCRIPT}" >&2
    exit 1
fi

echo "[sim-to-real] starting ${SIM_TO_REAL_PLATFORM} real-robot evaluation"

if [[ -d "${VIDEO_SOURCE_DIR}" ]]; then
    find "${VIDEO_SOURCE_DIR}" -maxdepth 1 -type f -printf "%f\n" | sort > "${RESULT_ROOT}/video_files_before.txt"
else
    : > "${RESULT_ROOT}/video_files_before.txt"
fi

(
    cd "${ENTRY_DIR}"
    "${PYTHON_BIN}" -u "${ENTRY_SCRIPT}"
) 2>&1 | tee "${RUN_LOG}"

if [[ -d "${VIDEO_SOURCE_DIR}" ]]; then
    find "${VIDEO_SOURCE_DIR}" -maxdepth 1 -type f -printf "%f\n" | sort > "${RESULT_ROOT}/video_files_after.txt"
    while IFS= read -r filename; do
        [[ -z "${filename}" ]] && continue
        if ! grep -Fxq "${filename}" "${RESULT_ROOT}/video_files_before.txt"; then
            cp "${VIDEO_SOURCE_DIR}/${filename}" "${VIDEO_OUTPUT_DIR}/${filename}"
        fi
    done < "${RESULT_ROOT}/video_files_after.txt"
fi

cat <<EOF2
[sim-to-real] finished
- run log: ${RUN_LOG}
- copied videos: ${VIDEO_OUTPUT_DIR}

Figure 13 and Table 4 are not summarized automatically.
Please inspect the robot execution results and record the final sim-to-real statistics manually.
EOF2
