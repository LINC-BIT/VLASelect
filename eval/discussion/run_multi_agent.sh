#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
EVAL_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${EVAL_ROOT}/common/interrupt_cleanup.sh"
STAMP=${MULTI_AGENT_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}
CUDA_DEVICES=${CUDA_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-/home/Maniskill/.venvs/internvl_qwen3/bin/python}
MANISKILL_ROOT=${MANISKILL_ROOT:-/home/Maniskill}
EXPERT_AGENT_DIR=${MULTI_AGENT_EXPERT_AGENT_DIR:-${EVAL_ROOT}/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942}
TRAJECTORY_H5_PATH=${MULTI_AGENT_TRAJECTORY_H5_PATH:-${MANISKILL_ROOT}/datasets/TwoRobotPickCube-v2/rl/trajectory.rgb+state_dict.pd_joint_delta_pos.physx_cuda.h5}
OUTPUT_BASE=${MULTI_AGENT_OUTPUT_BASE:-${EVAL_ROOT}/discussion/results/multi_agent/${STAMP}}
MANIFEST_PATH=${MULTI_AGENT_MANIFEST_PATH:-${SCRIPT_DIR}/multi_agent_${STAMP}.json}
METHODS=${MULTI_AGENT_METHODS:-mappo,ours}
MODEL_SELECTION="${MODEL_SELECTION:-}"

NUM_SUCCESSFUL_TRAJECTORIES=${MULTI_AGENT_NUM_SUCCESSFUL_TRAJECTORIES:-32}
NUM_COLLECT_ENVS=${MULTI_AGENT_NUM_COLLECT_ENVS:-64}
SFT_TOTAL_ITERS=${MULTI_AGENT_SFT_TOTAL_ITERS:-8}
BATCH_SIZE=${MULTI_AGENT_BATCH_SIZE:-8}
LOG_INTERVAL_ITERS=${MULTI_AGENT_LOG_INTERVAL_ITERS:-4}
STATE_STATS_BATCH_SIZE=${MULTI_AGENT_STATE_STATS_BATCH_SIZE:-32}
EVAL_INTERVAL_ITERS=${MULTI_AGENT_EVAL_INTERVAL_ITERS:-4}
EVAL_EPISODES=${MULTI_AGENT_EVAL_EPISODES:-4}
NUM_EVAL_ENVS=${MULTI_AGENT_NUM_EVAL_ENVS:-2}
IMAGE_SIZE=${MULTI_AGENT_IMAGE_SIZE:-112}
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
if [[ "$MWE" == "1" ]]; then
    NUM_SUCCESSFUL_TRAJECTORIES=${MULTI_AGENT_NUM_SUCCESSFUL_TRAJECTORIES:-4}
    NUM_COLLECT_ENVS=${MULTI_AGENT_NUM_COLLECT_ENVS:-8}
    SFT_TOTAL_ITERS=${MULTI_AGENT_SFT_TOTAL_ITERS:-2}
    BATCH_SIZE=${MULTI_AGENT_BATCH_SIZE:-4}
    LOG_INTERVAL_ITERS=${MULTI_AGENT_LOG_INTERVAL_ITERS:-1}
    STATE_STATS_BATCH_SIZE=${MULTI_AGENT_STATE_STATS_BATCH_SIZE:-8}
    EVAL_INTERVAL_ITERS=${MULTI_AGENT_EVAL_INTERVAL_ITERS:-1}
    EVAL_EPISODES=${MULTI_AGENT_EVAL_EPISODES:-2}
    NUM_EVAL_ENVS=${MULTI_AGENT_NUM_EVAL_ENVS:-1}
fi
vlaselect_install_cleanup_trap
TAIL_LOG=${TAIL_LOG:-1}

mkdir -p "${OUTPUT_BASE}"
export PYTHONPATH="${EVAL_ROOT}:${MANISKILL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ ! -f "${TRAJECTORY_H5_PATH}" ]; then
    echo "[error] trajectory file not found: ${TRAJECTORY_H5_PATH}" >&2
    exit 1
fi
if [ ! -d "${EXPERT_AGENT_DIR}" ]; then
    echo "[error] expert agent dir not found: ${EXPERT_AGENT_DIR}" >&2
    exit 1
fi

TRAIN_SCRIPT="${EVAL_ROOT}/train/multi_agents/two_robot_pick/bc_pretrain.py"
MAPPO_EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_multi_agent_mappo.py"
declare -a RUN_ROWS=()

append_run() {
    RUN_ROWS+=("$1|$2|$3|$4|$5|$6")
}

write_manifest() {
    MANIFEST_PATH="$MANIFEST_PATH" STAMP="$STAMP" OUTPUT_BASE="$OUTPUT_BASE" python - "${RUN_ROWS[@]}" <<'PY_MANIFEST'
import json
import os
import sys

runs = []
for line in sys.argv[1:]:
    if not line:
        continue
    method, model_backbone, run_dir, log_file, status, result_json = line.split('|', 5)
    runs.append({
        'method': method,
        'model_backbone': model_backbone,
        'run_dir': run_dir,
        'log_file': log_file,
        'status': status,
        'result_json': result_json,
    })

manifest = {
    'stamp': os.environ['STAMP'],
    'output_base': os.environ['OUTPUT_BASE'],
    'runs': runs,
}
with open(os.environ['MANIFEST_PATH'], 'w', encoding='utf-8') as handle:
    json.dump(manifest, handle, indent=2)
    handle.write(chr(10))
PY_MANIFEST
}

run_mappo() {
    local method="mappo"
    local run_dir="${OUTPUT_BASE}/${method}"
    local log_file="${run_dir}/eval.log"
    local result_json="${run_dir}/eval_metrics.json"
    mkdir -p "${run_dir}"
    local -a cmd=(
        "${PYTHON_BIN}" "${MAPPO_EVAL_SCRIPT}"
        --task-name TwoRobotPickCube-v2
        --expert-agent-dir "${EXPERT_AGENT_DIR}"
        --obs-mode rgb+state_dict
        --control-mode pd_joint_delta_pos
        --reward-mode normalized_dense
        --max-episode-steps 100
        --num-eval-envs "${NUM_EVAL_ENVS}"
        --eval-episodes "${EVAL_EPISODES}"
        --output-json "${result_json}"
    )

    echo "[run] method=${method} output=${run_dir}"
    if [[ "$TAIL_LOG" == "1" ]]; then
        if CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}" 2>&1 | tee "${log_file}"; then
            append_run "${method}" "mappo" "${run_dir}" "${log_file}" "completed" "${result_json}"
        else
            local status=$?
            echo "[warn] method=${method} exited with status=${status}; see ${log_file}"
            append_run "${method}" "mappo" "${run_dir}" "${log_file}" "failed" "${result_json}"
        fi
    else
        if CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}" >"${log_file}" 2>&1; then
            append_run "${method}" "mappo" "${run_dir}" "${log_file}" "completed" "${result_json}"
        else
            local status=$?
            echo "[warn] method=${method} exited with status=${status}; see ${log_file}"
            append_run "${method}" "mappo" "${run_dir}" "${log_file}" "failed" "${result_json}"
        fi
    fi
    write_manifest
}

run_ours() {
    local method="ours"
    local model_backbone="multi_agents"
    local run_dir="${OUTPUT_BASE}/${method}"
    local log_file="${run_dir}/train.log"
    local dataset_cache_path="${run_dir}/expert_sft_dataset.pt"
    local result_json="${run_dir}/metrics.json"

    mkdir -p "${run_dir}"
    local -a cmd=(
        "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
        --task-name TwoRobotPickCube-v2
        --save-dir "${OUTPUT_BASE}"
        --resume-dir "${run_dir}"
        --model-backbone "${model_backbone}"
        --image-size "${IMAGE_SIZE}"
        --expert-agent-dir "${EXPERT_AGENT_DIR}"
        --trajectory-h5-path "${TRAJECTORY_H5_PATH}"
        --dataset-cache-path "${dataset_cache_path}"
        --obs-mode rgb+state_dict
        --control-mode pd_joint_delta_pos
        --reward-mode normalized_dense
        --num-successful-trajectories "${NUM_SUCCESSFUL_TRAJECTORIES}"
        --num-collect-envs "${NUM_COLLECT_ENVS}"
        --sft-total-iters "${SFT_TOTAL_ITERS}"
        --batch-size "${BATCH_SIZE}"
        --log-interval-iters "${LOG_INTERVAL_ITERS}"
        --state-stats-batch-size "${STATE_STATS_BATCH_SIZE}"
        --eval-interval-iters "${EVAL_INTERVAL_ITERS}"
        --eval-episodes "${EVAL_EPISODES}"
        --num-eval-envs "${NUM_EVAL_ENVS}"
        --normalize-state
        --use-amp
        --policy-mode native
        --vision-token-pool-size 16
        --backbone-learning-rate 1e-4
        --head-learning-rate 2e-4
        --state-learning-rate 2e-4
        --attn-implementation sdpa
        --tiny-hidden-dim 640
        --tiny-vision-layers 7
        --tiny-decoder-layers 8
        --tiny-attention-heads 10
        --tiny-patch-size 14
        --tiny-ffn-mult 4
        --tiny-num-action-bins 256
        --tiny-prompt-length 24
    )

    echo "[run] method=${method} backbone=${model_backbone} output=${run_dir}"
    if [[ "$TAIL_LOG" == "1" ]]; then
        if CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}" 2>&1 | tee "${log_file}"; then
            append_run "${method}" "${model_backbone}" "${run_dir}" "${log_file}" "completed" "${result_json}"
        else
            local status=$?
            echo "[warn] method=${method} exited with status=${status}; see ${log_file}"
            append_run "${method}" "${model_backbone}" "${run_dir}" "${log_file}" "failed" "${result_json}"
        fi
    else
        if CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}" >"${log_file}" 2>&1; then
            append_run "${method}" "${model_backbone}" "${run_dir}" "${log_file}" "completed" "${result_json}"
        else
            local status=$?
            echo "[warn] method=${method} exited with status=${status}; see ${log_file}"
            append_run "${method}" "${model_backbone}" "${run_dir}" "${log_file}" "failed" "${result_json}"
        fi
    fi
    write_manifest
}

IFS=',' read -r -a methods <<< "${METHODS}"
for method in "${methods[@]}"; do
    method=$(printf '%s' "${method}" | xargs)
    case "${method}" in
        mappo) run_mappo ;;
        ours) run_ours ;;
        '') ;;
        *) echo "[error] unsupported method: ${method}" >&2; exit 1 ;;
    esac
done

write_manifest

printf '[summary] manifest=%s\n' "${MANIFEST_PATH}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_multi_agent.py" --manifest "${MANIFEST_PATH}"
