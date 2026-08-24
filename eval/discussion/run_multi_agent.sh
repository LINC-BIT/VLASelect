#!/usr/bin/env bash
set -euo pipefail

export ACCELERATE_USE_DEEPSPEED="${ACCELERATE_USE_DEEPSPEED:-false}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
EVAL_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${EVAL_ROOT}/common/interrupt_cleanup.sh"
source "${EVAL_ROOT}/common/sanity_check.sh"
source "${EVAL_ROOT}/common/mwe_time.sh"

STAMP=${MULTI_AGENT_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}
CUDA_DEVICES=${CUDA_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
OUTPUT_BASE=${MULTI_AGENT_OUTPUT_BASE:-${EVAL_ROOT}/discussion/results/multi_agent/${STAMP}}
MANIFEST_PATH=${MULTI_AGENT_MANIFEST_PATH:-${SCRIPT_DIR}/multi_agent_${STAMP}.json}
METHODS=${MULTI_AGENT_METHODS:-mappo,ours}
TAIL_LOG=${TAIL_LOG:-1}
MWE=${MWE:-0}

WRAPPER_ROOT="${EVAL_ROOT}/train/vla_adapter_smolvla/multi_agents/two_robot_pick"
MAPPO_WRAPPER="${WRAPPER_ROOT}/run_mappo_online_baseline.sh"
OURS_WRAPPER="${WRAPPER_ROOT}/run_mappo_online_wo_ag.sh"

USE_HF_MIRROR=${USE_HF_MIRROR:-1}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
MODEL_BACKBONE=${MULTI_AGENT_MODEL_BACKBONE:-mixed_tiny_vla_smolvla}
MODEL_DIR=${MULTI_AGENT_MODEL_DIR:-}
MAPPO_INIT_AGENT_PATH=${MULTI_AGENT_MAPPO_INIT_AGENT_PATH:-ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/latest_agent.pt}
OURS_INIT_AGENT_PATH=${MULTI_AGENT_OURS_INIT_AGENT_PATH:-ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/best_agent.pt}
NUM_ENVS=${MULTI_AGENT_NUM_ENVS:-128}
NUM_EVAL_ENVS=${MULTI_AGENT_NUM_EVAL_ENVS:-50}
ROLLOUT_STEPS=${MULTI_AGENT_ROLLOUT_STEPS:-16}
UPDATE_EPOCHS=${MULTI_AGENT_UPDATE_EPOCHS:-1}
NUM_MINIBATCH=${MULTI_AGENT_NUM_MINIBATCH:-16}
SAVE_INTERVAL_PER_ROLLOUT=${MULTI_AGENT_SAVE_INTERVAL_PER_ROLLOUT:-2}
CRITIC_WARMUP_ROLLOUTS=${MULTI_AGENT_CRITIC_WARMUP_ROLLOUTS:-0}
BACKBONE_LEARNING_RATE=${MULTI_AGENT_BACKBONE_LR:-1e-6}
HEAD_LEARNING_RATE=${MULTI_AGENT_HEAD_LR:-3e-6}
STATE_LEARNING_RATE=${MULTI_AGENT_STATE_LR:-3e-6}
VALUE_HEAD_LEARNING_RATE=${MULTI_AGENT_VALUE_HEAD_LR:-1e-4}
WEIGHT_DECAY=${MULTI_AGENT_WEIGHT_DECAY:-1e-6}
CLIP_EPS=${MULTI_AGENT_CLIP_EPS:-0.1}
TARGET_KL=${MULTI_AGENT_TARGET_KL:-0.05}
FULL_KL_COEF=${MULTI_AGENT_FULL_KL_COEF:-0.0}
USE_VLA_LORA=${MULTI_AGENT_USE_VLA_LORA:-0}
USE_VISION_LORA=${MULTI_AGENT_USE_VISION_LORA:-0}
TRAIN_VISION_BACKBONE=${MULTI_AGENT_TRAIN_VISION_BACKBONE:-0}
VISION_TOKEN_POOL_SIZE=${MULTI_AGENT_VISION_TOKEN_POOL_SIZE:-}
LORA_R=${MULTI_AGENT_LORA_R:-16}
LORA_ALPHA=${MULTI_AGENT_LORA_ALPHA:-16}
LORA_DROPOUT=${MULTI_AGENT_LORA_DROPOUT:-0.0}
RESUME_USE_BEST_AGENT=${MULTI_AGENT_RESUME_USE_BEST_AGENT:-0}
ENV_CHANGE_TIME_POINTS=${MULTI_AGENT_ENV_CHANGE_TIME_POINTS:-[31,61,91,121,151,181,211,241,271,301]}
EFFECTIVE_ENV_CHANGE_TIME_POINTS="$ENV_CHANGE_TIME_POINTS"
MAX_TIME=${MULTI_AGENT_MAX_TIME_MINUTES:-}
ATTN_IMPLEMENTATION=${MULTI_AGENT_ATTN_IMPLEMENTATION:-sdpa}
POLICY_MODE=${MULTI_AGENT_POLICY_MODE:-native}
FEATURE_SELECTOR_TOPK_TRAJECTORIES=${MULTI_AGENT_FEATURE_SELECTOR_TOPK_TRAJECTORIES:-4}

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
    EFFECTIVE_ENV_CHANGE_TIME_POINTS="$(vlaselect_convert_mwe_schedule_seconds_to_minutes "$ENV_CHANGE_TIME_POINTS")"
    NUM_ENVS=${MULTI_AGENT_NUM_ENVS:-16}
    NUM_EVAL_ENVS=${MULTI_AGENT_NUM_EVAL_ENVS:-4}
    UPDATE_EPOCHS=${MULTI_AGENT_UPDATE_EPOCHS:-1}
    NUM_MINIBATCH=${MULTI_AGENT_NUM_MINIBATCH:-4}
    SAVE_INTERVAL_PER_ROLLOUT=${MULTI_AGENT_SAVE_INTERVAL_PER_ROLLOUT:-1}
    MAX_TIME=${MULTI_AGENT_MAX_TIME_MINUTES:-4.5}
fi

vlaselect_install_cleanup_trap
vlaselect_run_sanity_check "run_multi_agent.sh" "$EVAL_ROOT" "$MWE" "50" "16"

mkdir -p "${OUTPUT_BASE}"
export PYTHONPATH="${EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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

require_file() {
    local path="$1"
    local label="$2"
    local resolved="$path"
    if [[ "$resolved" != /* ]]; then
        resolved="${EVAL_ROOT}/${resolved}"
    fi
    if [[ ! -f "$resolved" ]]; then
        echo "[error] ${label} not found: ${resolved}" >&2
        exit 1
    fi
}

for required_script in "$MAPPO_WRAPPER" "$OURS_WRAPPER"; do
    if [[ ! -f "$required_script" ]]; then
        echo "[error] missing launcher: $required_script" >&2
        exit 1
    fi
done

run_online_method() {
    local method="$1"
    local wrapper_script="$2"
    local init_agent_path="$3"
    local run_dir="$4"
    local log_file="$5"
    local result_json="$6"

    require_file "$init_agent_path" "${method} init agent"
    mkdir -p "$run_dir"

    local -a env_args=(
        "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
        "PYTHON_BIN=${PYTHON_BIN}"
        "USE_HF_MIRROR=${USE_HF_MIRROR}"
        "HF_ENDPOINT=${HF_ENDPOINT}"
        "INIT_AGENT_PATH=${init_agent_path}"
        "MODEL_BACKBONE=${MODEL_BACKBONE}"
        "MODEL_DIR=${MODEL_DIR}"
        "NUM_ENVS=${NUM_ENVS}"
        "NUM_EVAL_ENVS=${NUM_EVAL_ENVS}"
        "ROLLOUT_STEPS=${ROLLOUT_STEPS}"
        "UPDATE_EPOCHS=${UPDATE_EPOCHS}"
        "NUM_MINIBATCH=${NUM_MINIBATCH}"
        "SAVE_INTERVAL_PER_ROLLOUT=${SAVE_INTERVAL_PER_ROLLOUT}"
        "CRITIC_WARMUP_ROLLOUTS=${CRITIC_WARMUP_ROLLOUTS}"
        "BACKBONE_LEARNING_RATE=${BACKBONE_LEARNING_RATE}"
        "HEAD_LEARNING_RATE=${HEAD_LEARNING_RATE}"
        "STATE_LEARNING_RATE=${STATE_LEARNING_RATE}"
        "VALUE_HEAD_LEARNING_RATE=${VALUE_HEAD_LEARNING_RATE}"
        "WEIGHT_DECAY=${WEIGHT_DECAY}"
        "CLIP_EPS=${CLIP_EPS}"
        "TARGET_KL=${TARGET_KL}"
        "FULL_KL_COEF=${FULL_KL_COEF}"
        "USE_VLA_LORA=${USE_VLA_LORA}"
        "USE_VISION_LORA=${USE_VISION_LORA}"
        "TRAIN_VISION_BACKBONE=${TRAIN_VISION_BACKBONE}"
        "VISION_TOKEN_POOL_SIZE=${VISION_TOKEN_POOL_SIZE}"
        "LORA_R=${LORA_R}"
        "LORA_ALPHA=${LORA_ALPHA}"
        "LORA_DROPOUT=${LORA_DROPOUT}"
        "RESUME_DIR=${run_dir}"
        "RESUME_USE_BEST_AGENT=${RESUME_USE_BEST_AGENT}"
        "ENV_CHANGE_TIME_POINTS=${EFFECTIVE_ENV_CHANGE_TIME_POINTS}"
        "ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
        "POLICY_MODE=${POLICY_MODE}"
    )
    if [[ -n "$MAX_TIME" ]]; then
        env_args+=("MAX_TIME=${MAX_TIME}")
    fi
    if [[ "$method" == "ours" ]]; then
        env_args+=("FEATURE_SELECTOR_TOPK_TRAJECTORIES=${FEATURE_SELECTOR_TOPK_TRAJECTORIES}")
    fi

    echo "[run] method=${method} output=${run_dir}"
    if [[ "$TAIL_LOG" == "1" ]]; then
        if env "${env_args[@]}" bash "$wrapper_script" 2>&1 | tee "$log_file"; then
            append_run "$method" "$MODEL_BACKBONE" "$run_dir" "$log_file" "completed" "$result_json"
        else
            local status=$?
            echo "[warn] method=${method} exited with status=${status}; see ${log_file}"
            append_run "$method" "$MODEL_BACKBONE" "$run_dir" "$log_file" "failed" "$result_json"
        fi
    else
        if env "${env_args[@]}" bash "$wrapper_script" >"$log_file" 2>&1; then
            append_run "$method" "$MODEL_BACKBONE" "$run_dir" "$log_file" "completed" "$result_json"
        else
            local status=$?
            echo "[warn] method=${method} exited with status=${status}; see ${log_file}"
            append_run "$method" "$MODEL_BACKBONE" "$run_dir" "$log_file" "failed" "$result_json"
        fi
    fi
    write_manifest
}

IFS=',' read -r -a methods <<< "${METHODS}"
for method in "${methods[@]}"; do
    method=$(printf '%s' "${method}" | xargs)
    case "$method" in
        mappo)
            run_online_method "mappo" "$MAPPO_WRAPPER" "$MAPPO_INIT_AGENT_PATH" "${OUTPUT_BASE}/mappo" "${OUTPUT_BASE}/mappo/train.log" "${OUTPUT_BASE}/mappo/metrics.json"
            ;;
        ours)
            run_online_method "ours" "$OURS_WRAPPER" "$OURS_INIT_AGENT_PATH" "${OUTPUT_BASE}/ours" "${OUTPUT_BASE}/ours/train.log" "${OUTPUT_BASE}/ours/metrics.json"
            ;;
        '')
            ;;
        *)
            echo "[error] unsupported method: ${method}" >&2
            exit 1
            ;;
    esac
done

write_manifest
printf '[summary] manifest=%s\n' "${MANIFEST_PATH}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_multi_agent.py" --manifest "${MANIFEST_PATH}"
