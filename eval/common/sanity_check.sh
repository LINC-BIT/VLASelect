#!/usr/bin/env bash

# Source this file from experiment entry scripts, or run it directly.

vlaselect_sanity_log() {
    echo "[sanity] $*"
}

vlaselect_sanity_default_ckpt_paths() {
    cat <<'PATHS_EOF'
eval/ckpt/edgevla/ours/outputs/bc_unitree_g1_lift_apple_fbs/20260511-171959/best_policy.pt
eval/ckpt/tinyvla/ours/outputs/bc_open_cabinet_drawer_fbs/20260508-032529/best_policy.pt
eval/ckpt/vla_adapter_new/ours/outputs/20260502-112804/best_policy.pt
eval/ckpt/edgevla/env_verify/outputs/ppo_unitree_g1_lift_apple/20260511-063605/best_policy.pt
eval/ckpt/tinyvla/model_impl/outputs/ppo_open_cabinet_drawer/20260507-113650/best_policy.pt
eval/ckpt/vla_adapter_new/model_impl/outputs/ppo_hold_cube_in_hand/20260430-103518/best_policy.pt
eval/ckpt/vla_adapter_new/LIBERO-Object/model.safetensors
eval/ckpt/vla_adapter_new/LIBERO-Object/action_head--checkpoint.pt
eval/ckpt/vla_adapter_new/LIBERO-Object/proprio_projector--checkpoint.pt
eval/ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/best_agent.pt
eval/ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/latest_agent.pt
eval/ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/latest_opt.pt
eval/ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt
eval/ckpt/PickCube-v1/baselines/world_env/world_model/20260425-032853-run032_e2/checkpoints/best_with_reference.pt
eval/ckpt/PickCube-v1/baselines/vla_rft/world_model/20260424-160154-run32x2/checkpoints/best.pt
eval/ckpt/vla_adapter_new/vla_rft/outputs/world_model/latest/best_world_model.pt
eval/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942/best_agent.pt
eval/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942/latest_agent.pt
eval/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942/best_ag_panda_wristcam-0.pt
eval/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942/best_ag_panda_wristcam-1.pt
eval/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942/latest_ag_panda_wristcam-0.pt
eval/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942/latest_ag_panda_wristcam-1.pt
PATHS_EOF
}

vlaselect_sanity_python_bin() {
    local candidate
    for candidate in "${PYTHON_BIN:-}" python3 python; do
        if [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

vlaselect_sanity_count_selected_models() {
    local raw_selection="${FAMILY_SELECTION:-${MODEL_SELECTION:-}}"
    if [[ -z "$raw_selection" ]]; then
        printf '4\n'
        return 0
    fi
    printf '%s' "$raw_selection" | tr ',' '\n' | awk 'NF {count += 1} END {print count + 0}'
}

vlaselect_sanity_checkpoint_list() {
    local repo_root="$1"
    if [[ -n "${VLASELECT_SANITY_CKPT_LIST:-}" && -f "${VLASELECT_SANITY_CKPT_LIST}" ]]; then
        cat "${VLASELECT_SANITY_CKPT_LIST}"
        return 0
    fi
    if [[ -f "${repo_root}/hf_ckpt_paths.txt" ]]; then
        cat "${repo_root}/hf_ckpt_paths.txt"
        return 0
    fi
    vlaselect_sanity_default_ckpt_paths
}

vlaselect_sanity_check_python_modules() {
    local py_bin="$1"
    "$py_bin" - <<'PY'
import importlib
import sys
required = ["torch", "pypdf", "matplotlib"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("[sanity] missing Python modules: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY
}

vlaselect_sanity_check_torch_cuda() {
    local py_bin="$1"
    if command -v nvidia-smi >/dev/null 2>&1; then
        if ! "$py_bin" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print('[sanity] torch.cuda.is_available() is False although nvidia-smi is present.', file=sys.stderr)
    sys.exit(1)
PY
        then
            echo "[sanity] Hint: verify the NVIDIA driver, nvidia-container-toolkit, and container GPU visibility." >&2
            return 1
        fi
    fi
}

vlaselect_sanity_check_paths() {
    local eval_root="$1"
    local repo_root="$2"
    local -a missing=()
    local rel_path

    [[ -d "${eval_root}/envs" ]] || missing+=("eval/envs")
    [[ -d "${eval_root}/workloads" ]] || missing+=("eval/workloads")
    [[ -d "${eval_root}/datasets" ]] || missing+=("eval/datasets")

    while IFS= read -r rel_path; do
        [[ -z "$rel_path" ]] && continue
        if [[ ! -f "${repo_root}/${rel_path}" ]]; then
            missing+=("${rel_path}")
        fi
    done < <(vlaselect_sanity_checkpoint_list "$repo_root")

    if [[ "${#missing[@]}" -gt 0 ]]; then
        echo "[sanity] missing required files or directories:" >&2
        local item
        local shown=0
        for item in "${missing[@]}"; do
            echo "[sanity]   - ${item}" >&2
            shown=$((shown + 1))
            if [[ "$shown" -ge 12 ]]; then
                local remaining=$(( ${#missing[@]} - shown ))
                if [[ "$remaining" -gt 0 ]]; then
                    echo "[sanity]   ... ${remaining} more missing entries omitted" >&2
                fi
                break
            fi
        done
        echo "[sanity] Re-run 'bash dep.sh' to pull the Docker image and bundled checkpoints first." >&2
        return 1
    fi
}

vlaselect_sanity_check_vram() {
    local required_vram_gb="$1"
    local experiment_name="$2"
    local mwe="$3"

    if [[ "${required_vram_gb}" -le 0 ]]; then
        return 0
    fi
    if [[ "${VLASELECT_ALLOW_LOW_VRAM:-0}" == "1" ]]; then
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        vlaselect_sanity_log "nvidia-smi not found; skipping VRAM check for ${experiment_name}"
        return 0
    fi

    local max_total_mb=0
    local max_free_mb=0
    local gpu_count=0
    local total_mb free_mb
    while IFS=, read -r total_mb free_mb; do
        total_mb=$(echo "$total_mb" | tr -d ' ')
        free_mb=$(echo "$free_mb" | tr -d ' ')
        [[ -z "$total_mb" || -z "$free_mb" ]] && continue
        gpu_count=$((gpu_count + 1))
        if (( total_mb > max_total_mb )); then
            max_total_mb=$total_mb
        fi
        if (( free_mb > max_free_mb )); then
            max_free_mb=$free_mb
        fi
    done < <(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null)

    if (( gpu_count == 0 )); then
        vlaselect_sanity_log "no visible GPUs reported by nvidia-smi; skipping VRAM check for ${experiment_name}"
        return 0
    fi

    local required_mb=$(( required_vram_gb * 1024 ))
    if (( max_total_mb < required_mb || max_free_mb < required_mb )); then
        echo "[sanity] visible GPU memory is below the recommended level for ${experiment_name}." >&2
        echo "[sanity] highest total VRAM: $((max_total_mb / 1024)) GB; highest free VRAM: $((max_free_mb / 1024)) GB; recommended: ${required_vram_gb} GB" >&2
        if [[ "$mwe" == "1" ]]; then
            echo "[sanity] Even the MWE may OOM on this machine. Reduce other GPU workloads or set VLASELECT_ALLOW_LOW_VRAM=1 to bypass." >&2
        else
            echo "[sanity] Use MWE=1 first, reduce MODEL_SELECTION/FAMILY_SELECTION, or set VLASELECT_ALLOW_LOW_VRAM=1 to bypass this guard." >&2
        fi
        return 1
    fi
}

vlaselect_sanity_warn_parallel_pressure() {
    local experiment_name="$1"
    local selected_count
    selected_count="$(vlaselect_sanity_count_selected_models)"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    local gpu_count
    gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
    [[ -z "$gpu_count" ]] && gpu_count=0
    if [[ "$selected_count" -gt "$gpu_count" && "$gpu_count" -gt 0 ]]; then
        echo "[sanity] warning: ${experiment_name} selected ${selected_count} model/workload families but only ${gpu_count} GPU(s) are visible." >&2
        echo "[sanity] The scripts will queue runs, but full experiments may still be slow or hit VRAM pressure on busy GPUs." >&2
    fi
}

vlaselect_run_sanity_check() {
    local experiment_name="$1"
    local eval_root="$2"
    local mwe="$3"
    local full_required_vram_gb="$4"
    local mwe_required_vram_gb="$5"

    if [[ "${VLASELECT_SKIP_SANITY_CHECK:-0}" == "1" ]]; then
        return 0
    fi

    local repo_root
    repo_root="$(cd "${eval_root}/.." && pwd)"
    local py_bin
    py_bin="$(vlaselect_sanity_python_bin)" || {
        echo "[sanity] missing Python interpreter (tried PYTHON_BIN, python3, python)." >&2
        return 1
    }

    vlaselect_sanity_log "running preflight for ${experiment_name}"
    vlaselect_sanity_check_python_modules "$py_bin"
    vlaselect_sanity_check_torch_cuda "$py_bin"
    vlaselect_sanity_check_paths "$eval_root" "$repo_root"

    local required_vram_gb="$full_required_vram_gb"
    if [[ "$mwe" == "1" ]]; then
        required_vram_gb="$mwe_required_vram_gb"
    fi
    vlaselect_sanity_check_vram "$required_vram_gb" "$experiment_name" "$mwe"
    vlaselect_sanity_warn_parallel_pressure "$experiment_name"
    vlaselect_sanity_log "preflight passed for ${experiment_name}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-manual-sanity-check}"
    EVAL_ROOT="${EVAL_ROOT:-$(cd -- "$(dirname -- "$0")/.." && pwd)}"
    MWE_FLAG="${MWE:-0}"
    FULL_REQUIRED_VRAM_GB="${FULL_REQUIRED_VRAM_GB:-16}"
    MWE_REQUIRED_VRAM_GB="${MWE_REQUIRED_VRAM_GB:-8}"

    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --experiment)
                EXPERIMENT_NAME="$2"
                shift 2
                ;;
            --eval-root)
                EVAL_ROOT="$2"
                shift 2
                ;;
            --mwe)
                MWE_FLAG="$2"
                shift 2
                ;;
            --required-vram-gb)
                FULL_REQUIRED_VRAM_GB="$2"
                shift 2
                ;;
            --required-vram-gb-mwe)
                MWE_REQUIRED_VRAM_GB="$2"
                shift 2
                ;;
            *)
                echo "Unknown argument: $1" >&2
                exit 1
                ;;
        esac
    done

    vlaselect_run_sanity_check "$EXPERIMENT_NAME" "$EVAL_ROOT" "$MWE_FLAG" "$FULL_REQUIRED_VRAM_GB" "$MWE_REQUIRED_VRAM_GB"
fi
