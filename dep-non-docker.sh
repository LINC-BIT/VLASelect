#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HOST_ARCH=${HOST_ARCH:-$(uname -m)}
VENV_DIR=${VENV_DIR:-$ROOT_DIR/.venv-nondocker}
PYTHON_BIN=${PYTHON_BIN:-python3}
REQ_IN=${REQ_IN:-$ROOT_DIR/eval/requirements.txt}
REQ_FILTERED=${REQ_FILTERED:-/tmp/vlaselect-requirements.filtered.txt}

TORCH_VERSION=${TORCH_VERSION:-2.4.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.19.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.4.0}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}
TORCH_INDEX_URL_FALLBACKS=${TORCH_INDEX_URL_FALLBACKS:-}
PIP_INDEX_URL_DEFAULT=${PIP_INDEX_URL_DEFAULT:-https://pypi.org/simple}
PIP_INDEX_URL_FALLBACKS=${PIP_INDEX_URL_FALLBACKS:-https://pypi.tuna.tsinghua.edu.cn/simple}
PIP_EXTRA_INDEX_URL_DEFAULT=${PIP_EXTRA_INDEX_URL_DEFAULT:-}
MANISKILL_VERSION=${MANISKILL_VERSION:-3.0.0b22}
DEEPSPEED_VERSION=0.15.0

INSTALL_SYSTEM_DEPS=${INSTALL_SYSTEM_DEPS:-0}
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-0}
DOWNLOAD_CKPTS=${DOWNLOAD_CKPTS:-1}
ARM=${ARM:-0}

HF_CKPT_REPO=${HF_CKPT_REPO:-cz22edd/vlaselect_test}
HF_CKPT_REPO_TYPE=${HF_CKPT_REPO_TYPE:-model}
HF_CKPT_REVISION=${HF_CKPT_REVISION:-main}
HF_CKPT_LIST=${HF_CKPT_LIST:-}
HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}

log() {
    echo "[dep-non-docker.sh] $*"
}

iter_fallback_list() {
    local raw=${1:-}
    local item
    for item in $raw; do
        [[ -n "$item" ]] && printf '%s\n' "$item"
    done
}

iter_pip_indexes() {
    printf '%s\n' "$PIP_INDEX_URL_DEFAULT"
    iter_fallback_list "$PIP_INDEX_URL_FALLBACKS"
}

iter_torch_indexes() {
    printf '%s\n' "$TORCH_INDEX_URL"
    iter_fallback_list "$TORCH_INDEX_URL_FALLBACKS"
}

pip_args_for_index() {
    local index_url=$1
    local -a args
    args=(--index-url "$index_url")
    if [[ -n "$PIP_EXTRA_INDEX_URL_DEFAULT" ]]; then
        args+=(--extra-index-url "$PIP_EXTRA_INDEX_URL_DEFAULT")
    fi
    printf '%s\n' "${args[@]}"
}

run_pip_install_with_fallbacks() {
    local label=$1
    shift

    local index_url
    local -A seen=()
    local -a pip_args

    while IFS= read -r index_url; do
        [[ -z "$index_url" ]] && continue
        [[ -n "${seen[$index_url]:-}" ]] && continue
        seen["$index_url"]=1
        mapfile -t pip_args < <(pip_args_for_index "$index_url")
        log "trying ${label} via ${index_url}"
        if python -m pip install "${pip_args[@]}" "$@"; then
            return 0
        fi
        log "warning: ${label} failed via ${index_url}"
    done < <(iter_pip_indexes)

    return 1
}

run_torch_install_with_fallbacks() {
    local index_url
    local -A seen=()

    while IFS= read -r index_url; do
        [[ -z "$index_url" ]] && continue
        [[ -n "${seen[$index_url]:-}" ]] && continue
        seen["$index_url"]=1
        log "trying PyTorch ${TORCH_VERSION} via ${index_url}"
        if python -m pip install \
            "torch==${TORCH_VERSION}" \
            "torchvision==${TORCHVISION_VERSION}" \
            "torchaudio==${TORCHAUDIO_VERSION}" \
            --index-url "$index_url"; then
            return 0
        fi
        log "warning: PyTorch install failed via ${index_url}"
    done < <(iter_torch_indexes)

    return 1
}


die() {
    echo "[dep-non-docker.sh] $*" >&2
    exit 1
}

require_cmd() {
    local cmd=$1
    local hint=$2
    if ! command -v "$cmd" >/dev/null 2>&1; then
        die "missing required command: $cmd. $hint"
    fi
}

ensure_python() {
    require_cmd "$PYTHON_BIN" "Install Python first, then rerun this script."
    "$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required.")
PY
}

configure_arch_mode() {
    if [[ "$ARM" == "1" ]]; then
        log "ARM=1 detected; enabling ARM-compatible non-Docker mode for host arch ${HOST_ARCH}."
        if [[ "${TORCH_INDEX_URL}" == "https://download.pytorch.org/whl/cu124" ]]; then
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
            log "ARM mode switched the default PyTorch index to CPU wheels. Override TORCH_INDEX_URL manually if your ARM device needs a different official wheel index."
        fi
        INSTALL_FLASH_ATTN=0
        return
    fi

    case "$HOST_ARCH" in
        aarch64|arm64)
            log "ARM host detected (${HOST_ARCH}). Use ARM=1 to enable ARM-specific non-Docker defaults."
            ;;
    esac
}

install_system_packages() {
    if [[ "$INSTALL_SYSTEM_DEPS" != "1" ]]; then
        return
    fi

    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y git libvulkan1
    else
        apt-get update
        apt-get install -y git libvulkan1
    fi
}

create_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        log "creating virtual environment: $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi

    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"

    if ! run_pip_install_with_fallbacks "bootstrap packages" --upgrade pip setuptools wheel; then
        log "warning: failed to upgrade pip/setuptools/wheel from all configured package indexes; continuing with the venv-bundled versions."
    fi
}

install_torch() {
    log "installing PyTorch ${TORCH_VERSION}"
    run_torch_install_with_fallbacks || die "failed to install PyTorch from all configured torch indexes."
}

install_maniskill() {
    log "installing ManiSkill ${MANISKILL_VERSION}"
    run_pip_install_with_fallbacks "ManiSkill ${MANISKILL_VERSION}" "mani_skill==${MANISKILL_VERSION}"         || die "failed to install ManiSkill from all configured package indexes."
}

install_deepspeed() {
    log "installing DeepSpeed ${DEEPSPEED_VERSION}"
    run_pip_install_with_fallbacks "DeepSpeed ${DEEPSPEED_VERSION}" "deepspeed==${DEEPSPEED_VERSION}" \
        || die "failed to install DeepSpeed from all configured package indexes."
}

build_filtered_requirements() {
    [[ -f "$REQ_IN" ]] || die "requirements file not found: $REQ_IN"
    log "building filtered requirements: $REQ_FILTERED"

    REQ_IN="$REQ_IN" \
    REQ_FILTERED="$REQ_FILTERED" \
    INSTALL_FLASH_ATTN="$INSTALL_FLASH_ATTN" \
    python - <<'PY'
from pathlib import Path
import os
import re

req_in = Path(os.environ["REQ_IN"])
req_out = Path(os.environ["REQ_FILTERED"])
install_flash_attn = os.environ["INSTALL_FLASH_ATTN"] == "1"

skip_exact = {
    "torch",
    "torchvision",
    "torchaudio",
    "torchelastic",
    "mani_skill",
    "flash_attn",
    "deepspeed",
    "conda",
    "conda-build",
    "conda-content-trust",
    "conda-libmamba-solver",
    "conda-package-handling",
    "conda_package_streaming",
    "conda_index",
    "anaconda-anon-usage",
    "archspec",
    "libmambapy",
    "menuinst",
}

if install_flash_attn:
    skip_exact.discard("flash_attn")

lines_out = []
for raw in req_in.read_text().splitlines():
    line = raw.strip()
    if not line:
        continue
    if line.startswith("#"):
        lines_out.append(raw)
        continue
    if "@ file://" in line:
        continue
    if "githubfast.com/kvablack/dlimp@" in line:
        line = line.replace("https://githubfast.com/kvablack/dlimp@", "https://github.com/kvablack/dlimp@")

    name = re.split(r"[<>=!~@ ]", line, maxsplit=1)[0].strip()
    normalized = name.lower()
    if normalized in skip_exact:
        continue
    lines_out.append(line)

req_out.write_text("\n".join(lines_out) + "\n")
PY
}

install_filtered_requirements() {
    log "installing filtered Python dependencies from $REQ_FILTERED"
    run_pip_install_with_fallbacks "filtered requirements" -r "$REQ_FILTERED"         || die "failed to install filtered requirements from all configured package indexes."
    run_pip_install_with_fallbacks "runtime helper packages" -U "huggingface_hub[cli]" pypdf pin noise         || die "failed to install runtime helper packages from all configured package indexes."
}

default_hf_ckpt_paths() {
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
eval/ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/best_agent.pt.base
eval/ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/latest_agent.pt
eval/ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/latest_opt.pt
eval/ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt
eval/ckpt/PickCube-v1/baselines/world_env/world_model/20260425-032853-run032_e2/checkpoints/best_with_reference.pt
eval/ckpt/PickCube-v1/baselines/vla_rft/world_model/20260424-160154-run32x2/checkpoints/best.pt
eval/ckpt/vla_adapter_new/world_env/outputs/world_model/20260503-075340/checkpoints/best_with_reference.pt
eval/ckpt/vla_adapter_new/vla_rft/outputs/world_model/latest/best_world_model.pt
eval/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942/best_agent.pt
eval/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942/latest_agent.pt
eval/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942/best_ag_panda_wristcam-0.pt
eval/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942/best_ag_panda_wristcam-1.pt
eval/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942/latest_ag_panda_wristcam-0.pt
eval/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942/latest_ag_panda_wristcam-1.pt
PATHS_EOF
}

iter_hf_ckpt_paths() {
    if [[ -n "$HF_CKPT_LIST" ]]; then
        [[ -f "$HF_CKPT_LIST" ]] || die "checkpoint list file not found: $HF_CKPT_LIST"
        cat "$HF_CKPT_LIST"
        return
    fi

    if [[ -f "$ROOT_DIR/hf_ckpt_paths.txt" ]]; then
        cat "$ROOT_DIR/hf_ckpt_paths.txt"
        return
    fi

    default_hf_ckpt_paths
}

download_hf_checkpoints() {
    if [[ "$DOWNLOAD_CKPTS" != "1" || -z "$HF_CKPT_REPO" ]]; then
        return
    fi

    export HF_HUB_DOWNLOAD_TIMEOUT
    log "downloading checkpoints from Hugging Face repo: $HF_CKPT_REPO"

    local -a cmd
    cmd=(hf download "$HF_CKPT_REPO" --repo-type "$HF_CKPT_REPO_TYPE" --revision "$HF_CKPT_REVISION" --local-dir "$ROOT_DIR")
    while IFS= read -r rel_path; do
        [[ -z "$rel_path" ]] && continue
        cmd+=(--include "$rel_path")
    done < <(iter_hf_ckpt_paths)

    if [[ -n "${HF_TOKEN:-}" ]]; then
        cmd+=(--token "$HF_TOKEN")
    fi
    "${cmd[@]}"
}

run_post_checks() {
    log "running post-install checks"
    python - <<'PY'
import torch
print("[dep-non-docker.sh] torch.__version__ =", torch.__version__)
print("[dep-non-docker.sh] torch.cuda.is_available() =", torch.cuda.is_available())
PY

    if python -c 'import mani_skill' >/dev/null 2>&1; then
        log "ManiSkill import check passed."
    fi
}

print_summary() {
    cat <<EOF
[dep-non-docker.sh] environment preparation is complete.
[dep-non-docker.sh] repo root             : $ROOT_DIR
[dep-non-docker.sh] host arch            : $HOST_ARCH
[dep-non-docker.sh] ARM mode             : $ARM
[dep-non-docker.sh] virtualenv           : $VENV_DIR
[dep-non-docker.sh] torch index          : $TORCH_INDEX_URL
[dep-non-docker.sh] torch fallbacks      : ${TORCH_INDEX_URL_FALLBACKS:-<none>}
[dep-non-docker.sh] default pip index    : $PIP_INDEX_URL_DEFAULT
[dep-non-docker.sh] pip fallbacks        : ${PIP_INDEX_URL_FALLBACKS:-<none>}
[dep-non-docker.sh] ManiSkill version    : $MANISKILL_VERSION
[dep-non-docker.sh] DeepSpeed version    : $DEEPSPEED_VERSION
[dep-non-docker.sh] filtered requirements: $REQ_FILTERED

Next commands:
  source "$VENV_DIR/bin/activate"
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  python -m mani_skill.examples.demo_random_action
  bash eval/common/sanity_check.sh --experiment non-docker --eval-root "$ROOT_DIR/eval"

Notes:
  - PyTorch is installed separately using the configured PyTorch wheel index list.
  - General Python packages are installed from PyPI first, then retried against the configured fallback mirrors.
  - ManiSkill is installed separately before the filtered project requirements.
  - The filtered requirements file skips conda-local entries and DeepSpeed; DeepSpeed is installed separately at version 0.15.0.
  - Set INSTALL_SYSTEM_DEPS=1 to auto-install Ubuntu packages such as git and libvulkan1.
  - Set INSTALL_FLASH_ATTN=1 if you explicitly want the optional flash_attn package.
  - For ARM-based hosts, run the script with ARM=1 to switch the default PyTorch source to the official CPU wheels while keeping optional flash_attn disabled.
EOF
}

main() {
    require_cmd git "Install git first so pip can fetch git-based dependencies such as dlimp."
    ensure_python
    configure_arch_mode
    install_system_packages
    create_venv
    install_torch
    install_maniskill
    install_deepspeed
    build_filtered_requirements
    install_filtered_requirements
    download_hf_checkpoints
    run_post_checks
    print_summary
}

main "$@"
