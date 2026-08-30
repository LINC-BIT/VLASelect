#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONTAINER_NAME=${CONTAINER_NAME:-vlaselect-ae}
TYPE=${TYPE:-default}
DOCKER_IMAGE_FULL=${DOCKER_IMAGE_FULL:-cz22edd/pytorch:maniskillv2}
DOCKER_IMAGE_100M=${DOCKER_IMAGE_100M:-cz22edd/pytorch:maniskillv2-100m}
DOCKER_IMAGE=${DOCKER_IMAGE:-}
HOST_REPO_DIR=${HOST_REPO_DIR:-$ROOT_DIR}
CONTAINER_REPO_DIR=${CONTAINER_REPO_DIR:-$ROOT_DIR}
START_SCRIPT_PATH=${START_SCRIPT_PATH:-$ROOT_DIR/start_docker.sh}
CONTAINER_VENV_DIR=${CONTAINER_VENV_DIR:-/opt/vlaselect-venv}
SMALL_IMAGE_SENTINEL=${SMALL_IMAGE_SENTINEL:-/opt/vlaselect-venv/.vlaselect-ready}
SHM_SIZE=${SHM_SIZE:-32g}
DOCKER_NETWORK=${DOCKER_NETWORK:-host}
DOCKER_IPC=${DOCKER_IPC:-host}
DOCKER_PID=${DOCKER_PID:-host}
PULL_POLICY=${PULL_POLICY:-always}
RECREATE=${RECREATE:-0}
HF_CKPT_REPO=${HF_CKPT_REPO:-cz22edd/vlaselect_test}
HF_CKPT_REPO_TYPE=${HF_CKPT_REPO_TYPE:-model}
HF_CKPT_REVISION=${HF_CKPT_REVISION:-main}
HF_CKPT_LIST=${HF_CKPT_LIST:-}
HF_MANISKILL_DATA_REPO=${HF_MANISKILL_DATA_REPO:-$HF_CKPT_REPO}
HF_MANISKILL_DATA_REPO_TYPE=${HF_MANISKILL_DATA_REPO_TYPE:-$HF_CKPT_REPO_TYPE}
HF_MANISKILL_DATA_REVISION=${HF_MANISKILL_DATA_REVISION:-$HF_CKPT_REVISION}
HF_MANISKILL_DATA_LIST=${HF_MANISKILL_DATA_LIST:-}
DOWNLOAD_MANISKILL_DATA=${DOWNLOAD_MANISKILL_DATA:-1}
CONTAINER_HOME=${CONTAINER_HOME:-}
CONTAINER_MS_ASSET_DIR=${CONTAINER_MS_ASSET_DIR:-}
CONTAINER_MANISKILL_DATA_DIR=${CONTAINER_MANISKILL_DATA_DIR:-}
CONTAINER_PARTNET_DATA_DIR=${CONTAINER_PARTNET_DATA_DIR:-}
HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
HF_HUB_MAX_WORKERS=${HF_HUB_MAX_WORKERS:-8}
HF_HUB_HTTP_MAX_RETRIES=${HF_HUB_HTTP_MAX_RETRIES:-6}
HF_HUB_RETRY_BASE_SECONDS=${HF_HUB_RETRY_BASE_SECONDS:-15}
HF_HUB_FILE_DELAY_SECONDS=${HF_HUB_FILE_DELAY_SECONDS:-0.2}
DEEPSPEED_VERSION=0.15.0
PYPDF_VERSION=${PYPDF_VERSION:-6.16.2}
PIN_VERSION=${PIN_VERSION:-2.7.0}
NOISE_VERSION=${NOISE_VERSION:-1.2.2}

log() {
    echo "[dep.sh] $*"
}

require_cmd() {
    local cmd=$1
    local hint=$2
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "[dep.sh] missing required command: $cmd" >&2
        echo "[dep.sh] $hint" >&2
        exit 1
    fi
}

attempt_host_python_pip_install() {
    local python_bin=$1
    if ! command -v apt-get >/dev/null 2>&1; then
        return 1
    fi

    log "attempting to install python3-pip via apt-get"
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y python3-pip
    else
        apt-get update && apt-get install -y python3-pip
    fi

    "$python_bin" -m pip --version >/dev/null 2>&1
}

ensure_python_pip() {
    local python_bin=$1
    if "$python_bin" -m pip --version >/dev/null 2>&1; then
        return
    fi

    log "$python_bin is available, but the pip module is missing; attempting bootstrap via ensurepip"
    if "$python_bin" -m ensurepip --upgrade >/dev/null 2>&1; then
        if "$python_bin" -m pip --version >/dev/null 2>&1; then
            log "bootstrapped pip for $python_bin via ensurepip"
            "$python_bin" -m pip install --upgrade pip >/dev/null 2>&1 || true
            return
        fi
    fi

    if attempt_host_python_pip_install "$python_bin"; then
        log "installed python3-pip for $python_bin via apt-get"
        "$python_bin" -m pip install --upgrade pip >/dev/null 2>&1 || true
        return
    fi

    echo "[dep.sh] $python_bin is available, but the pip module is missing and automatic bootstrap via ensurepip/apt-get failed." >&2
    echo "[dep.sh] Install python3-pip (or an equivalent pip package for $python_bin) and rerun this script." >&2
    exit 1
}

resolve_docker_image() {
    if [[ -n "$DOCKER_IMAGE" ]]; then
        return
    fi

    case "$TYPE" in
        default|6G|full|FULL)
            DOCKER_IMAGE="$DOCKER_IMAGE_FULL"
            ;;
        100M|small|SMALL)
            DOCKER_IMAGE="$DOCKER_IMAGE_100M"
            ;;
        *)
            echo "[dep.sh] unsupported TYPE: $TYPE" >&2
            echo "[dep.sh] supported values: default, 6G, full, 100M, small" >&2
            exit 1
            ;;
    esac
}

is_small_image_type() {
    case "$TYPE" in
        100M|small|SMALL)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_container_paths() {
    if [[ -z "$CONTAINER_HOME" ]]; then
        local image_home
        image_home=$(docker image inspect "$DOCKER_IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | awk -F= '$1=="HOME" {print substr($0, 6); exit}')
        CONTAINER_HOME=${image_home:-/root}
    fi

    if [[ -z "$CONTAINER_MS_ASSET_DIR" ]]; then
        CONTAINER_MS_ASSET_DIR="$CONTAINER_HOME/.maniskill"
    fi

    if [[ -z "$CONTAINER_MANISKILL_DATA_DIR" ]]; then
        CONTAINER_MANISKILL_DATA_DIR="$CONTAINER_MS_ASSET_DIR/data"
    fi

    if [[ -z "$CONTAINER_PARTNET_DATA_DIR" ]]; then
        # Align PartNet-Mobility with ManiSkill default runtime asset lookup.
        CONTAINER_PARTNET_DATA_DIR="$CONTAINER_MS_ASSET_DIR/data"
    fi
}

container_has_small_image_runtime() {
    docker exec "$CONTAINER_NAME" bash -lc "[ -f '$SMALL_IMAGE_SENTINEL' ]" >/dev/null 2>&1
}

bootstrap_small_image_environment() {
    local started_here=0
    local status=0

    if [[ -z $(docker ps --format '{{.Names}}' | grep -Fx "$CONTAINER_NAME" || true) ]]; then
        log "starting TYPE=$TYPE container temporarily to install the remaining runtime"
        docker start "$CONTAINER_NAME" >/dev/null
        started_here=1
    fi

    if container_has_small_image_runtime; then
        log "TYPE=$TYPE container runtime is already bootstrapped"
    else
        log "bootstrapping TYPE=$TYPE container runtime via dep-non-docker.sh"
        docker exec \
            -e VENV_DIR="$CONTAINER_VENV_DIR" \
            -e DOWNLOAD_CKPTS=0 \
            -e INSTALL_SYSTEM_DEPS=0 \
            -e DEEPSPEED_VERSION="$DEEPSPEED_VERSION" \
            -e HF_CKPT_REPO= \
            "$CONTAINER_NAME" \
            bash -lc "cd '$CONTAINER_REPO_DIR' && bash dep-non-docker.sh" || status=$?

        if [[ "$status" -ne 0 ]]; then
            if [[ "$started_here" == "1" ]]; then
                log "stopping container after failed TYPE=$TYPE bootstrap"
                docker stop "$CONTAINER_NAME" >/dev/null || true
            fi
            return "$status"
        fi

        docker exec "$CONTAINER_NAME" bash -lc "mkdir -p '$(dirname "$SMALL_IMAGE_SENTINEL")' && touch '$SMALL_IMAGE_SENTINEL'"
    fi

    if [[ "$started_here" == "1" ]]; then
        log "stopping container after TYPE=$TYPE runtime bootstrap"
        docker stop "$CONTAINER_NAME" >/dev/null
    fi
}

check_nvidia_runtime() {
    if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q 'nvidia'; then
        cat >&2 <<'MSG'
[dep.sh] docker does not report the nvidia runtime.
[dep.sh] Please install and configure nvidia-container-toolkit first:
[dep.sh] https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
MSG
        exit 1
    fi
}

pull_image() {
    if [[ "$PULL_POLICY" == "always" ]]; then
        log "pulling Docker image: $DOCKER_IMAGE"
        docker pull "$DOCKER_IMAGE"
        return
    fi

    if docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        log "Docker image already exists locally: $DOCKER_IMAGE"
    else
        log "pulling Docker image: $DOCKER_IMAGE"
        docker pull "$DOCKER_IMAGE"
    fi
}

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"
}

remove_existing_container() {
    if [[ "$RECREATE" != "1" ]]; then
        return
    fi

    if container_exists; then
        log "removing existing container: $CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME" >/dev/null
    fi
}

ensure_container() {
    if container_exists; then
        log "reusing existing container: $CONTAINER_NAME"
        return
    fi

    create_container
}

default_hf_ckpt_paths() {
    cat <<'PATHS_EOF'
eval/ckpt/edgevla/ours/outputs/bc_unitree_g1_lift_apple_fbs/20260511-171959/best_policy.pt
eval/ckpt/edgevla/ours/outputs/bc_unitree_g1_lift_apple_fbs/20260511-171959/best_policy.pt.base
eval/ckpt/tinyvla/ours/outputs/bc_open_cabinet_drawer_fbs/20260508-032529/best_policy.pt
eval/ckpt/tinyvla/ours/outputs/bc_open_cabinet_drawer_fbs/20260508-032529/best_policy.pt.base
eval/ckpt/vla_adapter_new/ours/outputs/20260502-112804/best_policy.pt
eval/ckpt/vla_adapter_new/ours/outputs/20260502-112804/best_policy.pt.base
eval/ckpt/edgevla/env_verify/outputs/ppo_unitree_g1_lift_apple/20260511-063605/best_policy.pt
eval/ckpt/tinyvla/model_impl/outputs/ppo_open_cabinet_drawer/20260507-113650/best_policy.pt
eval/ckpt/vla_adapter_new/model_impl/outputs/ppo_hold_cube_in_hand/20260430-103518/best_policy.pt
eval/ckpt/vla_adapter_new/LIBERO-Object/model.safetensors
eval/ckpt/vla_adapter_new/LIBERO-Object/action_head--checkpoint.pt
eval/ckpt/vla_adapter_new/LIBERO-Object/proprio_projector--checkpoint.pt
eval/ckpt/vla_adapter_new/LIBERO-Object/added_tokens.json
eval/ckpt/vla_adapter_new/LIBERO-Object/config.json
eval/ckpt/vla_adapter_new/LIBERO-Object/configuration_prismatic.py
eval/ckpt/vla_adapter_new/LIBERO-Object/dataset_statistics.json
eval/ckpt/vla_adapter_new/LIBERO-Object/generation_config.json
eval/ckpt/vla_adapter_new/LIBERO-Object/merges.txt
eval/ckpt/vla_adapter_new/LIBERO-Object/modeling_prismatic.py
eval/ckpt/vla_adapter_new/LIBERO-Object/preprocessor_config.json
eval/ckpt/vla_adapter_new/LIBERO-Object/processing_prismatic.py
eval/ckpt/vla_adapter_new/LIBERO-Object/processor_config.json
eval/ckpt/vla_adapter_new/LIBERO-Object/special_tokens_map.json
eval/ckpt/vla_adapter_new/LIBERO-Object/tokenizer.json
eval/ckpt/vla_adapter_new/LIBERO-Object/tokenizer_config.json
eval/ckpt/vla_adapter_new/LIBERO-Object/vocab.json
eval/ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/best_agent.pt
eval/ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/best_agent.pt.base
eval/ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/latest_agent.pt
eval/ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/latest_opt.pt
eval/ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt
eval/ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth
eval/ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260409-153956-feature_aggregator_lr3e-5_dual_stream_action_gate_reg_0_h4_2layergate_none/[agent1]/checkpoints copy/best_success_end.pt
eval/ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260409-153956-feature_aggregator_lr3e-5_dual_stream_action_gate_reg_0_h4_2layergate_none/[agent1]/checkpoints copy/best_success_end.pt.feature_aggregators
eval/ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260409-153956-feature_aggregator_lr3e-5_dual_stream_action_gate_reg_0_h4_2layergate_none/[agent2]/checkpoints copy/best_success_end.pt
eval/ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260409-153956-feature_aggregator_lr3e-5_dual_stream_action_gate_reg_0_h4_2layergate_none/[agent2]/checkpoints copy/best_success_end.pt.feature_aggregators
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
        if [[ ! -f "$HF_CKPT_LIST" ]]; then
            echo "[dep.sh] checkpoint list file not found: $HF_CKPT_LIST" >&2
            exit 1
        fi
        cat "$HF_CKPT_LIST"
        return
    fi
    default_hf_ckpt_paths
}

default_hf_maniskill_data_paths() {
    cat <<'PATHS_EOF'
partnet_mobility/**
assets.zip
PATHS_EOF
}

iter_hf_maniskill_data_paths() {
    if [[ -n "$HF_MANISKILL_DATA_LIST" ]]; then
        if [[ ! -f "$HF_MANISKILL_DATA_LIST" ]]; then
            echo "[dep.sh] ManiSkill data list file not found: $HF_MANISKILL_DATA_LIST" >&2
            exit 1
        fi
        cat "$HF_MANISKILL_DATA_LIST"
        return
    fi

    if [[ -f "$ROOT_DIR/hf_maniskill_data_paths.txt" ]]; then
        cat "$ROOT_DIR/hf_maniskill_data_paths.txt"
        return
    fi

    default_hf_maniskill_data_paths
}

require_hf_downloader() {
    if command -v hf >/dev/null 2>&1; then
        log "found Hugging Face CLI: $(command -v hf)"
        return
    fi

    require_cmd python3 "Install Python 3 or the Hugging Face CLI on the host machine."
    ensure_python_pip python3
    if python3 -c 'import huggingface_hub, requests' >/dev/null 2>&1; then
        log "found Python packages: huggingface_hub, requests"
        return
    fi

    log "installing missing Hugging Face downloader dependencies via pip"
    python3 -m pip install -U "huggingface_hub[cli]" requests

    if command -v hf >/dev/null 2>&1; then
        log "found Hugging Face CLI after installation: $(command -v hf)"
        return
    fi

    if python3 -c 'import huggingface_hub, requests' >/dev/null 2>&1; then
        log "found Python packages after installation: huggingface_hub, requests"
        return
    fi

    echo "[dep.sh] failed to install Hugging Face downloader dependencies automatically." >&2
    echo '[dep.sh] Please install them manually: python3 -m pip install -U "huggingface_hub[cli]" requests' >&2
    exit 1
}

download_hf_checkpoints() {
    if [[ -z "$HF_CKPT_REPO" ]]; then
        return
    fi

    require_hf_downloader
    export HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_MAX_WORKERS HF_HUB_HTTP_MAX_RETRIES HF_HUB_RETRY_BASE_SECONDS HF_HUB_FILE_DELAY_SECONDS

    log "downloading checkpoints from Hugging Face repo: $HF_CKPT_REPO"
    if [[ -n "$HF_CKPT_LIST" ]]; then
        log "checkpoint list: $HF_CKPT_LIST"
    else
        log "checkpoint list: built into dep.sh"
    fi

    if command -v hf >/dev/null 2>&1; then
        local -a cmd
        cmd=(hf download "$HF_CKPT_REPO" --repo-type "$HF_CKPT_REPO_TYPE" --revision "$HF_CKPT_REVISION" --local-dir "$HOST_REPO_DIR")
        while IFS= read -r rel_path; do
            [[ -z "$rel_path" ]] && continue
            cmd+=(--include "$rel_path")
        done < <(iter_hf_ckpt_paths)
        if [[ -n "${HF_TOKEN:-}" ]]; then
            cmd+=(--token "$HF_TOKEN")
        fi
        "${cmd[@]}"
        return
    fi

    local patterns_json
    patterns_json=$(iter_hf_ckpt_paths | python3 -c 'import json, sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')

    HF_CKPT_REPO="$HF_CKPT_REPO" \
    HF_CKPT_REPO_TYPE="$HF_CKPT_REPO_TYPE" \
    HF_CKPT_REVISION="$HF_CKPT_REVISION" \
    HF_CKPT_LOCAL_DIR="$HOST_REPO_DIR" \
    HF_CKPT_PATTERNS_JSON="$patterns_json" \
    HF_HUB_HTTP_MAX_RETRIES="$HF_HUB_HTTP_MAX_RETRIES" \
    HF_HUB_RETRY_BASE_SECONDS="$HF_HUB_RETRY_BASE_SECONDS" \
    HF_HUB_FILE_DELAY_SECONDS="$HF_HUB_FILE_DELAY_SECONDS" \
    HF_TOKEN="${HF_TOKEN:-}" \
    python3 - <<'PY'
import fnmatch
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from requests.exceptions import HTTPError

repo_id = os.environ['HF_CKPT_REPO']
repo_type = os.environ['HF_CKPT_REPO_TYPE']
revision = os.environ['HF_CKPT_REVISION']
local_dir = Path(os.environ['HF_CKPT_LOCAL_DIR'])
patterns = json.loads(os.environ['HF_CKPT_PATTERNS_JSON'])
max_retries = max(1, int(os.environ.get('HF_HUB_HTTP_MAX_RETRIES', '6')))
retry_base_seconds = max(1.0, float(os.environ.get('HF_HUB_RETRY_BASE_SECONDS', '15')))
file_delay_seconds = max(0.0, float(os.environ.get('HF_HUB_FILE_DELAY_SECONDS', '0.2')))
token = os.environ.get('HF_TOKEN') or None
api = HfApi(token=token)


def log_progress(message):
    print(f'[dep.sh] {message}', flush=True)


def with_429_retry(label, action):
    for attempt in range(1, max_retries + 1):
        try:
            return action()
        except HTTPError as exc:
            status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status_code != 429 or attempt >= max_retries:
                raise
            wait_seconds = min(300.0, retry_base_seconds * (2 ** (attempt - 1)))
            print(
                f'[dep.sh] Hugging Face returned 429 while downloading {label}; retrying in {wait_seconds:.1f}s',
                flush=True,
            )
            time.sleep(wait_seconds)


def resolve_paths():
    wildcard_patterns = [pattern for pattern in patterns if any(ch in pattern for ch in '*?[')]
    exact_paths = [pattern for pattern in patterns if pattern not in wildcard_patterns]
    resolved_paths = []
    seen = set()

    for rel_path in exact_paths:
        if rel_path not in seen:
            resolved_paths.append(rel_path)
            seen.add(rel_path)

    if not wildcard_patterns:
        return resolved_paths

    def iter_tree():
        return list(api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, revision=revision, recursive=True))

    log_progress('listing checkpoint files from Hugging Face repo tree')
    for entry in with_429_retry('checkpoint file listing', iter_tree):
        entry_type = getattr(entry, 'type', None)
        rel_path = getattr(entry, 'path', '')
        entry_size = getattr(entry, 'size', None)
        if entry_type != 'file' and entry_size is None:
            continue
        if not rel_path or rel_path.endswith('/'):
            continue
        if any(fnmatch.fnmatch(rel_path, pattern) for pattern in wildcard_patterns) and rel_path not in seen:
            resolved_paths.append(rel_path)
            seen.add(rel_path)
    log_progress(f'resolved {len(resolved_paths)} checkpoint files for download')
    return resolved_paths


def download_file(rel_path):
    target_path = local_dir / rel_path
    if target_path.is_file():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)

    def run_download():
        return hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            filename=rel_path,
            local_dir=local_dir,
            token=token,
        )

    with_429_retry(rel_path, run_download)
    if file_delay_seconds > 0:
        time.sleep(file_delay_seconds)


resolved_paths = resolve_paths()
for index, rel_path in enumerate(resolved_paths, start=1):
    if index == 1 or index % 20 == 0 or index == len(resolved_paths):
        log_progress(f'checkpoint download progress: {index}/{len(resolved_paths)} ({rel_path})')
    download_file(rel_path)
PY
}

ensure_default_state_norm_stats() {
    local target="$ROOT_DIR/eval/ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth"
    local source="$ROOT_DIR/eval/train/octo/ours/PickCube-v1-state-max-min.pth"

    if [[ -f "$target" || ! -f "$source" ]]; then
        return
    fi

    log "staging fallback state norm stats: $target"
    mkdir -p "$(dirname "$target")"
    cp "$source" "$target"
}


ensure_container_maniskill_dirs() {
    docker exec "$CONTAINER_NAME" bash -lc \
        "mkdir -p '$CONTAINER_MANISKILL_DATA_DIR' '$CONTAINER_PARTNET_DATA_DIR'"
}

download_hf_maniskill_data() {
    if [[ "$DOWNLOAD_MANISKILL_DATA" != "1" || -z "$HF_MANISKILL_DATA_REPO" ]]; then
        return
    fi

    local started_here=0
    local status=0
    local patterns_json

    patterns_json=$(iter_hf_maniskill_data_paths | python3 -c 'import json, sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')

    if [[ -z $(docker ps --format '{{.Names}}' | grep -Fx "$CONTAINER_NAME" || true) ]]; then
        log "starting container temporarily to download ManiSkill assets"
        docker start "$CONTAINER_NAME" >/dev/null
        started_here=1
    fi

    if ! docker exec "$CONTAINER_NAME" bash -lc "python -c 'import huggingface_hub, requests'" >/dev/null 2>&1; then
        log "installing container Python packages: huggingface_hub, requests"
        docker exec "$CONTAINER_NAME" bash -lc "python -m pip install -U 'huggingface_hub' requests" || status=$?
    fi

    if [[ "$status" -eq 0 ]]; then
        ensure_container_maniskill_dirs || status=$?
    fi

    if [[ "$status" -eq 0 ]]; then
        log "downloading ManiSkill assets from Hugging Face repo: $HF_MANISKILL_DATA_REPO"
        log "target directory for regular ManiSkill data: $CONTAINER_MANISKILL_DATA_DIR (inside container)"
        log "target directory for PartNet-Mobility: $CONTAINER_PARTNET_DATA_DIR (inside container)"
        if [[ -n "$HF_MANISKILL_DATA_LIST" ]]; then
            log "ManiSkill data list: $HF_MANISKILL_DATA_LIST"
        elif [[ -f "$ROOT_DIR/hf_maniskill_data_paths.txt" ]]; then
            log "ManiSkill data list: $ROOT_DIR/hf_maniskill_data_paths.txt"
        else
            log "ManiSkill data list: built into dep.sh"
        fi

        docker exec             -e HF_MANISKILL_DATA_REPO="$HF_MANISKILL_DATA_REPO"             -e HF_MANISKILL_DATA_REPO_TYPE="$HF_MANISKILL_DATA_REPO_TYPE"             -e HF_MANISKILL_DATA_REVISION="$HF_MANISKILL_DATA_REVISION"             -e HF_MANISKILL_DATA_LOCAL_DIR="$CONTAINER_MANISKILL_DATA_DIR"             -e HF_MANISKILL_PARTNET_LOCAL_DIR="$CONTAINER_PARTNET_DATA_DIR"             -e HF_MANISKILL_DATA_PATTERNS_JSON="$patterns_json"             -e HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT"             -e HF_HUB_HTTP_MAX_RETRIES="$HF_HUB_HTTP_MAX_RETRIES"             -e HF_HUB_RETRY_BASE_SECONDS="$HF_HUB_RETRY_BASE_SECONDS"             -e HF_HUB_FILE_DELAY_SECONDS="$HF_HUB_FILE_DELAY_SECONDS"             -e HF_TOKEN             -e HTTP_PROXY             -e HTTPS_PROXY             -e ALL_PROXY             -e NO_PROXY             "$CONTAINER_NAME"             bash -lc "python - <<'PY_HF_DATA'
import fnmatch
import json
import os
import time
from pathlib import Path
import zipfile

from huggingface_hub import HfApi, hf_hub_download
from requests.exceptions import HTTPError

repo_id = os.environ['HF_MANISKILL_DATA_REPO']
repo_type = os.environ['HF_MANISKILL_DATA_REPO_TYPE']
revision = os.environ['HF_MANISKILL_DATA_REVISION']
regular_local_dir = Path(os.environ['HF_MANISKILL_DATA_LOCAL_DIR'])
partnet_local_dir = Path(os.environ['HF_MANISKILL_PARTNET_LOCAL_DIR'])
patterns = json.loads(os.environ['HF_MANISKILL_DATA_PATTERNS_JSON'])
max_retries = max(1, int(os.environ.get('HF_HUB_HTTP_MAX_RETRIES', '6')))
retry_base_seconds = max(1.0, float(os.environ.get('HF_HUB_RETRY_BASE_SECONDS', '15')))
file_delay_seconds = max(0.0, float(os.environ.get('HF_HUB_FILE_DELAY_SECONDS', '0.2')))
token = os.environ.get('HF_TOKEN') or None
api = HfApi(token=token)


def log_progress(message):
    print(f'[dep.sh] {message}', flush=True)


def with_429_retry(label, action):
    for attempt in range(1, max_retries + 1):
        try:
            return action()
        except HTTPError as exc:
            status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status_code != 429 or attempt >= max_retries:
                raise
            wait_seconds = min(300.0, retry_base_seconds * (2 ** (attempt - 1)))
            print(
                f'[dep.sh] Hugging Face returned 429 while downloading {label}; retrying in {wait_seconds:.1f}s',
                flush=True,
            )
            time.sleep(wait_seconds)


repo_file_paths = None


def get_repo_file_paths():
    global repo_file_paths
    if repo_file_paths is not None:
        return repo_file_paths

    def iter_tree():
        return list(api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, revision=revision, recursive=True))

    repo_file_paths = []
    log_progress('listing ManiSkill files from Hugging Face repo tree')
    for entry in with_429_retry('ManiSkill file listing', iter_tree):
        entry_type = getattr(entry, 'type', None)
        rel_path = getattr(entry, 'path', '')
        entry_size = getattr(entry, 'size', None)
        if entry_type != 'file' and entry_size is None:
            continue
        if not rel_path or rel_path.endswith('/'):
            continue
        repo_file_paths.append(rel_path)
    log_progress(f'ManiSkill repo tree listing finished: {len(repo_file_paths)} files discovered')
    return repo_file_paths


def resolve_paths(allow_patterns):
    if not allow_patterns:
        return []

    file_paths = get_repo_file_paths()
    wildcard_patterns = [pattern for pattern in allow_patterns if any(ch in pattern for ch in '*?[')]
    exact_paths = [pattern for pattern in allow_patterns if pattern not in wildcard_patterns]
    resolved_paths = []
    seen = set()

    for rel_path in exact_paths:
        if rel_path in file_paths:
            if rel_path not in seen:
                resolved_paths.append(rel_path)
                seen.add(rel_path)
            continue

        prefix = rel_path.rstrip('/') + '/'
        prefix_matches = [path for path in file_paths if path.startswith(prefix)]
        if prefix_matches:
            for match in prefix_matches:
                if match not in seen:
                    resolved_paths.append(match)
                    seen.add(match)
            continue

        raise RuntimeError(f'No Hugging Face files matched path or directory prefix: {rel_path}')

    for rel_path in file_paths:
        if any(fnmatch.fnmatch(rel_path, pattern) for pattern in wildcard_patterns) and rel_path not in seen:
            resolved_paths.append(rel_path)
            seen.add(rel_path)
    return resolved_paths


def download_files(allow_patterns, local_dir, label):
    resolved_paths = resolve_paths(allow_patterns)
    log_progress(f'{label}: matched {len(resolved_paths)} files')
    for index, rel_path in enumerate(resolved_paths, start=1):
        target_path = local_dir / rel_path
        if target_path.is_file():
            if index == 1 or index % 50 == 0 or index == len(resolved_paths):
                log_progress(f'{label}: {index}/{len(resolved_paths)} already present ({rel_path})')
            continue
        if index == 1 or index % 50 == 0 or index == len(resolved_paths):
            log_progress(f'{label}: downloading {index}/{len(resolved_paths)} ({rel_path})')
        target_path.parent.mkdir(parents=True, exist_ok=True)

        def run_download():
            return hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                filename=rel_path,
                local_dir=local_dir,
                token=token,
            )

        with_429_retry(f'{label}: {rel_path}', run_download)
        if file_delay_seconds > 0:
            time.sleep(file_delay_seconds)


partnet_patterns = [pattern for pattern in patterns if pattern == 'partnet_mobility' or pattern.startswith('partnet_mobility/')]
regular_patterns = [pattern for pattern in patterns if pattern not in partnet_patterns]
partnet_archive_patterns = [pattern for pattern in partnet_patterns if pattern.lower().endswith('.zip')]
partnet_regular_patterns = [pattern for pattern in partnet_patterns if not pattern.lower().endswith('.zip')]
archive_patterns = [pattern for pattern in regular_patterns if pattern.lower().endswith('.zip')]
regular_patterns = [pattern for pattern in regular_patterns if not pattern.lower().endswith('.zip')]

download_files(regular_patterns, regular_local_dir, 'regular ManiSkill assets')
download_files(archive_patterns, regular_local_dir, 'regular ManiSkill archives')
download_files(partnet_regular_patterns, partnet_local_dir, 'PartNet-Mobility assets')
download_files(partnet_archive_patterns, partnet_local_dir, 'PartNet-Mobility archives')


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or '..' in member_path.parts:
                raise RuntimeError(f'unsafe zip entry: {member.filename}')
            resolved = (target_dir / member_path).resolve()
            target_root = target_dir.resolve()
            if target_root not in resolved.parents and resolved != target_root:
                raise RuntimeError(f'zip entry escapes target dir: {member.filename}')
        zf.extractall(target_dir)


for archive_pattern, archive_dir in [
    *[(pattern, regular_local_dir) for pattern in archive_patterns],
    *[(pattern, partnet_local_dir) for pattern in partnet_archive_patterns],
]:
    for archive_path in sorted(archive_dir.glob(archive_pattern)):
        if not archive_path.is_file():
            continue
        safe_extract_zip(archive_path, archive_dir)
        archive_path.unlink()
PY_HF_DATA" || status=$?
    fi

    if [[ "$started_here" == "1" ]]; then
        log "stopping container after ManiSkill asset download"
        docker stop "$CONTAINER_NAME" >/dev/null || true
    fi

    if [[ "$status" -ne 0 ]]; then
        return "$status"
    fi
}

ensure_container_python_package() {
    local module_name="$1"
    local package_spec="$2"

    if docker exec "$CONTAINER_NAME" bash -lc "python -c 'import ${module_name}'" >/dev/null 2>&1; then
        log "container Python package already available: $package_spec"
        return
    fi

    log "installing container Python package: $package_spec"
    docker exec "$CONTAINER_NAME" bash -lc "python -m pip install -U '$package_spec'"
}

install_container_deepspeed() {
    log "installing container DeepSpeed ${DEEPSPEED_VERSION}"
    docker exec "$CONTAINER_NAME" bash -lc "python -m pip install -U 'deepspeed==${DEEPSPEED_VERSION}'"
}

install_container_runtime_dependencies() {
    if is_small_image_type; then
        bootstrap_small_image_environment
        return
    fi

    local started_here=0

    if [[ -z $(docker ps --format '{{.Names}}' | grep -Fx "$CONTAINER_NAME" || true) ]]; then
        log "starting container temporarily to install runtime Python packages"
        docker start "$CONTAINER_NAME" >/dev/null
        started_here=1
    fi

    ensure_container_python_package pypdf "pypdf==${PYPDF_VERSION}"
    ensure_container_python_package pinocchio "pin==${PIN_VERSION}"
    ensure_container_python_package noise "noise==${NOISE_VERSION}"
    install_container_deepspeed

    if [[ "$started_here" == "1" ]]; then
        log "stopping container after dependency installation"
        docker stop "$CONTAINER_NAME" >/dev/null
    fi
}

create_container() {
    log "creating container: $CONTAINER_NAME"
    docker create \
        --name "$CONTAINER_NAME" \
        --gpus all \
        --network "$DOCKER_NETWORK" \
        --ipc "$DOCKER_IPC" \
        --pid "$DOCKER_PID" \
        --shm-size "$SHM_SIZE" \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e PYTHONUNBUFFERED=1 \
        -e VLASELECT_REPO_DIR="$CONTAINER_REPO_DIR" \
        -e VLASELECT_VENV_DIR="$CONTAINER_VENV_DIR" \
        -e MS_ASSET_DIR="$CONTAINER_MS_ASSET_DIR" \
        -v "$HOST_REPO_DIR:$CONTAINER_REPO_DIR" \
        -w "$CONTAINER_REPO_DIR" \
        "$DOCKER_IMAGE" \
        sleep infinity >/dev/null
}

generate_start_script() {
    log "generating start script: $START_SCRIPT_PATH"
    cat > "$START_SCRIPT_PATH" <<START_EOF
#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME=${CONTAINER_NAME@Q}
CONTAINER_REPO_DIR=${CONTAINER_REPO_DIR@Q}
CONTAINER_VENV_DIR=${CONTAINER_VENV_DIR@Q}

if ! command -v docker >/dev/null 2>&1; then
    echo "[start_docker.sh] docker is not installed or not in PATH." >&2
    exit 1
fi

if ! docker ps -a --format '{{.Names}}' | grep -Fxq "\$CONTAINER_NAME"; then
    echo "[start_docker.sh] container '\$CONTAINER_NAME' does not exist." >&2
    echo "[start_docker.sh] Please run 'bash dep.sh' first." >&2
    exit 1
fi

if [[ -z \$(docker ps --format '{{.Names}}' | grep -Fx "\$CONTAINER_NAME" || true) ]]; then
    echo "[start_docker.sh] starting container '\$CONTAINER_NAME'"
    docker start "\$CONTAINER_NAME" >/dev/null
fi

exec docker exec -it \
    -e VLASELECT_IN_CONTAINER=1 \
    -e PS1='(vlaselect-container) \u@\h:\w\$ ' \
    "\$CONTAINER_NAME" \
    bash -lc "printf '%s\n' '==== VLASelect Docker container ====' 'Container: \$CONTAINER_NAME' 'Workdir: \$CONTAINER_REPO_DIR' 'Type exit to return to the host shell.' '==================================='; if [[ -f "\$CONTAINER_VENV_DIR/bin/activate" ]]; then source "\$CONTAINER_VENV_DIR/bin/activate"; fi; cd "\$CONTAINER_REPO_DIR"; exec bash -i"
START_EOF
    chmod +x "$START_SCRIPT_PATH"
}

print_summary() {
    cat <<MSG
[dep.sh] environment preparation is complete.
[dep.sh] container name : $CONTAINER_NAME
[dep.sh] image type     : $TYPE
[dep.sh] recreate mode  : $RECREATE
[dep.sh] DeepSpeed version: $DEEPSPEED_VERSION
[dep.sh] docker image   : $DOCKER_IMAGE
[dep.sh] container venv : $CONTAINER_VENV_DIR
[dep.sh] gpu access     : all GPUs
[dep.sh] network mode   : $DOCKER_NETWORK
[dep.sh] ipc mode       : $DOCKER_IPC
[dep.sh] pid mode       : $DOCKER_PID
[dep.sh] ManiSkill data : $CONTAINER_NAME:$CONTAINER_MANISKILL_DATA_DIR
[dep.sh] PartNet data   : $CONTAINER_NAME:$CONTAINER_PARTNET_DATA_DIR
[dep.sh] repo mount     : $HOST_REPO_DIR -> $CONTAINER_REPO_DIR
[dep.sh] start script   : $START_SCRIPT_PATH

[start_docker.sh] usage:
  bash start_docker.sh
MSG
}

require_cmd docker "Install Docker Engine first: https://docs.docker.com/engine/install/ubuntu/"
require_cmd grep "grep is required on the host machine."

if ! docker version >/dev/null 2>&1; then
    echo "[dep.sh] docker is installed but the daemon is not reachable." >&2
    echo "[dep.sh] Make sure the Docker service is running and your user can access it." >&2
    exit 1
fi

resolve_docker_image
check_nvidia_runtime
pull_image
resolve_container_paths
download_hf_checkpoints
ensure_default_state_norm_stats
remove_existing_container
ensure_container
install_container_runtime_dependencies
download_hf_maniskill_data
generate_start_script
print_summary
