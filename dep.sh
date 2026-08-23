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
HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}

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
    if python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
        log "found Python package: huggingface_hub"
        return
    fi

    log "installing missing Hugging Face downloader via pip"
    python3 -m pip install -U "huggingface_hub[cli]"

    if command -v hf >/dev/null 2>&1; then
        log "found Hugging Face CLI after installation: $(command -v hf)"
        return
    fi

    if python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
        log "found Python package after installation: huggingface_hub"
        return
    fi

    echo "[dep.sh] failed to install Hugging Face downloader automatically." >&2
    echo '[dep.sh] Please install it manually: python3 -m pip install -U "huggingface_hub[cli]"' >&2
    exit 1
}

download_hf_checkpoints() {
    if [[ -z "$HF_CKPT_REPO" ]]; then
        return
    fi

    require_hf_downloader
    export HF_HUB_DOWNLOAD_TIMEOUT

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
    HF_TOKEN="${HF_TOKEN:-}" \
    python3 - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ['HF_CKPT_REPO']
repo_type = os.environ['HF_CKPT_REPO_TYPE']
revision = os.environ['HF_CKPT_REVISION']
local_dir = Path(os.environ['HF_CKPT_LOCAL_DIR'])
patterns = json.loads(os.environ['HF_CKPT_PATTERNS_JSON'])
token = os.environ.get('HF_TOKEN') or None

snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    revision=revision,
    local_dir=local_dir,
    allow_patterns=patterns,
    token=token,
)
PY
}

ensure_container_maniskill_dirs() {
    docker exec "$CONTAINER_NAME" bash -lc "mkdir -p '$CONTAINER_MANISKILL_DATA_DIR'"
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

    if ! docker exec "$CONTAINER_NAME" bash -lc "python -c 'import huggingface_hub'" >/dev/null 2>&1; then
        log "installing container Python package: huggingface_hub"
        docker exec "$CONTAINER_NAME" bash -lc "python -m pip install -U 'huggingface_hub'" || status=$?
    fi

    if [[ "$status" -eq 0 ]]; then
        ensure_container_maniskill_dirs || status=$?
    fi

    if [[ "$status" -eq 0 ]]; then
        log "downloading ManiSkill assets from Hugging Face repo: $HF_MANISKILL_DATA_REPO"
        log "target directory: $CONTAINER_MANISKILL_DATA_DIR (inside container)"
        if [[ -n "$HF_MANISKILL_DATA_LIST" ]]; then
            log "ManiSkill data list: $HF_MANISKILL_DATA_LIST"
        elif [[ -f "$ROOT_DIR/hf_maniskill_data_paths.txt" ]]; then
            log "ManiSkill data list: $ROOT_DIR/hf_maniskill_data_paths.txt"
        else
            log "ManiSkill data list: built into dep.sh"
        fi

        docker exec             -e HF_MANISKILL_DATA_REPO="$HF_MANISKILL_DATA_REPO"             -e HF_MANISKILL_DATA_REPO_TYPE="$HF_MANISKILL_DATA_REPO_TYPE"             -e HF_MANISKILL_DATA_REVISION="$HF_MANISKILL_DATA_REVISION"             -e HF_MANISKILL_DATA_LOCAL_DIR="$CONTAINER_MANISKILL_DATA_DIR"             -e HF_MANISKILL_DATA_PATTERNS_JSON="$patterns_json"             -e HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT"             -e HF_TOKEN             -e HTTP_PROXY             -e HTTPS_PROXY             -e ALL_PROXY             -e NO_PROXY             "$CONTAINER_NAME"             bash -lc "python - <<'PY_HF_DATA'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ['HF_MANISKILL_DATA_REPO']
repo_type = os.environ['HF_MANISKILL_DATA_REPO_TYPE']
revision = os.environ['HF_MANISKILL_DATA_REVISION']
local_dir = Path(os.environ['HF_MANISKILL_DATA_LOCAL_DIR'])
patterns = json.loads(os.environ['HF_MANISKILL_DATA_PATTERNS_JSON'])
token = os.environ.get('HF_TOKEN') or None

snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    revision=revision,
    local_dir=local_dir,
    allow_patterns=patterns,
    token=token,
)
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

    ensure_container_python_package pypdf pypdf
    ensure_container_python_package pinocchio pin
    ensure_container_python_package noise noise

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
[dep.sh] docker image   : $DOCKER_IMAGE
[dep.sh] container venv : $CONTAINER_VENV_DIR
[dep.sh] gpu access     : all GPUs
[dep.sh] network mode   : $DOCKER_NETWORK
[dep.sh] ipc mode       : $DOCKER_IPC
[dep.sh] pid mode       : $DOCKER_PID
[dep.sh] ManiSkill data : $CONTAINER_NAME:$CONTAINER_MANISKILL_DATA_DIR
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
remove_existing_container
ensure_container
install_container_runtime_dependencies
download_hf_maniskill_data
generate_start_script
print_summary
