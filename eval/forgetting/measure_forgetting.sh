#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$EVAL_ROOT"
cd "$REPO_ROOT"
source "$EVAL_ROOT/common/interrupt_cleanup.sh"
source "$EVAL_ROOT/common/sanity_check.sh"
source "$EVAL_ROOT/common/resource_summary.sh"

MWE="${MWE:-0}"
FORGETTING_MWE_MAX_TIME_MINUTES="${FORGETTING_MWE_MAX_TIME_MINUTES:-3}"
SUITE_STAMP="${SUITE_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${FORGETTING_RUN_ROOT:-forgetting/results/${SUITE_STAMP}}"
ENV_IDS_FULL="${FORGETTING_ENVS_ID:-['PickCubeObjectScaleDown1p2-v1','PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1']}"
if [[ "$MWE" == "1" ]]; then
    ENV_IDS="$(python3 - "$ENV_IDS_FULL" <<'PY'
import ast, sys
values = list(ast.literal_eval(sys.argv[1]))
print(repr(values[:3]))
PY
)"
    ENV_POINTS='[1,2,3]'
    MAX_TIME="$FORGETTING_MWE_MAX_TIME_MINUTES"
    export VLASELECT_MWE_USE_TRAIN_SUCCESS_ONLY=1
    export MWE_ACTIVE_RUNTIME_ONLY=1
    export FORGETTING_EVAL_STEPS="${FORGETTING_EVAL_STEPS:-50}"
else
    ENV_IDS="$ENV_IDS_FULL"
    ENV_POINTS="${FORGETTING_ENV_CHANGE_TIME_POINTS:-[31,62,96,131,151,163,207,247,271,300]}"
    MAX_TIME="${FORGETTING_MAX_TIME_MINUTES:-301}"
fi

FIRST_ENV_ID="$(python3 - "$ENV_IDS" <<'PY'
import ast, sys
values = list(ast.literal_eval(sys.argv[1]))
if not values:
    raise SystemExit('FORGETTING_ENVS_ID must contain at least one environment')
print(values[0])
PY
)"

mkdir -p "$RUN_ROOT/logs"
printf '%s\n' "$SUITE_STAMP" > "$RUN_ROOT/latest.txt"
vlaselect_resource_summary_start "measure_forgetting.sh"
vlaselect_install_cleanup_trap
vlaselect_run_sanity_check "measure_forgetting.sh" "$EVAL_ROOT" "$MWE" "16" "8"

CPU_TOTAL="${CPU_TOTAL_OVERRIDE:-$(nproc)}"
if [[ "${CPU_THROTTLE_ENABLED:-1}" == "1" ]]; then
    if (( CPU_TOTAL <= 4 )); then CPU_THREAD_LIMIT=1
    elif (( CPU_TOTAL <= 8 )); then CPU_THREAD_LIMIT=2
    elif (( CPU_TOTAL <= 16 )); then CPU_THREAD_LIMIT=4
    else CPU_THREAD_LIMIT=8
    fi
    export OMP_NUM_THREADS="${CPU_THREAD_LIMIT_OVERRIDE:-$CPU_THREAD_LIMIT}"
    export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
    export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS" VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
    export BLIS_NUM_THREADS="$OMP_NUM_THREADS" TOKENIZERS_PARALLELISM=false
fi

declare -A METHOD_SCRIPT=(
    [self_improv]="$SCRIPT_DIR/training/self_improv/run_online_rl.sh"
    [vla_rft]="$SCRIPT_DIR/training/vla_rft/run_online_rl.sh"
    [world_env]="$SCRIPT_DIR/training/world_env/run_online_rl.sh"
    [vlaselect]="$SCRIPT_DIR/training/vlaselect/run_online_rl.sh"
)
declare -A METHOD_GPU=([self_improv]=0 [vla_rft]=0 [world_env]=0 [vlaselect]=0)

METHODS_TO_RUN="${FORGETTING_METHODS:-self_improv,vla_rft,world_env,vlaselect}"
while IFS= read -r method; do
    [[ -z "$method" ]] && continue
    if [[ -z "${METHOD_SCRIPT[$method]+x}" ]]; then
        echo "Unknown method in FORGETTING_METHODS: $method" >&2
        exit 1
    fi
    log_file="$RUN_ROOT/logs/${method}.log"
    exp_name="forgetting/${SUITE_STAMP}/${method}"
    echo "[forgetting] starting ${method} (serial)"
    common_env=(
        SUITE_STAMP="$SUITE_STAMP"
        PYTHONPATH="$EVAL_ROOT:${PYTHONPATH:-}"
        EXP_NAME="$exp_name"
        CUDA_DEVICES="${METHOD_GPU[$method]}"
        ENV_ID_OVERRIDE="$FIRST_ENV_ID"
        ENVS_ID_OVERRIDE="$ENV_IDS"
        ENV_CHANGE_TIME_POINTS_OVERRIDE="$ENV_POINTS"
        MAX_TIME_OVERRIDE="$MAX_TIME"
        TAIL_LOG=0
        LAUNCH_DIRECT=1
        ENABLE_SELF_CURVE_WATCHER=0
        NUM_EVAL_ENVS_OVERRIDE="${FORGETTING_NUM_EVAL_ENVS:-32}"
        FORGETTING_EVAL_STEPS="${FORGETTING_EVAL_STEPS:-50}"
    )
    if [[ "$MWE" == "1" ]]; then
        common_env+=(
            MWE=1
            SMOKE=1
            TOTAL_TIMESTEPS_OVERRIDE=24576
            NUM_ENVS_OVERRIDE=2
            NUM_EVAL_ENVS_OVERRIDE=8
            NUM_STEPS_OVERRIDE=16
            NUM_EVAL_STEPS_OVERRIDE=50
            NUM_MINIBATCHES_OVERRIDE=2
            UPDATE_EPOCHS_OVERRIDE=1
            EVAL_FREQ_OVERRIDE=4
            SUPERVISED_UPDATES_PER_ITER_OVERRIDE=1
            SUPERVISED_BATCH_SIZE_OVERRIDE=2
            WANDB_MODE=disabled
            WANDB_SILENT=true
        )
        if [[ "$method" == "vlaselect" ]]; then
            common_env+=(MWE_MAX_RUNTIME_MINUTES="$MAX_TIME")
        fi
    fi
    if [[ "$method" == "vla_rft" ]]; then
        common_env+=(WORLD_MODEL_CKPT="${VLA_RFT_WM_CKPT:-ckpt/PickCube-v1/baselines/vla_rft/world_model/20260424-160154-run32x2/checkpoints/best.pt}")
    elif [[ "$method" == "world_env" ]]; then
        common_env+=(WORLD_MODEL_CKPT="${WORLD_ENV_WM_CKPT:-ckpt/PickCube-v1/baselines/world_env/world_model/20260425-032853-run032_e2/checkpoints/best_with_reference.pt}")
    fi
    env "${common_env[@]}" bash "${METHOD_SCRIPT[$method]}"
    printf '%s\t%s\n' "$method" "$exp_name" >> "$RUN_ROOT/runs.tsv"
done < <(printf '%s\n' "$METHODS_TO_RUN" | tr ',' '\n')

python3 "$SCRIPT_DIR/summarize_forgetting.py" \
    --run-root "ckpt/forgetting/${SUITE_STAMP}" \
    --env-count "$([[ "$MWE" == "1" ]] && echo 3 || echo 10)" \
    --output-dir "$RUN_ROOT"
