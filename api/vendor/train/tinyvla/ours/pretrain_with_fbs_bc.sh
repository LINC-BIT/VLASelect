#!/usr/bin/env bash
set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OURS_DIR="train/tinyvla/ours"
LOG_DIR="$OURS_DIR/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME-pretrain-with-fbs-bc-open-cabinet-drawer.log"
OUTPUT_DIR_BASE="$OURS_DIR/outputs/bc_open_cabinet_drawer_fbs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR_BASE"

CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
ENV_ID=${ENV_ID_OVERRIDE:-OpenCabinetDrawerEasyLevel0-v1}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS_OVERRIDE:-50}
EARLY_STOP_ZERO_SUCCESS_MINUTES=${EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE:-45000}
RUN_NAME=${RUN_NAME_OVERRIDE:-}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT_OVERRIDE:-ckpt/tinyvla/model_impl/outputs/ppo_open_cabinet_drawer/20260507-113650/best_policy.pt}
STUDENT_INIT_POLICY=${STUDENT_INIT_POLICY_OVERRIDE:-teacher}
RESUME_FROM=${RESUME_FROM_OVERRIDE:-}
SAVE_VIDEO=${SAVE_VIDEO_OVERRIDE:-false}

PYTHON_CMD=(
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_DIR/pretrain_with_fbs_bc.py"
    --mode train
    --seed 1
    --env-id "$ENV_ID"
    --control-mode pd_joint_delta_pos
    --reward-mode normalized_dense
    --obs-mode rgb+state_dict
    --model-dir eval/ckpt/vla_adapter_new/LIBERO-Object
    --output-dir "$OUTPUT_DIR_BASE"
    --teacher-checkpoint "$TEACHER_CHECKPOINT"
    --student-init-policy "$STUDENT_INIT_POLICY"
    --total-timesteps 100000000
    --num-envs 128
    --num-eval-envs 8
    --num-steps 100
    --num-minibatches 16
    --update-epochs 2
    --backbone-learning-rate 3e-5
    --head-learning-rate 3e-5
    --state-learning-rate 3e-5
    --value-head-learning-rate 3e-5
    --weight-decay 1e-6
    --max-grad-norm 0.5
    --eval-episodes 50
    --eval-every-updates 5
    --max-runtime-hours "$MAX_RUNTIME_HOURS"
    --rollout-micro-batch-size 256
    --eval-micro-batch-size 256
    --update-micro-batch-size 32
    --freeze-vla-backbone false
    --run-setup-smoke false
    --save-video "$SAVE_VIDEO"
    --test-video-num-envs 4
    --test-video-episodes 4
    --action-dim 8
    --env-action-dim 13
    --state-dim 44
    --early-stop-zero-success-minutes "$EARLY_STOP_ZERO_SUCCESS_MINUTES"
    --cuda-device "$CUDA_DEVICES"
)

if [ -n "$RESUME_FROM" ]; then
    PYTHON_CMD+=(--resume-from "$RESUME_FROM")
fi

if [ -n "$RUN_NAME" ]; then
    PYTHON_CMD+=(--run-name "$RUN_NAME")
fi

if [ "$LAUNCH_DIRECT" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
    export TORCH_DIST_TIMEOUT_HOURS=${TORCH_DIST_TIMEOUT_HOURS:-6}
    export NCCL_SHM_DISABLE=1
    export NCCL_IB_DISABLE=1
    exec "${PYTHON_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES TORCH_DIST_TIMEOUT_HOURS=${TORCH_DIST_TIMEOUT_HOURS:-6} NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 nohup "${PYTHON_CMD[@]}" > "$LOG_FILE" 2>&1 &

    TRAIN_PID=$!
    echo "TRAIN_PID=$TRAIN_PID"
    echo "OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE"
    echo "LOG_FILE=$LOG_FILE"
    echo "ENV_ID=$ENV_ID"
    echo "RUN_NAME=$RUN_NAME"
    echo "TEACHER_CHECKPOINT=$TEACHER_CHECKPOINT"
    echo "STUDENT_INIT_POLICY=$STUDENT_INIT_POLICY"
    echo "RESUME_FROM=$RESUME_FROM"

    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
