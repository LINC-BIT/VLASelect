#!/usr/bin/env bash
set -euo pipefail

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/edgevla/env_verify/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME-unitree-g1-lift-apple.log"
OUTPUT_DIR_BASE="train/edgevla/env_verify/outputs/ppo_unitree_g1_lift_apple"
ENV_ID=${ENV_ID_OVERRIDE:-UnitreeG1LiftApple-v1}
mkdir -p "$LOG_DIR" "$OUTPUT_DIR_BASE"

CUDA_DEVICES=${CUDA_DEVICES:-1,2,3,4}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}

PYTHON_CMD=(
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train/edgevla/env_verify/online_rl_unitree_g1_lift_apple.py
    --mode train
    --seed 1
    --env-id "$ENV_ID"
    --control-mode pd_joint_delta_pos
    --reward-mode normalized_dense
    --obs-mode rgb+state_dict
    --model-dir ckpt/vla_adapter_new/LIBERO-Object
    --output-dir "$OUTPUT_DIR_BASE"
    --total-timesteps 100000000
    --num-envs 128
    --num-eval-envs 8
    --num-steps 64
    --num-minibatches 16
    --update-epochs 2
    --learning-rate 6e-5
    --head-learning-rate 6e-5
    --state-learning-rate 6e-5
    --value-head-learning-rate 6e-5
    --backbone-learning-rate 6e-5
    --weight-decay 1e-6
    --gamma 0.99
    --gae-lambda 0.95
    --clip-coef 0.2
    --ent-coef 1e-3
    --vf-coef 0.5
    --max-grad-norm 0.5
    --target-kl 0.02
    --minibatch-target-kl-factor 1.0
    --eval-episodes 16
    --eval-every-updates 1
    --max-runtime-hours 400
    --rollout-micro-batch-size 256
    --eval-micro-batch-size 256
    --update-micro-batch-size 32
    --rollout-progress-log-interval 10
    --freeze-vla-backbone false
    --backbone-warmup-updates 0
    --run-setup-smoke false
    --save-video false
    --save-train-video-freq 10
    --train-video-num-envs 4
    --test-video-num-envs 4
    --test-video-episodes 4
    --action-dim 12
    --env-action-dim 25
    --state-dim 73
    --cuda-device "$CUDA_DEVICES"
)

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

    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
