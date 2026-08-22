#!/usr/bin/env bash

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/vla_adapter_new/model_impl/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME-hand.log"
OUTPUT_DIR_BASE="train/vla_adapter_new/model_impl/outputs/ppo_rotate_hand"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR_BASE"

SEED=${SEED:-1}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}

ENV_ID=${ENV_ID:-EasierRotateSingleObjectInHandLevel0-v1}
CONTROL_MODE=${CONTROL_MODE:-pd_joint_delta_pos}
REWARD_MODE=${REWARD_MODE:-normalized_dense}
OBS_MODE=${OBS_MODE:-rgb+state_dict}
MODEL_DIR=${MODEL_DIR:-eval/ckpt/vla_adapter_new/LIBERO-Object}

TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-30000000}
NUM_ENVS=${NUM_ENVS:-128}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS:-32}
NUM_STEPS=${NUM_STEPS:-60}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-8}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-1}

ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE:-16}
EVAL_MICRO_BATCH_SIZE=${EVAL_MICRO_BATCH_SIZE:-16}
UPDATE_MICRO_BATCH_SIZE=${UPDATE_MICRO_BATCH_SIZE:-16}

HEAD_LR=${HEAD_LR:-3e-5}
STATE_LR=${STATE_LR:-3e-5}
VALUE_HEAD_LR=${VALUE_HEAD_LR:-3e-5}
BACKBONE_LR=${BACKBONE_LR:-3e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-6}

GAMMA=${GAMMA:-0.99}
GAE_LAMBDA=${GAE_LAMBDA:-0.95}
CLIP_COEF=${CLIP_COEF:-0.2}
ENT_COEF=${ENT_COEF:-1e-3}
VF_COEF=${VF_COEF:-0.5}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.5}
TARGET_KL=${TARGET_KL:-0.02}
MINIBATCH_TARGET_KL_FACTOR=${MINIBATCH_TARGET_KL_FACTOR:-1.5}

EVAL_EPISODES=${EVAL_EPISODES:-4}
EVAL_EVERY_UPDATES=${EVAL_EVERY_UPDATES:-10}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-40}
ROLLOUT_PROGRESS_LOG_INTERVAL=${ROLLOUT_PROGRESS_LOG_INTERVAL:-5}

FREEZE_VLA_BACKBONE=${FREEZE_VLA_BACKBONE:-false}
BACKBONE_WARMUP_UPDATES=${BACKBONE_WARMUP_UPDATES:-0}
RUN_SETUP_SMOKE=${RUN_SETUP_SMOKE:-false}

ACTION_DIM=${ACTION_DIM:-16}
STATE_DIM=${STATE_DIM:-105}

SAVE_VIDEO=${SAVE_VIDEO:-true}
SAVE_TRAIN_VIDEO_FREQ=${SAVE_TRAIN_VIDEO_FREQ:-10}
TRAIN_VIDEO_NUM_ENVS=${TRAIN_VIDEO_NUM_ENVS:-4}
TEST_VIDEO_NUM_ENVS=${TEST_VIDEO_NUM_ENVS:-4}
TEST_VIDEO_EPISODES=${TEST_VIDEO_EPISODES:-4}

PYTHON_CMD=(
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train/vla_adapter_new/model_impl/online_rl_hand.py
    --mode train
    --seed "$SEED"
    --env-id "$ENV_ID"
    --control-mode "$CONTROL_MODE"
    --reward-mode "$REWARD_MODE"
    --obs-mode "$OBS_MODE"
    --model-dir "$MODEL_DIR"
    --output-dir "$OUTPUT_DIR_BASE"
    --total-timesteps "$TOTAL_TIMESTEPS"
    --num-envs "$NUM_ENVS"
    --num-eval-envs "$NUM_EVAL_ENVS"
    --num-steps "$NUM_STEPS"
    --num-minibatches "$NUM_MINIBATCHES"
    --update-epochs "$UPDATE_EPOCHS"
    --head-learning-rate "$HEAD_LR"
    --state-learning-rate "$STATE_LR"
    --value-head-learning-rate "$VALUE_HEAD_LR"
    --backbone-learning-rate "$BACKBONE_LR"
    --weight-decay "$WEIGHT_DECAY"
    --gamma "$GAMMA"
    --gae-lambda "$GAE_LAMBDA"
    --clip-coef "$CLIP_COEF"
    --ent-coef "$ENT_COEF"
    --vf-coef "$VF_COEF"
    --max-grad-norm "$MAX_GRAD_NORM"
    --target-kl "$TARGET_KL"
    --minibatch-target-kl-factor "$MINIBATCH_TARGET_KL_FACTOR"
    --eval-episodes "$EVAL_EPISODES"
    --eval-every-updates "$EVAL_EVERY_UPDATES"
    --max-runtime-hours "$MAX_RUNTIME_HOURS"
    --rollout-micro-batch-size "$ROLLOUT_MICRO_BATCH_SIZE"
    --eval-micro-batch-size "$EVAL_MICRO_BATCH_SIZE"
    --update-micro-batch-size "$UPDATE_MICRO_BATCH_SIZE"
    --rollout-progress-log-interval "$ROLLOUT_PROGRESS_LOG_INTERVAL"
    --freeze-vla-backbone "$FREEZE_VLA_BACKBONE"
    --backbone-warmup-updates "$BACKBONE_WARMUP_UPDATES"
    --run-setup-smoke "$RUN_SETUP_SMOKE"
    --save-video "$SAVE_VIDEO"
    --save-train-video-freq "$SAVE_TRAIN_VIDEO_FREQ"
    --train-video-num-envs "$TRAIN_VIDEO_NUM_ENVS"
    --test-video-num-envs "$TEST_VIDEO_NUM_ENVS"
    --test-video-episodes "$TEST_VIDEO_EPISODES"
    --action-dim "$ACTION_DIM"
    --state-dim "$STATE_DIM"
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
