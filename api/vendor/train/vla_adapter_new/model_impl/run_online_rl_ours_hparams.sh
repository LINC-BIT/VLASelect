#!/usr/bin/env bash

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/vla_adapter_new/model_impl/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME.log"
OUTPUT_DIR_BASE="train/vla_adapter_new/model_impl/outputs/ppo_verify_ours_hparams"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR_BASE"

SEED=${SEED:-1}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
ENV_ID=${ENV_ID:-PickCube-v1}
CONTROL_MODE=${CONTROL_MODE:-pd_ee_delta_pos}
REWARD_MODE=${REWARD_MODE:-normalized_dense}
OBS_MODE=${OBS_MODE:-rgb+depth+state_dict}
MODEL_DIR=${MODEL_DIR:-eval/ckpt/vla_adapter_new/LIBERO-Object}
DEMO_H5=${DEMO_H5:-datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.h5}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-50}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-100000000}
NUM_STEPS=${NUM_STEPS:-50}
ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE:-32}
EVAL_MICRO_BATCH_SIZE=${EVAL_MICRO_BATCH_SIZE:-32}
UPDATE_MICRO_BATCH_SIZE=${UPDATE_MICRO_BATCH_SIZE:-8}
EVAL_EPISODES=${EVAL_EPISODES:-20}
EVAL_EVERY_UPDATES=${EVAL_EVERY_UPDATES:-20}
NUM_ENVS=${NUM_ENVS:-128}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS:-8}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-4}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
HEAD_LR=${HEAD_LR:-1e-5}
ACTION_HEAD_LR=${ACTION_HEAD_LR:-1e-5}
PROPRIO_LR=${PROPRIO_LR:-1e-5}
VALUE_HEAD_LR=${VALUE_HEAD_LR:-1e-4}
LOG_STD_LR=${LOG_STD_LR:-1e-4}
BACKBONE_LR=${BACKBONE_LR:-3e-6}
GAMMA=${GAMMA:-0.99}
GAE_LAMBDA=${GAE_LAMBDA:-0.95}
CLIP_COEF=${CLIP_COEF:-0.2}
ENT_COEF=${ENT_COEF:-0.0}
VF_COEF=${VF_COEF:-0.5}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.5}
FREEZE_ACTION_HEAD_UPDATES=${FREEZE_ACTION_HEAD_UPDATES:-0}
TARGET_KL=${TARGET_KL:-0.03}
MINIBATCH_TARGET_KL_FACTOR=${MINIBATCH_TARGET_KL_FACTOR:-1.5}
DEMO_PRETRAIN_FRAMES=${DEMO_PRETRAIN_FRAMES:-0}
DEMO_PRETRAIN_EPOCHS=${DEMO_PRETRAIN_EPOCHS:-0}
DEMO_PRETRAIN_BATCH_SIZE=${DEMO_PRETRAIN_BATCH_SIZE:-8}
VLA_XYZ_SCALE=${VLA_XYZ_SCALE:-0.25}
SMOKE_STEPS=${SMOKE_STEPS:-32}
SAVE_VIDEO=${SAVE_VIDEO:-true}
SAVE_TRAIN_VIDEO_FREQ=${SAVE_TRAIN_VIDEO_FREQ:-20}
TRAIN_VIDEO_NUM_ENVS=${TRAIN_VIDEO_NUM_ENVS:-4}
TEST_VIDEO_NUM_ENVS=${TEST_VIDEO_NUM_ENVS:-4}
TEST_VIDEO_EPISODES=${TEST_VIDEO_EPISODES:-4}
ROLLOUT_PROGRESS_LOG_INTERVAL=${ROLLOUT_PROGRESS_LOG_INTERVAL:-5}
FREEZE_VLA_BACKBONE=${FREEZE_VLA_BACKBONE:-false}
FREEZE_PROPRIO_PROJECTOR=${FREEZE_PROPRIO_PROJECTOR:-false}

PYTHON_CMD=(
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train/vla_adapter_new/model_impl/online_rl.py
    --mode train
    --seed "$SEED"
    --env-id "$ENV_ID"
    --control-mode "$CONTROL_MODE"
    --reward-mode "$REWARD_MODE"
    --obs-mode "$OBS_MODE"
    --model-dir "$MODEL_DIR"
    --demo-h5 "$DEMO_H5"
    --output-dir "$OUTPUT_DIR_BASE"
    --total-timesteps "$TOTAL_TIMESTEPS"
    --learning-rate "$LEARNING_RATE"
    --head-learning-rate "$HEAD_LR"
    --action-head-learning-rate "$ACTION_HEAD_LR"
    --proprio-learning-rate "$PROPRIO_LR"
    --value-head-learning-rate "$VALUE_HEAD_LR"
    --log-std-learning-rate "$LOG_STD_LR"
    --backbone-learning-rate "$BACKBONE_LR"
    --freeze-vla-backbone "$FREEZE_VLA_BACKBONE"
    --freeze-proprio-projector "$FREEZE_PROPRIO_PROJECTOR"
    --freeze-action-head-updates "$FREEZE_ACTION_HEAD_UPDATES"
    --num-envs "$NUM_ENVS"
    --num-eval-envs "$NUM_EVAL_ENVS"
    --num-steps "$NUM_STEPS"
    --gamma "$GAMMA"
    --gae-lambda "$GAE_LAMBDA"
    --num-minibatches "$NUM_MINIBATCHES"
    --update-epochs "$UPDATE_EPOCHS"
    --clip-coef "$CLIP_COEF"
    --ent-coef "$ENT_COEF"
    --vf-coef "$VF_COEF"
    --max-grad-norm "$MAX_GRAD_NORM"
    --target-kl "$TARGET_KL"
    --minibatch-target-kl-factor "$MINIBATCH_TARGET_KL_FACTOR"
    --eval-every-updates "$EVAL_EVERY_UPDATES"
    --eval-episodes "$EVAL_EPISODES"
    --demo-pretrain-frames "$DEMO_PRETRAIN_FRAMES"
    --demo-pretrain-epochs "$DEMO_PRETRAIN_EPOCHS"
    --demo-pretrain-batch-size "$DEMO_PRETRAIN_BATCH_SIZE"
    --vla-xyz-scale "$VLA_XYZ_SCALE"
    --smoke-steps "$SMOKE_STEPS"
    --save-video "$SAVE_VIDEO"
    --save-train-video-freq "$SAVE_TRAIN_VIDEO_FREQ"
    --train-video-num-envs "$TRAIN_VIDEO_NUM_ENVS"
    --test-video-num-envs "$TEST_VIDEO_NUM_ENVS"
    --test-video-episodes "$TEST_VIDEO_EPISODES"
    --max-runtime-hours "$MAX_RUNTIME_HOURS"
    --rollout-micro-batch-size "$ROLLOUT_MICRO_BATCH_SIZE"
    --eval-micro-batch-size "$EVAL_MICRO_BATCH_SIZE"
    --update-micro-batch-size "$UPDATE_MICRO_BATCH_SIZE"
    --rollout-progress-log-interval "$ROLLOUT_PROGRESS_LOG_INTERVAL"
    --cuda-device "$CUDA_DEVICES"
)

if [ "$LAUNCH_DIRECT" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
    export NCCL_SHM_DISABLE=1
    export NCCL_IB_DISABLE=1
    exec "${PYTHON_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 nohup "${PYTHON_CMD[@]}" > "$LOG_FILE" 2>&1 &

    TRAIN_PID=$!
    echo "TRAIN_PID=$TRAIN_PID"
    echo "OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE"
    echo "LOG_FILE=$LOG_FILE"

    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi
