#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-1}"
cd "$SCRIPT_DIR"

ENV_ID=${PRETRAIN_ENV_ID:-PickCube-v1}
CONTROL_MODE=${PRETRAIN_CONTROL_MODE:-pd_ee_delta_pos}
NUM_ENVS=${PRETRAIN_NUM_ENVS:-2048}
NUM_EVAL_ENVS=${PRETRAIN_NUM_EVAL_ENVS:-32}
NUM_STEPS=${PRETRAIN_NUM_STEPS:-20}
NUM_EVAL_STEPS=${PRETRAIN_NUM_EVAL_STEPS:-200}
UPDATE_EPOCHS=${PRETRAIN_UPDATE_EPOCHS:-8}
NUM_MINIBATCHES=${PRETRAIN_NUM_MINIBATCHES:-32}
STAGE1_TIMESTEPS=${PRETRAIN_STAGE1_TIMESTEPS:-2000000}
STAGE2_TIMESTEPS=${PRETRAIN_STAGE2_TIMESTEPS:-2000000}
EVAL_FREQ=${PRETRAIN_EVAL_FREQ:-10}
SPARSITY=${DISTILL_MAX_SPARSITY:-0.1}
OUTPUT_DIR=${MLP_PRETRAIN_OUTPUT_DIR:-$SCRIPT_DIR/ckpt/mlp_pretrain}
STAGE1_OUTPUT="$OUTPUT_DIR/best-stage1.pt"
FINAL_OUTPUT=${MLP_PRETRAIN_OUTPUT_OVERRIDE:-$OUTPUT_DIR/best.pt}

if [[ "${MWE:-0}" == "1" ]]; then
  NUM_ENVS=${PRETRAIN_NUM_ENVS:-16}
  NUM_EVAL_ENVS=${PRETRAIN_NUM_EVAL_ENVS:-4}
  NUM_STEPS=${PRETRAIN_NUM_STEPS:-8}
  NUM_EVAL_STEPS=${PRETRAIN_NUM_EVAL_STEPS:-50}
  UPDATE_EPOCHS=${PRETRAIN_UPDATE_EPOCHS:-1}
  NUM_MINIBATCHES=${PRETRAIN_NUM_MINIBATCHES:-4}
  STAGE1_TIMESTEPS=${PRETRAIN_STAGE1_TIMESTEPS:-512}
  STAGE2_TIMESTEPS=${PRETRAIN_STAGE2_TIMESTEPS:-512}
  EVAL_FREQ=1
fi

COMMON_ARGS=(
  --env-id "$ENV_ID"
  --control-mode "$CONTROL_MODE"
  --num-envs "$NUM_ENVS"
  --num-eval-envs "$NUM_EVAL_ENVS"
  --num-steps "$NUM_STEPS"
  --num-eval-steps "$NUM_EVAL_STEPS"
  --update-epochs "$UPDATE_EPOCHS"
  --num-minibatches "$NUM_MINIBATCHES"
  --eval-freq "$EVAL_FREQ"
  --no-capture-video
)

echo "[pretrain] stage 1/2: dense PPO env=$ENV_ID timesteps=$STAGE1_TIMESTEPS"
python -u "$SCRIPT_DIR/pretrain_mlp_rl.py" \
  "${COMMON_ARGS[@]}" \
  --exp-name "mlp-pretrain-dense" \
  --total-timesteps "$STAGE1_TIMESTEPS" \
  --output "$STAGE1_OUTPUT"

echo "[pretrain] stage 2/2: FBS PPO sparsity=$SPARSITY timesteps=$STAGE2_TIMESTEPS"
python -u "$SCRIPT_DIR/pretrain_mlp_rl.py" \
  "${COMMON_ARGS[@]}" \
  --exp-name "mlp-pretrain-fbs-${SPARSITY}" \
  --total-timesteps "$STAGE2_TIMESTEPS" \
  --checkpoint "$STAGE1_OUTPUT" \
  --enable-fbs \
  --fbs-sparsity "$SPARSITY" \
  --output "$FINAL_OUTPUT"

echo "[pretrain] finished: $FINAL_OUTPUT"
