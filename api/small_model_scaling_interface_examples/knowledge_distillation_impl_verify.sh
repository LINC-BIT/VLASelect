#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"

# MWE=1 is handled by api.unified_online_rl.parse_args: it keeps this exact launch
# path and only shortens collected experience and the wall-clock guard.
OUTPUT_DIR=${OUTPUT_DIR_OVERRIDE:-"$SCRIPT_DIR/outputs/knowledge_distillation_online_rl_cl"}
RUN_NAME=${RUN_NAME_OVERRIDE:-}
ENV_ID=${ENV_ID_OVERRIDE:-HoldCubeInHandObjectScaleDown1p2-v1}
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
export CUDA_VISIBLE_DEVICES

ARGS=(--env-id "$ENV_ID" --output-dir "$OUTPUT_DIR" \
  --envs-id "['HoldCubeInHandObjectScaleDown1p2-v1','HoldHammerInHandObjectScaleDown1p6-v1','HoldWrenchInHandObjectScaleUp1p2-v1','HoldWoodBlockInHandObjectScaleDown1p6-v1','HoldHammerInHandObjectScaleUp1p6-v1','HoldHammerInHandObjectScaleDown1p4-v1','HoldWrenchInHandObjectScaleUp1p6-v1','HoldHammerInHandObjectScaleDown1p2-v1','HoldHammerInHandObjectScaleUp1p4-v1','HoldWrenchInHandObjectScaleDown1p6-v1']" \
  --env-change-time-points "[31,62,96,131,151,163,207,247,271,300]" \
  --control-mode pd_joint_delta_pos --reward-mode normalized_dense --obs-mode rgb+state_dict \
  --model-dir eval/ckpt/vla_adapter_new/LIBERO-Object \
  --num-envs 256 --num-eval-envs 8 --num-steps 50 --num-minibatches 16 --update-epochs 2 \
  --learning-rate 3e-5 --head-learning-rate 3e-5 --state-learning-rate 3e-5 --value-head-learning-rate 3e-5 --backbone-learning-rate 3e-5 \
  --weight-decay 1e-6 --gamma 0.8 --gae-lambda 0.9 --clip-coef 0.2 --ent-coef 0.0 --vf-coef 0.5 --max-grad-norm 0.5 --target-kl 0.2 --minibatch-target-kl-factor 1.0 \
  --eval-episodes 50 --eval-every-updates 50 --max-runtime-hours 5.1 --rollout-micro-batch-size 256 --eval-micro-batch-size 256 --update-micro-batch-size 32 \
  --freeze-vla-backbone false --backbone-warmup-updates 0 --save-video false --action-dim 16 --state-dim 105 \
  --large-agent-checkpoint eval/ckpt/vla_adapter_new/ours/outputs/20260502-112804/best_policy.pt \
  --small-model-scaling-strategy target-single-traj --small-model-scaling-policy small \
  --small-model-feedback-schedule before_per_rollout_if_success_improv_is_larger_than_0.2 \
  --small-model-regeneration-schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters \
  --small-model-feedback-alpha 0.1 --small-model-regeneration-increment-ratio 0.05 --reset-optimizer-after-regeneration true --max-sparsity 0.8 \
  --early-stop-zero-success-minutes 45000 --cuda-device "$CUDA_VISIBLE_DEVICES")
if [[ -n "$RUN_NAME" ]]; then ARGS+=(--run-name "$RUN_NAME"); fi
exec python -u "$SCRIPT_DIR/knowledge_distillation_impl.py" "${ARGS[@]}"
