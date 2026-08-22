#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
MANISKILL_ROOT=${MANISKILL_ROOT:-/home/Maniskill}
export PYTHONPATH="${EVAL_ROOT}:${MANISKILL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN=${PYTHON_BIN:-/home/Maniskill/.venvs/internvl_qwen3/bin/python}
BATCH_SIZE=${BATCH_SIZE:-128}
RESUME_DIR=${RESUME_DIR:-${EVAL_ROOT}/ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306}

CMD=(
  "${PYTHON_BIN}" "${EVAL_ROOT}/train/multi_agents/two_robot_pick/bc_pretrain.py"
  --task-name TwoRobotPickCube-v2
  --model-backbone multi_agents
  --image-size 112
  --expert-agent-dir "${EVAL_ROOT}/ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942"
  --trajectory-h5-path /home/Maniskill/datasets/TwoRobotPickCube-v2/rl/trajectory.rgb+state_dict.pd_joint_delta_pos.physx_cuda.h5
  --obs-mode rgb+state_dict
  --control-mode pd_joint_delta_pos
  --expert-control-mode pd_joint_delta_pos
  --reward-mode normalized_dense
  --num-successful-trajectories 10240
  --num-collect-envs 1024
  --sft-total-iters 800000
  --batch-size "${BATCH_SIZE}"
  --log-interval-iters 2000
  --state-stats-batch-size 1024
  --eval-interval-iters 2000
  --eval-episodes 50
  --num-eval-envs 16
  --normalize-state
  --use-amp
  --policy-mode native
  --vision-token-pool-size 16
  --backbone-learning-rate 1e-4
  --head-learning-rate 2e-4
  --state-learning-rate 2e-4
  --attn-implementation sdpa
  --tiny-hidden-dim 640
  --tiny-vision-layers 7
  --tiny-decoder-layers 8
  --tiny-attention-heads 10
  --tiny-patch-size 14
  --tiny-ffn-mult 4
  --tiny-num-action-bins 256
  --tiny-prompt-length 24
)

if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=(--resume-dir "${RESUME_DIR}")
fi

"${CMD[@]}"
