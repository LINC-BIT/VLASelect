#!/usr/bin/env bash
set -euo pipefail

MODEL_TYPE_NAME=${1:?model type is required}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
EVAL_ROOT="$REPO_ROOT/eval"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_TYPE="$MODEL_TYPE_NAME"
cd "$SCRIPT_DIR"

export WANDB_API_KEY=${WANDB_API_KEY:-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}
if [[ "${WANDB_AUTO_LOGIN:-0}" == "1" ]]; then
  wandb login --relogin "$WANDB_API_KEY" || true
fi

DATE=$(date +%Y%m%d-%H%M%S)
RUN_NAME=${RUN_NAME_OVERRIDE:-${EXP_NAME_OVERRIDE:-results/$MODEL_TYPE_NAME/$DATE}}
LOG_FILE=${LOG_FILE_OVERRIDE:-$SCRIPT_DIR/results/$MODEL_TYPE_NAME/logs/$DATE.log}
mkdir -p "$(dirname "$LOG_FILE")"

CUDA_DEVICES=${CUDA_DEVICES:-1}
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
if [[ "$MODEL_TYPE_NAME" == "mlp" ]]; then
  DEFAULT_MAX_SPARSITY=0.1
else
  DEFAULT_MAX_SPARSITY=0.8
fi

# These defaults are the same relative paths as the reference Octo launcher;
# resolve them against the repository root so this script is cwd-independent.
ENV_CONFIG_PATH=${ENV_CONFIG_PATH_OVERRIDE:-$EVAL_ROOT/datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json}
STATE_NORM_STATS_PATH=${STATE_NORM_STATS_PATH_OVERRIDE:-$EVAL_ROOT/ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth}
if [[ -n "${CHECKPOINT_PATH_OVERRIDE:-}" ]]; then
  CHECKPOINT_PATH="$CHECKPOINT_PATH_OVERRIDE"
elif [[ "$MODEL_TYPE_NAME" == "mlp" ]]; then
  CHECKPOINT_PATH="$SCRIPT_DIR/best-mlp-model"
else
  CHECKPOINT_PATH="$EVAL_ROOT/ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt"
fi

PYTHON_CMD=(
  python -u -m train.octo.ours_single_agent.online_rl_cl
  --exp-name "$RUN_NAME"
  --env-id "${ENV_ID_OVERRIDE:-PickCubeObjectScaleUp1p2-v1}"
  --envs-id "${ENVS_ID_OVERRIDE:-['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']}"
  --env-change-time-points "${ENV_CHANGE_TIME_POINTS_OVERRIDE:-[31,62,96,131,151,163,207,247,271,300]}"
  --env_config_path "$ENV_CONFIG_PATH"
  --state-norm-stats-path "$STATE_NORM_STATS_PATH"
  --checkpoint "$CHECKPOINT_PATH"
  --total_timesteps "${TOTAL_TIMESTEPS_OVERRIDE:-100000000}"
  --learning_rate "${LEARNING_RATE_OVERRIDE:-3e-5}"
  --eval_freq "${EVAL_FREQ_OVERRIDE:-1}"
  --track
  --max-sparsity "${MAX_SPARSITY_OVERRIDE:-$DEFAULT_MAX_SPARSITY}"
  --actor-logstd "${ACTOR_LOGSTD_OVERRIDE:--0.5}"
  --num_envs "${NUM_ENVS_OVERRIDE:-256}"
  --num_eval_envs "${NUM_EVAL_ENVS_OVERRIDE:-32}"
  --num_steps "${NUM_STEPS_OVERRIDE:-50}"
  --num_eval_steps "${NUM_EVAL_STEPS_OVERRIDE:-50}"
  --num_minibatches "${NUM_MINIBATCHES_OVERRIDE:-16}"
  --update_epochs "${UPDATE_EPOCHS_OVERRIDE:-2}"
  --small_model_generation_strategy target-single-traj
  --small_model_feedback_schedule before_per_rollout_if_success_improv_is_larger_than_0.2
  --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
  --small_model_feedback_alpha 0.1
  --small_model_regeneration_increment_ratio 0.05
  --reset_optimizer_after_regeneration
  --small_model_generation_policy small
  --tag "model-type-$MODEL_TYPE_NAME"
  --max_time "${MAX_TIME_OVERRIDE:-301}"
)

if [[ -n "${PRE_GENERATED_MODEL:-}" ]]; then
  PYTHON_CMD+=(
    --pre-generated-model "$PRE_GENERATED_MODEL"
    --small_model_feedback_schedule once
    --small_model_regeneration_schedule once
  )
fi

if [[ "${ENABLE_RICL_INJECTION:-0}" == "1" ]]; then
  PYTHON_CMD+=(
    --enable-ricl-injection
    --ricl-bank-capacity "${RICL_BANK_CAPACITY:-4096}"
    --ricl-bank-add-per-iter "${RICL_BANK_ADD_PER_ITER:-128}"
    --ricl-num-neighbors "${RICL_NUM_NEIGHBORS:-4}"
    --ricl-retrieval-temperature "${RICL_RETRIEVAL_TEMPERATURE:-10.0}"
    --ricl-state-dim-cap "${RICL_STATE_DIM_CAP:-32}"
    --ricl-context-hidden-dim "${RICL_CONTEXT_HIDDEN_DIM:-128}"
    --ricl-prompt-feature-scale "${RICL_PROMPT_FEATURE_SCALE:-0.12}"
  )
fi

if [[ "$LAUNCH_DIRECT" == "1" ]]; then
  "${PYTHON_CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
  nohup "${PYTHON_CMD[@]}" >"$LOG_FILE" 2>&1 &
  TRAIN_PID=$!
  echo "TRAIN_PID=$TRAIN_PID"
  if [[ "$TAIL_LOG" == "1" ]]; then
    tail --pid="$TRAIN_PID" -f "$LOG_FILE"
  else
    set +e
    wait "$TRAIN_PID"
    TRAIN_STATUS=$?
    set -e
    if [[ "$TRAIN_STATUS" -ne 0 ]]; then
      echo "training failed (exit=$TRAIN_STATUS); log: $LOG_FILE" >&2
      tail -80 "$LOG_FILE" >&2 || true
      exit "$TRAIN_STATUS"
    fi
  fi
fi

python "$SCRIPT_DIR/_plot_acc.py" \
  --run-dir "$SCRIPT_DIR/ckpt/$RUN_NAME/[agent]" \
  --output "$SCRIPT_DIR/${MODEL_TYPE_NAME^^}-ACC.png" \
  --title "${MODEL_TYPE_NAME^^} training accuracy"
