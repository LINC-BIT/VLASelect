


export WANDB_API_KEY=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

if [ "${WANDB_AUTO_LOGIN:-0}" = "1" ]; then
    wandb login --relogin "$WANDB_API_KEY" || true
fi

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/octo/ours_single_agent/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME.log"
mkdir -p "$LOG_DIR"


# nohup python -m mani_skill.trajectory.replay_trajectory --traj-path datasets/PickCube-v1/motionplanning/trajectory.h5 --save-traj --reward-mode dense --record-rewards -o rgb+depth+state_dict -c pd_ee_delta_pos > tmp.log 2>&1 &

# tail -f tmp.log

# --ppo_pretrained_model1_path ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260320-080327-feature_aggregator_gate_reg/[agent1]/checkpoints/best_success_end.pt \
# --ppo_pretrained_model2_path ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260320-080327-feature_aggregator_gate_reg/[agent2]/checkpoints/best_success_end.pt \
    



CUDA_DEVICES=${CUDA_DEVICES:-1}
EXP_NAME=${EXP_NAME:-}
TAIL_LOG=${TAIL_LOG:-1}
LAUNCH_DIRECT=${LAUNCH_DIRECT:-0}
LOG_FILE=${LOG_FILE_OVERRIDE:-$LOG_FILE}
ENV_ID=${ENV_ID_OVERRIDE:-PickCubeObjectScaleUp1p2-v1}
ENVS_ID=${ENVS_ID_OVERRIDE:-"['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']"}
ENV_CHANGE_TIME_POINTS=${ENV_CHANGE_TIME_POINTS_OVERRIDE:-"[31,62,96,131,151,163,207,247,271,300]"}
ENV_CONFIG_PATH=${ENV_CONFIG_PATH_OVERRIDE:-datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json}
STATE_NORM_STATS_PATH=${STATE_NORM_STATS_PATH_OVERRIDE:-ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth}
CHECKPOINT_PATH=${CHECKPOINT_PATH_OVERRIDE:-ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS_OVERRIDE:-100000000}
NUM_ENVS=${NUM_ENVS_OVERRIDE:-256}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS_OVERRIDE:-32}
NUM_STEPS=${NUM_STEPS_OVERRIDE:-50}
NUM_EVAL_STEPS=${NUM_EVAL_STEPS_OVERRIDE:-50}
NUM_MINIBATCHES=${NUM_MINIBATCHES_OVERRIDE:-16}
UPDATE_EPOCHS=${UPDATE_EPOCHS_OVERRIDE:-2}
LEARNING_RATE=${LEARNING_RATE_OVERRIDE:-3e-5}
MAX_SPARSITY=${MAX_SPARSITY_OVERRIDE:-0.8}
ACTOR_LOGSTD=${ACTOR_LOGSTD_OVERRIDE:--0.5}
EVAL_FREQ=${EVAL_FREQ_OVERRIDE:-1}
MAX_TIME=${MAX_TIME_OVERRIDE:-301}
ENABLE_RICL_INJECTION=${ENABLE_RICL_INJECTION:-0}
RICL_BANK_CAPACITY=${RICL_BANK_CAPACITY:-4096}
RICL_BANK_ADD_PER_ITER=${RICL_BANK_ADD_PER_ITER:-128}
RICL_NUM_NEIGHBORS=${RICL_NUM_NEIGHBORS:-4}
RICL_RETRIEVAL_TEMPERATURE=${RICL_RETRIEVAL_TEMPERATURE:-10.0}
RICL_STATE_DIM_CAP=${RICL_STATE_DIM_CAP:-32}
RICL_CONTEXT_HIDDEN_DIM=${RICL_CONTEXT_HIDDEN_DIM:-128}
RICL_PROMPT_FEATURE_SCALE=${RICL_PROMPT_FEATURE_SCALE:-0.12}

EXTRA_ARGS=()
RUN_DIR=""
if [ -n "$EXP_NAME" ]; then
    EXTRA_ARGS+=(--exp-name "$EXP_NAME")
    RUN_DIR="ckpt/$EXP_NAME/[agent]"
fi

# 多agent逻辑，暂不需要
# --ppo_pretrained_model1_path 'ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260409-153956-feature_aggregator_lr3e-5_dual_stream_action_gate_reg_0_h4_2layergate_none/[agent1]/checkpoints copy/best_success_end.pt'
# 多agent逻辑，暂不需要
# --ppo_pretrained_model2_path 'ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260409-153956-feature_aggregator_lr3e-5_dual_stream_action_gate_reg_0_h4_2layergate_none/[agent2]/checkpoints copy/best_success_end.pt'
# 多agent逻辑，暂不需要
# --feature_selector_topk_trajectories 4
# 多agent逻辑，暂不需要
# --feature_selector_temporal_pool_steps 8
# 多agent逻辑，暂不需要
# --feature_selector_strategy return_span
# 多agent逻辑，暂不需要
# --feature_aggregator_attention_num_heads 4
# 多agent逻辑，暂不需要
# --feature_aggregator_gate_type two-layers
# 多agent逻辑，暂不需要
# --feature_aggregator_gate_activation relu
# 多agent逻辑，暂不需要
# --feature_aggregator_norm_type none
# 多agent逻辑，暂不需要
# --enable_feature_fusion
# 多agent逻辑，暂不需要
# --update_feature_aggregator_lr 0.
# 多agent逻辑，暂不需要
# --data_manager_url http://localhost:8015

# [31,62,96,131,151,163,207,247,271,300]


PYTHON_CMD=(
    python -u forgetting/training/vlaselect/online_rl.py
    --env-id "$ENV_ID"
    --envs-id "$ENVS_ID"
    --env-change-time-points "$ENV_CHANGE_TIME_POINTS"
    --env_config_path "$ENV_CONFIG_PATH"
    --state-norm-stats-path "$STATE_NORM_STATS_PATH"
    --checkpoint "$CHECKPOINT_PATH"
    --total_timesteps "$TOTAL_TIMESTEPS"
    --learning_rate "$LEARNING_RATE"
    --eval_freq "$EVAL_FREQ"
    --track
    --max-sparsity "$MAX_SPARSITY"
    --actor-logstd "$ACTOR_LOGSTD"
    --num_envs "$NUM_ENVS"
    --num_eval_envs "$NUM_EVAL_ENVS"
    --num_steps "$NUM_STEPS"
    --num_eval_steps "$NUM_EVAL_STEPS"
    --num_minibatches "$NUM_MINIBATCHES"
    --update_epochs "$UPDATE_EPOCHS"
    --small_model_generation_strategy target-single-traj
    --small_model_feedback_schedule before_per_rollout_if_success_improv_is_larger_than_0.2
    --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
    --small_model_feedback_alpha 0.1
    --small_model_regeneration_increment_ratio 0.05
    --reset_optimizer_after_regeneration
    --small_model_generation_policy small
    --tag ours-single-agent-cl-targetsingletraj-feedif0.2-regen_per_rollout_0.1_4-small_policy-feedback0.1-arch_update0.05-reset_optimizer
    --max_time "$MAX_TIME"
    "${EXTRA_ARGS[@]}"
)

if [ "$ENABLE_RICL_INJECTION" = "1" ]; then
    PYTHON_CMD+=(
        --enable-ricl-injection
        --ricl-bank-capacity "$RICL_BANK_CAPACITY"
        --ricl-bank-add-per-iter "$RICL_BANK_ADD_PER_ITER"
        --ricl-num-neighbors "$RICL_NUM_NEIGHBORS"
        --ricl-retrieval-temperature "$RICL_RETRIEVAL_TEMPERATURE"
        --ricl-state-dim-cap "$RICL_STATE_DIM_CAP"
        --ricl-context-hidden-dim "$RICL_CONTEXT_HIDDEN_DIM"
        --ricl-prompt-feature-scale "$RICL_PROMPT_FEATURE_SCALE"
    )
fi

if [ "$LAUNCH_DIRECT" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
    exec "${PYTHON_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup "${PYTHON_CMD[@]}" > "$LOG_FILE" 2>&1 &

    TRAIN_PID=$!
    echo "TRAIN_PID=$TRAIN_PID"
    echo "RUN_DIR=$RUN_DIR"
    echo "LOG_FILE=$LOG_FILE"

    if [ "$TAIL_LOG" = "1" ]; then
        tail --pid="$TRAIN_PID" -f "$LOG_FILE"
    fi
fi

# CUDA_DEVICES=0
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup python -u -m train.octo.ours.deft_multiple_models.online_rl \
#     --env-id PickCube-v1-mutable \
#     --env_config_path datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json \
#     --state-norm-stats-path ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth \
#     --total_timesteps 100000000 \
#     --learning_rate 1e-4 \
#     --eval_freq 1 \
#     --track \
#     --max-sparsity 0.8 \
#     --num_envs 256 \
#     --num_eval_envs 32 \
#     --num_minibatches 16 \
#     --ppo_pretrained_model1_path ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260401-073052-feature_aggregator_gate_reg/[agent1]/checkpoints/best_success_end.pt \
#     --ppo_pretrained_model2_path ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260401-073052-feature_aggregator_gate_reg/[agent2]/checkpoints/best_success_end.pt \
#     --no-enable_feature_fusion \
#     --use_source_domain_data_for_small_model_generation \
#     --tag naive_baseline-lr1e4 \
#     --data_manager_url http://localhost:8001 \
#     > "$LOG_FILE" 2>&1 &

# tail -f "$LOG_FILE"


# nohup uvicorn ours.de_feature_fusion.data_manager:app --host 0.0.0.0 --port 8000 > data_manager.log 2>&1 &
# tail -f data_manager.log
