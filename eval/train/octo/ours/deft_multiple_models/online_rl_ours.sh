


export WANDB_API_KEY=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

wandb login

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/octo/ours/deft_multiple_models/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME.log"
mkdir -p "$LOG_DIR"


# nohup python -m mani_skill.trajectory.replay_trajectory --traj-path datasets/PickCube-v1/motionplanning/trajectory.h5 --save-traj --reward-mode dense --record-rewards -o rgb+depth+state_dict -c pd_ee_delta_pos > tmp.log 2>&1 &

# tail -f tmp.log

# --ppo_pretrained_model1_path ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260320-080327-feature_aggregator_gate_reg/[agent1]/checkpoints/best_success_end.pt \
# --ppo_pretrained_model2_path ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260320-080327-feature_aggregator_gate_reg/[agent2]/checkpoints/best_success_end.pt \
    



CUDA_DEVICES=2

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup python -u -m train.octo.ours.deft_multiple_models.online_rl \
    --env-id PickCube-v1-mutable \
    --env_config_path datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json \
    --state-norm-stats-path ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth \
    --total_timesteps 100000000 \
    --learning_rate 3e-5 \
    --eval_freq 1 \
    --track \
    --max-sparsity 0.8 \
    --num_envs 256 \
    --num_eval_envs 32 \
    --num_minibatches 16 \
    --ppo_pretrained_model1_path 'ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260409-153956-feature_aggregator_lr3e-5_dual_stream_action_gate_reg_0_h4_2layergate_none/[agent1]/checkpoints copy/best_success_end.pt' \
    --ppo_pretrained_model2_path 'ckpt/PickCube-v1/ours/octo/pretrain_feature_aggregator/20260409-153956-feature_aggregator_lr3e-5_dual_stream_action_gate_reg_0_h4_2layergate_none/[agent2]/checkpoints copy/best_success_end.pt' \
    --feature_selector_topk_trajectories 4 \
    --feature_selector_temporal_pool_steps 8 \
    --feature_selector_strategy return_span \
    --feature_aggregator_attention_num_heads 4 \
    --feature_aggregator_gate_type two-layers \
    --feature_aggregator_gate_activation relu \
    --feature_aggregator_norm_type none \
    --enable_feature_fusion \
    --update_feature_aggregator_lr 0. \
    --small_model_generation_strategy 'target-single-traj' \
    --small_model_feedback_schedule 'before_per_rollout_if_success_improv_is_larger_than_0.2' \
    --small_model_regeneration_schedule 'before_per_rollout_if_success_improv_less_than_0.1_for_4_iters' \
    --small_model_feedback_alpha 0.1 \
    --small_model_regeneration_increment_ratio 0.05 \
    --reset_optimizer_after_regeneration \
    --small_model_generation_policy small \
    --tag ours-targetsingletraj-feedif0.2-regen_per_rollout_0.1_4-small_policy-feedback0.1-arch_update0.05-reset_optimizer \
    --data_manager_url http://localhost:8015 \
    --max_time 61 \
    > "$LOG_FILE" 2>&1 &

tail -f "$LOG_FILE"

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
