
CUDA_DEVICES=1

export WANDB_API_KEY=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

wandb login

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/octo/ours/deft_multiple_models/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME.log"
mkdir -p "$LOG_DIR"


# nohup python -m mani_skill.trajectory.replay_trajectory --traj-path datasets/PickCube-v1/motionplanning/trajectory.h5 --save-traj --reward-mode dense --record-rewards -o rgb+depth+state_dict -c pd_ee_delta_pos > tmp.log 2>&1 &

# tail -f tmp.log

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup python -u -m train.octo.ours.deft_multiple_models.pretrain_feature_aggregator \
    --env-id PickCube-v1 \
    --env_config_path datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json \
    --state-norm-stats-path ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth \
    --total_timesteps 100000000 \
    --learning_rate 3e-5 \
    --eval_freq 5 \
    --track \
    --max-sparsity 0.8 \
    --num_envs 256 \
    --num_minibatches 16 \
    --ppo_pretrained_model1_path ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt \
    --ppo_pretrained_model2_path ckpt/PickCube-v1/ours/octo/pretrain_large_model_wo_depth_ppo/20260206-141004/checkpoints/best_success_end.pt \
    --target_kl 0.2 \
    --gamma 0.8 \
    --head_learning_rate 1e-5 \
    --gate_reg_coef 0. \
    --aggregator_target_kl 2.0 \
    --feature_selector_topk_trajectories 4 \
    --feature_selector_temporal_pool_steps 8 \
    --feature_selector_strategy topk_return \
    --feature_aggregator_attention_num_heads 4 \
    --feature_aggregator_gate_type two-layers \
    --feature_aggregator_gate_activation relu \
    --feature_aggregator_norm_type none \
    --tag feature_aggregator_lr3e-5_dual_stream_action_gate_reg_0_h4_2layergate_none \
    > "$LOG_FILE" 2>&1 &

tail -f "$LOG_FILE"


# nohup uvicorn ours.de_feature_fusion.data_manager:app --host 0.0.0.0 --port 8000 > data_manager.log 2>&1 &
# tail -f data_manager.log
