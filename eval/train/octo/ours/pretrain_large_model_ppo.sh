
CUDA_DEVICES=0

export WANDB_API_KEY=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

wandb login

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/octo/ours/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME.log"
mkdir -p "$LOG_DIR"


# nohup python -m mani_skill.trajectory.replay_trajectory --traj-path datasets/PickCube-v1/motionplanning/trajectory.h5 --save-traj --reward-mode dense --record-rewards -o rgb+depth+state_dict -c pd_ee_delta_pos > tmp.log 2>&1 &

# tail -f tmp.log

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m train.octo.ours.pretrain_large_model_ppo \
    --env-id PickCube-v1 \
    --env_config_path datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json \
    --state-norm-stats-path ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth \
    --bc_pretrained_fbs_model_path ckpt/PickCube-v1/ours/octo/pretrain_large_model_with_fbs/20260129-070845/checkpoints/best_eval_success_once.pt \
    --total_timesteps 100000000 \
    --learning_rate 3e-4 \
    --eval_freq 5 \
    --track \
    --max-sparsity 0.8 \
    --tag eval-only \
    --continue-train-from ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt 
    # > "$LOG_FILE" 2>&1 &

# tail -f "$LOG_FILE"
