
CUDA_DEVICES=2

export WANDB_API_KEY=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

wandb login

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/octo/ours/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME.log"
mkdir -p "$LOG_DIR"


# nohup python -m mani_skill.trajectory.replay_trajectory --traj-path datasets/PickCube-v1/motionplanning/trajectory.h5 --save-traj --reward-mode dense --record-rewards -o rgb+depth+state_dict -c pd_ee_delta_pos > tmp.log 2>&1 &

# tail -f tmp.log
gen_small_model_strategy=target-mean

if [ "$gen_small_model_strategy" = "target-mean" ]; then
    CUDA_DEVICES=2
elif [ "$gen_small_model_strategy" = "source" ]; then
    CUDA_DEVICES=0
fi


CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup python -m train.octo.ours.ft_large_model_ppo_over_evolving_envs \
    --env-id PickCube-v1-mutable \
    --env_config_path datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json \
    --state-norm-stats-path ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth \
    --bc_pretrained_fbs_model_path ckpt/PickCube-v1/ours/octo/pretrain_large_model_with_fbs/20260129-070845/checkpoints/best_eval_success_once.pt \
    --total_timesteps 2000000 \
    --num_eval_envs 32 \
    --learning_rate 3e-4 \
    --eval_freq 1 \
    --track \
    --max-sparsity 0.8 \
    --tag pick-blue-smaller-cube-gen-small-model-on-$gen_small_model_strategy \
    --continue-train-from ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt \
    --gen_small_model_strategy $gen_small_model_strategy \
    > "$LOG_FILE" 2>&1 &

tail -f "$LOG_FILE"

# num_envs和num_minibatches都除以2（不除以2环境初始化会炸显存），环境初始化完成时占47GB显存，训练时占62GB显存（即模型训练占15GB显存）
# --evolving_envs_setting_fp train/octo/evolving-env-setting.json \
#     --num-envs 256 \
#     --num_minibatches 16 \