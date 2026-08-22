
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

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup python -m train.octo.ours.pretrain_large_model_with_fbs \
    --env-id PickCube-v1 \
    --demo-path datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode pd_ee_delta_pos \
    --batch-size 1024 \
    --log-freq 10 \
    --lr 2e-5 \
    --total_iters 2000000 \
    --scheduler_step_size 1000000 \
    --scheduler-gamma 0.1 \
    --eval-freq 20000 \
    --track \
    --pretrained-model ckpt/PickCube-v1/ours/octo/pretrain_large_model/20260121-092802/octo.pt \
    --continue-train-from ckpt/PickCube-v1/ours/octo/pretrain_large_model_with_fbs/20260126-145530/checkpoints/last.pt \
    --max_norm 1.0 \
    --normalize_states \
    > "$LOG_FILE" 2>&1 &

tail -f "$LOG_FILE"
