
CUDA_DEVICES=2

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
    --lr 1e-5 \
    --total_iters 2000000 \
    --scheduler_step_size 1000000 \
    --scheduler-gamma 0.1 \
    --eval-freq 10000 \
    --track \
    --pretrained-model ckpt/PickCube-v1/ours/octo/pretrain_large_model/20260121-092802/octo.pt \
    --continue-train-from ckpt/PickCube-v1/ours/octo/pretrain_large_model_with_fbs/20260126-145530/checkpoints/best_eval_success_once.pt \
    --max_norm 1.0 \
    --normalize_states \
    --tag lower-lr-of-1e-5 \
    > "$LOG_FILE" 2>&1 &

tail -f "$LOG_FILE"
