
CUDA_DEVICES=0

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/octo/ours/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME.log"
mkdir -p "$LOG_DIR"


# nohup python -m mani_skill.trajectory.replay_trajectory --traj-path datasets/PickCube-v1/motionplanning/trajectory.h5 --save-traj --reward-mode dense --record-rewards -o rgb+depth+state_dict -c pd_ee_delta_pos > tmp.log 2>&1 &

# tail -f tmp.log

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup python -m train.octo.ours.pretrain_large_model \
    --env-id PickCube-v1 \
    --demo-path datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode pd_ee_delta_pos \
    --batch-size 1024 \
    --log-freq 10 \
    --lr 3e-4 \
    --total_iters 2000000 \
    --scheduler_step_size 1000000 \
    --scheduler-gamma 0.1 \
    --eval-freq 20000 \
    > "$LOG_FILE" 2>&1 &

tail -f "$LOG_FILE"
