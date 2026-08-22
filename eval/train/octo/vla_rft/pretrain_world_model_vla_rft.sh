DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="train/octo/vla_rft/nohup_out/$DATE"
LOG_FILE="$LOG_DIR/$TIME-pretrain_world_model.log"
mkdir -p "$LOG_DIR"

CUDA_DEVICES=${CUDA_DEVICES:-1}

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup python -u -m train.octo.vla_rft.pretrain_world_model \
    --dataset-path datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.h5 \
    --state-norm-stats-path ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth \
    --epochs 10 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --latent-dim 256 \
    --max-reference-bank-size 256 \
    --tag default \
    > "$LOG_FILE" 2>&1 &

tail -f "$LOG_FILE"
