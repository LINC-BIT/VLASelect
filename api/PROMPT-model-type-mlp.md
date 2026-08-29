在 `api/model_type` 中使用 `ppo-mlp-example.py` 给出的官方 ManiSkill MLP 模型和 PPO
训练逻辑对 PickCube-v1 进行两阶段强化学习预训练，不再使用数据集行为克隆或 CNN 蒸馏。

## 已确认的实现约定

- 模型沿用官方 PPO 示例：actor 和 critic 分别为 3 个 256 宽度 Tanh 隐藏层，actor 输出连续动作均值，critic 输出 value，actor_logstd 参与 PPO 训练。
- 第一阶段直接进行 dense PPO 预训练，不启用 FBS；默认参数沿用示例首部命令：`num_envs=2048`、`update_epochs=8`、`num_minibatches=32`、`total_timesteps=2_000_000`、`eval_freq=10`、`num_steps=20`。
- 第二阶段从第一阶段最佳成功率 checkpoint 继续 PPO；在 actor 和 critic 各自第 1、3 个隐藏 Linear 层接入 FBS，并在全部 rollout、评测和更新中将 sparsity 固定为 `0.1`。
- dense checkpoint 载入 FBS 模型时，原 Linear 权重映射到对应 FBS 模块的 `raw_linear`，新增的 FBS gate 参数由第二阶段 PPO 训练。
- 两阶段均在 PickCube-v1 的 state observation 下训练，保持官方 PPO 的 GAE、clipped policy loss、value loss 和确定性评测逻辑。控制模式设为 `pd_ee_delta_pos`，使 4 维动作输出与 `mlp.sh` 及现有环境配置一致。
- 每次评测按 `success_once` 保存最优 checkpoint。第一阶段输出 `api/model_type/ckpt/mlp_pretrain/best-stage1.pt`，第二阶段输出兼容 `mlp.sh` 的 `api/model_type/ckpt/mlp_pretrain/best.pt`。
- `pretrain_mlp.sh` 可从任意当前工作目录启动，通过 `CUDA_DEVICES` 指定 GPU。正式训练参数可用 `PRETRAIN_*` 环境变量覆盖；`MWE=1` 只缩小规模用于完整路径验证。
