# 使用有监督训练算法训练带有FBS模块的大模型


在`train/vla_adapter_new/ours`下增加文件`pretrain_with_fbs_bc.py`和对应的启动脚本：
- 在每个训练迭代中，从环境中拿到输入数据后，先使用teacher model（未增加FBS的原始模型，checkpoint位于`'train/vla_adapter_new/model_impl/outputs/ppo_hold_cube_in_hand/20260430-103518/best_policy.pt'`）生成对应的动作标签，然后使用该标签训练带有FBS模块的大模型；
- 整体代码组织、架构、模型实现等参考`train/vla_adapter_new/model_impl/workload_verify/online_rl_hold_cube_in_hand.py`。因为`train/vla_adapter_new/ours/pretrain_with_fbs*.py`中有太多为了修补PPO增加的补丁、有很多不必要甚至可能不正确的代码，可能只有加载FBS大模型的代码你需要借鉴它；
- 加载teacher model的方式可参考`train/vla_adapter_new/model_impl/workload_verify/online_rl_hold_cube_in_hand.py`，加载带FBS大模型的方式可参考`train/vla_adapter_new/ours/pretrain_with_fbs.py`；
- 有监督训练算法的实现，也许你可以看看`train/vla_adapter_new/model_impl/VLA-Adapter-main.zip`，其中应该有使用交叉熵损失函数进行有监督训练的实现；
- 你可以使用卡0至卡7并行运行该有监督训练，最大化训练效率；
- 你不能修改和删除已有文件。
