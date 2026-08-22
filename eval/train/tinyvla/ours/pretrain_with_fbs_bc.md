# 使用有监督训练算法训练带有FBS模块的大模型


模仿`train/vla_adapter_new/ours/pretrain_with_fbs_bc.py`和`train/vla_adapter_new/ours/pretrain_with_fbs_bc.sh`，在`train/edgevla/ours`下实现EdgeVLA模型和OpenCabinetDrawerEasyLevel0-v1下的有监督FBS大模型训练。
- 你可以使用卡0至卡7并行运行该有监督训练，最大化训练效率；
- 你不能修改和删除已有文件。
