# 优化我提出的基于多Agent特征融合的Online RL增强算法

两个Agent分别在自己所处的新环境中进行online RL（见`train/toy_cnn/ours/deft_multiple_models/online_rl.py`）。

我所提出的算法通过`train/toy_cnn/ours/deft_multiple_models/online_rl_ours.sh`运行，核心逻辑是（实现见`ours/de_feature_fusion`）：
- 对于每个Agent，在开始online RL前，根据新环境数据将原始模型转化为参数量更小的小模型；
- 对于每个Agent，在online RL的rollout阶段中，允许通过Attention模块融入和利用其它Agent在rollout时所产生的中间特征，以增强自己所采集样本的质量。

基准算法通过`train/toy_cnn/ours/deft_multiple_models/online_rl_baseline.sh`运行，根据预训练所在环境将原始模型转化为参数量更小的小模型，且不涉及跨Agent的特征交换和利用。

