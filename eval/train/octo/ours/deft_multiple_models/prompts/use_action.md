# ours改进

我希望你改动两处代码：
- ours核心实现：位于`ours/de_feature_fusion`
- 特征聚合器离线预训练：`train/toy_cnn/ours/deft_multiple_models/pretrain_feature_aggregator.py`和`train/toy_cnn/ours/deft_multiple_models/pretrain_feature_aggregator.sh`

目前只在agent之间传递中间特征信息（即“看到了什么“），而不传递动作信息（即“针对看到的东西做了什么反应“），仅凭中间特征信息很难发挥作用。我想把动作信息（例如模型输出）也纳入传递内容，该如何设计？（先不用写代码，先列出可行方案）
