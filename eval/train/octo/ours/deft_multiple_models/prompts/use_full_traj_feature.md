# ours改进

我希望你改动两处代码：
- ours核心实现：位于`ours/de_feature_fusion`
- 特征聚合器离线预训练：`train/toy_cnn/ours/deft_multiple_models/pretrain_feature_aggregator.py`和`train/toy_cnn/ours/deft_multiple_models/pretrain_feature_aggregator.sh`

改进思路如下：

## 特征选择

我希望agent之间传递的特征更加细粒度，应该是一个三维tensor：
- 第1维代表轨迹数量（所选择的轨迹是return最高的数条轨迹）
- 第2维代表一条轨迹中所包含的动作步数/模型forward数
- 第3维代表一条轨迹中的一步所对应的模型forward所产生中间特征
例如，`torch.rand(4, 5, 6)`代表有4条轨迹、每条轨迹包含5步、每步的中间特征是一个长度为6的一维向量。

## 特征聚合

得到远端传来的特征（三维tensor）后，本地的特征聚合依然采用Attention模块，思路是：
- 使用本地特征作为query、远端特征作为key和value
- 期望的物理意义是：希望本地特征能够找到远端特征中对自己最有益的部分，增强本地特征，提高所采集样本的质量

你只需要改动我上面所说到的，暂时不需要改动和优化其它地方。

你被禁止进行任何文件删除操作！

你不需要进行git提交。
