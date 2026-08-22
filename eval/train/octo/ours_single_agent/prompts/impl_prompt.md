## 实现我的方法在单agent上运行实验的代码

`train/toy_cnn/ours/deft_multiple_models/online_rl.py`是我的方法在多agent下运行实验的代码，其中涉及`ours.de_feature_fusion.client`的函数接口都属于对多agent运行的实现。

我希望你在`train/toy_cnn/ours_single_agent/online_rl.py`中实现我的方法在单agent上运行实验的代码。实现要求如下：
- 尽可能保留`train/toy_cnn/ours/deft_multiple_models/online_rl.py`的代码，即不要对其进行重构、不要对其进行优化、不要改变代码结构，原则上只需要将和多agent运行相关的代码给注释掉即可。在注释时，在注释上方标明`# 多agent逻辑，暂不需要`；
- 唯一的agent使用`class Agent(nn.Module)`，测试时加载的checkpoint为`ckpt/PickCube-v1/ours/toy_cnn/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt`；
- 实现完成后，对`train/toy_cnn/ours_single_agent/online_rl.py`和`train/toy_cnn/ours/deft_multiple_models/online_rl.py`进行diff，将diff结果保存在`train/toy_cnn/ours_single_agent/online_rl.diff`中；
- 你还需要准备启动实验的bash脚本，借鉴`train/toy_cnn/ours/deft_multiple_models/online_rl_ours.sh`，不需要修改超参数，只需要用注释去掉与多agent运行相关的参数；
- 实现完成后，将你的实现过程写入文档`train/toy_cnn/ours_single_agent/codex_run_logs/impl/${DATETIME}.md`。

## 注意事项

- 实现过程中你可以使用卡1验证运行效果；
- 你不能改动任何已有的文件；
- 你不能删除任何已有的文件；
- 你不需要做任何的git提交。
