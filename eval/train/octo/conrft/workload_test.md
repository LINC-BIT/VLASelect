# 测试指定Env序列上的持续学习是否能跑通


参考`train/toy_cnn/ours_single_agent/online_rl_cl.py`的实现，使用设置`--envs-id "['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']"`和`--env-change-time-points "[2,4,6,8,10,12,14,16,18,20]"`进行冒烟试验，确认能跑通，在10个给定的env上都能跑通、顺利结束训练。


注意，你可能会遇到使用Env时报错的问题，如果报错关于`extra`，那么可以参考`train/toy_cnn/ours_single_agent/online_rl_cl.py`进行修复，例如其`_resolve_extra_tensor()`。

## 注意事项

- 实现完成后，将你的实现过程以及修改前后代码的diff结果写入文档`train/toy_cnn/conrft/codex_run_logs/workload_test/${DATETIME}.md`；
- 实现过程中你可以使用卡0验证运行效果；
- 你不能删除任何已有的文件；
- 你不需要做任何的git提交。
