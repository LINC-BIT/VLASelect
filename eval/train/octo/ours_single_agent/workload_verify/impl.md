# 验证所构造的workload符合实验需求

我希望新构造的Env能够让我们方法在有限时间（例如30分钟内）取得较明显的精度提升，而不是一直是0左右。新构造的Env位于`workloads/table_top/__init__.py`中，具体为`['PickCubeLightStronger50-v1', 'PickCubeLightWeaker50-v1', 'PickCubeObjectBlack-v1', 'PickCubeObjectPurple-v1', 'PickCubeColorTempLower50-v1', 'PickCubeColorTempHigher50-v1', 'PushCubeLightStronger50-v1', 'PushCubeLightWeaker50-v1', 'PushCubeObjectBlack-v1', 'PushCubeObjectPurple-v1', 'PushCubeColorTempLower50-v1', 'PushCubeColorTempHigher50-v1', 'PokeCubeLightStronger50-v1', 'PokeCubeLightWeaker50-v1', 'PokeCubeObjectBlack-v1', 'PokeCubeObjectPurple-v1', 'PokeCubeColorTempLower50-v1', 'PokeCubeColorTempHigher50-v1', 'RollBallLightStronger50-v1', 'RollBallLightWeaker50-v1', 'RollBallObjectBlack-v1', 'RollBallObjectPurple-v1', 'RollBallColorTempLower50-v1', 'RollBallColorTempHigher50-v1', 'StackCubeLightStronger50-v1', 'StackCubeLightWeaker50-v1', 'StackCubeObjectBlack-v1', 'StackCubeObjectPurple-v1', 'StackCubeColorTempLower50-v1', 'StackCubeColorTempHigher50-v1', 'PlaceSphereLightStronger50-v1', 'PlaceSphereLightWeaker50-v1', 'PlaceSphereObjectBlack-v1', 'PlaceSphereObjectPurple-v1', 'PlaceSphereColorTempLower50-v1', 'PlaceSphereColorTempHigher50-v1']`。

具体地，我希望你并行地测试和报告我们方法在这些workload上的30分钟训练精度：
- 借鉴`train/toy_cnn/ours_single_agent/online_rl_cl.py`和`train/toy_cnn/ours_single_agent/online_rl_ours_single_agent_cl.sh`来验证；
- 为验证一个Env的效果（例如'PickCubeObjectPurple-v1'），只需要将`train/toy_cnn/ours_single_agent/online_rl_ours_single_agent_cl.sh`中的设置改为`--envs-id "['PickCubeObjectPurple-v1']"`和`--env-change-time-points "[30]"`；
- 不能修改其它超参数，例如并行环境数、学习率等等；
- 你可以充分利用卡0至卡6，同时对多个Env开展测试；
- 最终我想看到的是两类文件：
    - 一个文件夹，里面有很多图片，每个图片即在每个Env上进行测试的精度曲线，x轴为时间，y轴为success_once和success_at_end；
    - 一个csv文件，里面汇总了每个Env上30分钟训练能够带来的精度提升数值，以及30分钟内的平均精度
