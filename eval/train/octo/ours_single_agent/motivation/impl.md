# 比较原始模型和小模型的精度

在`train/toy_cnn/ours_single_agent/online_rl_cl.py`的基础上，我想比较训练原始模型和压缩静态小模型在同一个实验设置下的精度。

## 实现要求

- 复制`train/toy_cnn/ours_single_agent/online_rl_cl.py`到`train/toy_cnn/ours_single_agent/motivation/original_model.py`，略作修改以训练原始模型：
    - 将`args.max_sparsity`设为0，并且直接用原始模型进行训练，不使用`generate_small_cnn_with_verify()`生成静态小模型。在训练过程中始终仅训练该原始模型，不进行任何的模型生成和反馈操作；
    - 记录训练过程中，每个原始模型神经元的重要性。每个神经元指卷积层中的一个filter或线性层中的一个行向量。每个神经元的重要性等于`abs(其权重乘以其梯度)`，平均为一个浮点数。将统计结果保存为文件；
- 复制`train/toy_cnn/ours_single_agent/online_rl_cl.py`到`train/toy_cnn/ours_single_agent/motivation/small_model.py`，略作修改以训练压缩静态小模型：
    - 将`args.max_sparsity`设为0.8，然后使用`--small_model_generation_strategy 'target-single-traj'`生成静态小模型。注意，在训练过程中始终仅训练最初生成的压缩静态小模型，不进行任何的模型生成和反馈操作；
    - 记录训练过程中，压缩静态小模型中每个神经元的重要性。每个神经元指卷积层中的一个filter或线性层中的一个行向量。每个神经元的重要性等于`abs(其权重乘以其梯度)`，平均为一个浮点数。将统计结果保存为文件；
- 两个python文件分别编写启动脚本运行，启动脚本参考`train/toy_cnn/ours_single_agent/online_rl_ours_single_agent_cl.sh`；
- workload的设置统一为：
    - `--envs-id`设为`['PokeCubeLightWeaker50-v1','PushCubeColorTempLower50-v1','StackCubeObjectBlack-v1','StackCubeLightWeaker50-v1','StackCubeColorTempLower50-v1','RollBallLightWeaker50-v1','RollBallObjectBlack-v1','PickCubeLightWeaker50-v1','RollBallColorTempHigher50-v1','PokeCubeColorTempLower50-v1']`
    - `--env-change-time-points`设为`[30, 60, 90, 120, 150, 180, 210, 240, 270, 300]`，即每半个小时切换一次环境
- 训练原始模型和压缩静态小模型的超参数保持一致，保证公平比较；
- 单独写一个画精度图的脚本，在训练过程中，同步画出精度图用于比较训练原始模型和压缩静态小模型的精度。图的x轴为时间、y轴为success_at_end、两条线分别代表原始模型和压缩静态小模型的精度。画图脚本在训练的同时每1分钟运行一次，将图片保存在`train/toy_cnn/ours_single_agent/motivation/res.png`中；
- 单独写一个可视化神经元重要性的脚本，在训练过程中，同步画图用于比较训练原始模型和压缩静态小模型中神经元的重要性：
    - 每次运行时，遍历输出图片`train/toy_cnn/ours_single_agent/motivation/res_images/original_model/${env_id}-${iteration}.png`和`train/toy_cnn/ours_single_agent/motivation/res_images/small_model/${env_id}-${iteration}.png`，其中`iteration`的范围是`range(0, 画图当时训练进度所在的iteration, 5)`；
    - 每张图片是一个矩阵热力图，图的第i行对应原始模型或压缩小模型的第i个卷积层或线性层，图的第j列对应层中的第j个神经元。i的范围是0至7，j的范围是0至7，即我们只选取前8层、每层的前8个神经元做可视化。

## 注意事项

- 分别使用卡2和卡3训练原始模型和压缩小模型；
- 你不能修改和删除任何已有的文件；
- 你不需要做任何的git提交。
