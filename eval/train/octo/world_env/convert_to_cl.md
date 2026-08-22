# 将目前的单Env上的online RL改为多Env上的持续online RL

`train/toy_cnn/world_env`文件夹下是基准方法world_env的实验代码，目前仅在单个Env（通过`--env-id`指定）上进行online RL。我现在想将其改为**多个Env连续出现**的场景，在多个Env上对同一个模型顺序进行online RL。

## 实现要求

- 我希望你不要用for循环来实现多个Env的连续出现。我希望你在目前代码的基础上，用定时重新实例化env的方式实现。具体来说，目前env的实例化（即`gym.make()`）是在online RL开始之前一次性进行，在训练过程中就不再改变。我希望你根据启动脚本传入的env序列（例如`--envs-id`，其值为`['PickCubeLightStronger50-v1', 'PickCubeLightWeaker50-v1', ...]）和env更新时间点（例如`--env-change-time-points`，其值为`[20, 35, ...]`、代表在第20分钟和第35分钟左右切换为下一个env进行训练和评估），在训练过程中定时重新构造envs和eval_envs；
- 使用`import workloads.table_top`导入可能的env实现，避免`gym.make()`找不到环境；
- `--envs-id`和`--env-change-time-points`的长度相等。`--env-change-time-points`的第一个值代表将第一个环境切换为第二个环境的时间；`--env-change-time-points`的最后一个值代表online RL结束运行、退出实验的时间；
- 以上两条要求的实现，可参考或者最好直接照搬`train/toy_cnn/ours_single_agent/online_rl_cl.py`中的实现方法，以保证基准方法和我自己方法实验设置的完全一致；
- 对应的启动脚本也需要更新，使用和`train/toy_cnn/ours_single_agent/online_rl_ours_single_agent_cl.sh`一致的`--envs-id`和`--env-change-time-points`；
- 实现完成后，使用设置`--envs-id "['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']"`和`--env-change-time-points "[2,4,6,8,10,12,14,16,18,20]"`进行冒烟试验，确认能跑通，在10个给定的env上都能跑通、顺利结束训练。
- 注意，你可能会遇到使用Env时报错的问题，如果报错关于`extra`，那么可以参考`train/toy_cnn/ours_single_agent/online_rl_cl.py`进行修复，例如其`_resolve_extra_tensor()`。

## 注意事项

- 实现完成后，将你的实现过程以及修改前后代码的diff结果写入文档`train/toy_cnn/world_env/codex_run_logs/convert_to_cl/${DATETIME}.md`；
- 实现过程中你可以使用卡7验证运行效果；
- 你不能删除任何已有的文件；
- 你不需要做任何的git提交。
