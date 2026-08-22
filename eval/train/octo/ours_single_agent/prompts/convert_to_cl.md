# 将目前的单Env上的online RL改为多Env上的持续online RL

`train/toy_cnn/ours_single_agent`文件夹下是我自己方法的实验代码，目前仅在单个Env（通过`--env-id`指定）上进行online RL。我现在想将其改为**多个Env连续出现**的场景，在多个Env上对同一个模型顺序进行online RL。

## 实现要求

- 我希望你不要用for循环来实现多个Env的连续出现。我希望你在目前代码的基础上，用定时重新实例化env的方式实现。具体来说，目前env的实例化（即`gym.make()`）是在online RL开始之前一次性进行，在训练过程中就不再改变。我希望你根据启动脚本传入的env序列（例如`--envs-id`，其值为`['PickCubeLightStronger50-v1', 'PickCubeLightWeaker50-v1', ...]）和env更新时间点（例如`--env-change-time-points`，其值为`[20, 35, ...]`、代表在第20分钟和第35分钟左右切换为下一个env进行训练和评估），在训练过程中定时重新构造envs和eval_envs；
- `--envs-id`和`--env-change-time-points`的长度相等。`--env-change-time-points`的第一个值代表将第一个环境切换为第二个环境的时间；`--env-change-time-points`的最后一个值代表online RL结束运行、退出实验的时间。

## 注意事项

- 实现完成后，将你的实现过程以及修改前后代码的diff结果写入文档`train/toy_cnn/prompts/codex_run_logs/convert_to_cl/${DATETIME}.md`；
- 实现过程中你可以使用卡1验证运行效果；
- 你不能改动任何已有的文件；
- 你不能删除任何已有的文件；
- 你不需要做任何的git提交。
