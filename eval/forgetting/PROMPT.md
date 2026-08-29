模仿eval/acc_comparison/run_acc_task_env_change_single_arm_robot.sh，在eval/forgetting中编写measure_forgetting.sh。其环境设置、超参数、训练逻辑等等与eval/acc_comparison/run_acc_task_env_change_single_arm_robot.sh完全一致。

区别是：
- 我希望报告的是模型在**过去task/env**上的精度。即，第一个Env结束后，统计模型在第一个Env上的平均精度；第二个Env结束后，统计模型在第一和第二个Env上的平均精度，以此类推。不用报告以时间为x轴的精度数据，只需要报告以Env序号为x轴、y轴为精度的数据。最后在终端输出ours/vlaselect相比于其它每个方法的精度提升百分比，输出为表格，行数等于基准方法数，列数等于经过训练的环境数+1（最后一列是在所有Env上的平均精度提升百分比）。
- MWE=1模式下，首先遵循eval/acc_comparison/run_acc_task_env_change_single_arm_robot.sh及其相关训练python脚本中有关MWE=1的设置。然后在时间设置上改为：每个Env的训练时间（rollout+update）上限为60秒。然后MWE模式下，每个方法只需要在前3个Env上训练，完成后即退出。
- measure_forgetting.sh只需要串行运行self_improv,vla_rft,world_env,vlaselect四个方法即可。

我希望你不去修改任何现有的文件，如果需要修改，那么都拷贝到eval/forgetting之后再修改。

需求选择：
- 过去 Env 精度采用方案 1A：每完成一个 Env，使用当前模型重新评测所有已完成 Env，再计算平均精度。结果解析器同时兼容训练脚本写入的逐 Env 精度字段；若字段不存在，则回退到该 Env 阶段最后一次 `success_once` 记录并明确标记为回退数据。
- 精度指标采用方案 2A：`success_once`。
- 提升百分比采用方案 3B：绝对百分点差 `(VLASelect - baseline) * 100`，终端以 `pp` 输出。
