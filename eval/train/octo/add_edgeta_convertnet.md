# 新增两个baselines：EdgeTA和ConvertNet

两个baselines都是ours_single_agent的消融版本。你需要为两个baselines分别新增两个文件夹`train/toy_cnn/edgeta`和`train/toy_cnn/convertnet`，但是其中不用包含新的python脚本，直接使用`train/toy_cnn/ours_single_agent/online_rl_cl.py`即可。

EdgeTA的实验启动脚本和`train/toy_cnn/ours_single_agent/online_rl_ours_single_agent_cl.sh`的区别是，将以下参数改为下列值：
```bash
--small_model_generation_strategy target-single-traj
--small_model_feedback_schedule before_per_rollout
--small_model_regeneration_schedule before_per_rollout
--small_model_feedback_alpha 1.0
--small_model_regeneration_increment_ratio 1.0
--reset_optimizer_after_regeneration
```

ConvertNet的实验启动脚本和`train/toy_cnn/ours_single_agent/online_rl_ours_single_agent_cl.sh`的区别是，将以下参数改为下列值：
```bash
--small_model_generation_strategy target-single-traj
--small_model_feedback_schedule before_per_rollout
--small_model_regeneration_schedule before_per_rollout
--small_model_feedback_alpha 0.5
--small_model_regeneration_increment_ratio 1.0
--reset_optimizer_after_regeneration
```

## 实现要求

- 不允许修改和删除已有代码；
- 不需要做任何git提交；
- 你可以使用卡0来进行代码验证；
- 实现完成后，将实现过程文档写入`train/toy_cnn/edgeta/codex_run_logs/impl/${DATETIME}.md`和`train/toy_cnn/convertnet/codex_run_logs/impl/${DATETIME}.md`。
