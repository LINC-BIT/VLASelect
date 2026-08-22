# 比较ppo_gen在压缩模型、原始模型、原始模型+PEFT的性能

`train/toy_cnn/ppo_gen/online_rl_ppo_gen.sh`已经实现了ppo_gen在压缩模型上进行训练，并且测试结果在`ckpt/cl_suite/20260428-ours-ue2-edgeta06-convertnet-rerun/ppo_gen`中已有记录。

我还想比较ppo_gen在原始模型上进行训练所有参数、ppo_gen在原始模型上使用PEFT技术训练少部分参数的精度和速度。请你参考`train/toy_cnn/ppo_gen/online_rl_ppo_gen.sh`和`train/toy_cnn/ppo_gen/online_rl.py`，在`train/toy_cnn/other_test/original_model`和`train/toy_cnn/other_test/original_model_peft`下实现这两种方法，并分别在卡2和卡3上运行这两个方法。注意，模型中大部分为卷积层，针对卷积层也需要进行PEFT。

运行时：
- 仿照CL Suite的画图方式，每100秒画一次图，里面三根线分别是ppo_gen在压缩模型、原始模型、原始模型+PEFT的精度；
- 新建一个csv，在其中统计三种方法的运行速度。

## 实现要求

- 不允许修改和删除已有代码；
- 不需要做任何git提交；
- 你可以使用卡2来进行代码验证；
- 实现完成后，将实现过程文档写入`train/toy_cnn/other_test/codex_run_logs/impl/${DATETIME}.md`。
