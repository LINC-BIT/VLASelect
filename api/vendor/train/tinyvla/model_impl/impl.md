# 在Maniskill Benchmark中接入EdgeVLA模型

## 实现目标

你需要实现、接入并加载EdgeVLA模型，并且使其：
- 能在Maniskill环境中读取环境数据、进行推理、输出动作；
- 能使用PPO方法，在OpenCabinetDrawer-v1这个Env中进行online RL训练。

## 实现参考资料

- 模型实现上：EdgeVLA和VLA-Adapter的**唯一区别**是：EdgeVLA一次推理即可输出所有action token（但和VLA-Adapter一样，先输出离散化的action token），而VLA-Adapter使用自回归的形式、需要多次推理依次输出所有action token。所以，你可以充分参考`train/vla_adapter_new/model_impl/online_rl_hold_cube_in_hand.py`、EdgeVLA的公开源码（`train/edgevla/model_impl/evla-main.zip`）、以及如下所示EdgeVLA论文原句，将`train/vla_adapter_new/model_impl/online_rl_hold_cube_in_hand.py`中VLA-Adapter的实现直接搬过来，针对该唯一区别稍作修改即可。下面是EdgeVLA论文的原句：
    > Traditional VLAs employ an autoregressive approach to predicting end-effector positions, mimicking the causal nature of language generation. However, we hypothesize that for robotic control, this restriction is not inherently necessary. We propose that predicting the entire end-effector position jointly, rather than sequentially, does not compromise the model’s encoding capabilities while significantly improving inference speed. By removing the causal mask in the LLM and training the model to output the entire end-effector position at once, we eliminate autoregressive requirements, achieving a six-times speedup in inference - a critical improvement for real-time applications on edge devices.
- 环境接入上：应该和之前的`HoldCubeInHand-v1`差不多，你确认一下；
- 训练算法PPO上：应该和之前应用于VLA-Adapter的PPO差不多，你确认一下action token输出方式的变化对PPO的实现有没有影响，有的话就将PPO对EdgeVLA做针对性的修改。


## 实现要求

- 尽量使用单文件来实现整个online RL的逻辑（类似`train/vla_adapter_new/model_impl/online_rl_hold_cube_in_hand.py`的做法），以保证实验代码的可复现性和避免不同方法代码之间的互相耦合（不过，用于接入和实现EdgeVLA模型的文件组织形式不限）；
- online RL的启动脚本中，训练超参数与`train/vla_adapter_new/model_impl/run_online_rl_hold_cube_in_hand.sh`保持完全一致；
- online RL时，如果可能的话，可加载VLA-Adapter的权重文件`train/vla_adapter_new/model_impl/outputs/ppo_hold_cube_in_hand/20260430-103518/best_policy.pt`初始化EdgeVLA的参数，以加快训练收敛；
- 不允许修改和删除已有代码；
- 你新增代码和修改代码仅能在文件夹`train/edgevla/model_impl`中进行；
- 不需要做任何git提交；
- 你可以使用卡4来进行代码运行和验证；
- 实现完成后，将实现过程文档写入`train/edgevla/model_impl/codex_run_logs/impl/${DATETIME}.md`。
