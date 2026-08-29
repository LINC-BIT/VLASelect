我想要验证我们的方法能够支持CNN或MLP的训练。

新建文件夹api/model_type，在其中新加cnn.sh，用于验证我们的方法能够支持CNN的训练。该脚本可以完全复制eval/train/octo/ours_single_agent/online_rl_ours_single_agent_cl.sh及其相关文件，因为这个脚本和workload就是在CNN上跑的。你可以把所有相关的文件（不包括模型权重）都拷贝到api/model_type中，保证api/model_type中的代码是可以独立运行的，不需要依赖于其它文件夹，与其它文件夹隔离。运行完成后，自动绘图，绘图逻辑和eval/acc_comparison/plot_acc_task_env.py保持一致，固定将图片保存至api/model_type/CNN-ACC.png。

然后新加mlp.sh，和cnn.sh保持一模一样的逻辑（例如小模型生成和反馈等，以及MWE=1模式），把模型换成官方 PPO 示例的 state-only MLP。运行完成后固定将图片保存至api/model_type/MLP-ACC.png。

原则：
- 你不能改动除了api文件夹下的其它任何文件！如果需要改动，则将其复制到api文件夹中改动并引用。坚决不能改动api文件夹之外的任何文件！请保证api文件夹下的脚本和程序能够独立运行，不依赖其它文件夹。
- 不需要关注git提交状态等，我来手动提交。

## 已确认的实现约定

- 独立运行的边界采用选项 1A：将运行所需的 Python 源码、配置和绘图代码复制到 `api/model_type`；允许依赖已安装的第三方 Python 包，以及通过原脚本路径使用仓库其它目录中的模型权重。
- 各类 checkpoint、state normalization 和环境配置路径保持与 `eval/train/octo/ours_single_agent/online_rl_ours_single_agent_cl.sh` 完全一致。
- MLP 采用官方 PPO 示例结构：只使用 state 输入，actor 和 critic 各自为 3 个 256 宽度的 Tanh 隐藏层，action/value 输出接口和 CNN 模型保持一致。
- MLP 的 PPO 预训练第一阶段使用 dense 模型；FBS 第二阶段在 actor 和 critic 各自第 1、3 个隐藏 Linear 层接入 FBS，固定稀疏度为 `0.1`。
- 绘图采用选项 5A：生成单张训练成功率曲线图，绘图逻辑与 `eval/acc_comparison/plot_acc_task_env.py` 保持一致，并分别固定输出到 `api/model_type/CNN-ACC.png` 和 `api/model_type/MLP-ACC.png`。
- 脚本启动采用选项 6A：可从任意当前工作目录启动，代码、日志、metrics 和图片路径均相对于 `api/model_type` 解析。
- MWE 不调整配置，完全沿用 `eval/train/octo/ours_single_agent/online_rl_cl.py` 中已经定义的 MWE 行为。
