我希望在MLP和CNN上，再新建两个脚本cnn-ppo-gen.sh和mlp-ppo-gen.sh，即将PPO-Gen方法应用在CNN和MLP上，与cnn.sh和mlp.sh比较精度。

PPO-Gen方法和我们方法的区别为：
- 不进行任何的模型重生成和反馈
- 模型初始化是利用我们方法（cnn.sh和mlp.sh）在源环境PickCube-v1上生成的小模型（你需要编写代码，先生成这两个模型，保存为api/model_type/ppo-gen-mlp-model和api/model_type/ppo-gen-cnn-model，然后通过参数的方式传入这两个模型的路径）
- --update_epochs改为1

请实现。不要改动任何api/model_type之外的代码。

## 已确认的实现约定

- 生成小模型使用源环境 `PickCube-v1`，沿用现有环境 JSON 配置；生成完成后将 FBS 模块折叠为普通 Linear/Conv 层，不保留稀疏度或动态门控。
- 新增独立生成流程，在 `PickCube-v1` 上各执行一次现有 VLASelect 小模型生成逻辑，生成结果固定保存为 `api/model_type/ppo-gen-cnn-model` 和 `api/model_type/ppo-gen-mlp-model`。
- `run.sh` 不重新生成上述模型；模型需要预先生成，四个实验脚本直接复用固定文件。
- CNN 生成使用 `cnn.sh` 当前默认 checkpoint，MLP 生成使用 `api/model_type/ckpt/mlp_pretrain/best.pt`。
- 新增 `cnn-conrft.sh` 和 `mlp-conrft.sh`（PPO-Gen 方法改名为 ConRFT），直接加载预生成普通小模型作为唯一 PPO agent；训练期间固定模型结构和参数形状，不执行任何额外模型处理、FBS、重生成或反馈。
- ConRFT 完全沿用 `cnn.sh`/`mlp.sh` 的目标 continual schedule；源环境只用于生成小模型。
- ConRFT 仅将 `update_epochs` 改为 `1`，其它参数和 MWE 行为沿用对应脚本。
- 生成模型保存为可由 ConRFT 直接加载的完整 PyTorch checkpoint，路径无扩展名。
- `run.sh` 只负责运行四个实验（可用 `MWE=0/1` 或命令行参数选择），四个实验可独立运行；模型生成不由 `run.sh` 负责。
- 合并图 `api/model_type/MLP-CNN-ACC-COMPARE.png` 包含 CNN、MLP 两个子图；每个子图绘制 `VLASelect` 和 `ConRFT` 的 `success_once` 曲线，横轴为 `Time (minutes)`，纵轴为 `Accuracy`。
- 合并图从四个实验的 metrics 文件自动读取数据，缺少实验数据时明确报错。`MWE` 只影响 PPO 训练规模，两个 ConRFT 脚本始终使用预生成模型。
