请按以下需求修改相关代码、增强Feature Aggregator的表征能力：
- 将Feature Aggregator中Attention的head数设置为可配置项
- gate改为两层MLP、中间加非线性激活函数（设置为可配置项，选项为'single-layer'（单层线性层为gate）或'two-layers'）
- 在你觉得必要的地方增加Norm层（设置为可配置项）

先列出你的实现计划，经过我确认后再写代码。
