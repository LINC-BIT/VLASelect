api/vla_model_interface_examples/vla_adapter_impl_verify.sh现已支持--scaling-method knowledge_distillation。我希望将现在的knowledge_distillation改名为logit_distillation，然后再支持更多的--scaling-method，如下：

- llm_pruner：与默认的小模型生成相同，区别是用于scaling的数据仅来自于source env（即HoldCubeInHand）
- logit_distillation
- feature_distillation：蒸馏中间层的隐藏状态
- attn_distillation：蒸馏注意力分布
- data_distillation：大模型生成数据后，小模型在该数据上进行训练
- minillm：通过反向KL散度来提升学生学习效率（可联网搜索论文MiniLLM: Knowledge Distillation of Large Language Models查看相关细节）
- distillm：提出新的KL loss（可联网搜索论文DistiLLM: Towards Streamlined Distillation for Large Language Models查看相关细节）
- llm_in_a_flash：与默认的小模型生成相同，但仅对FFN层进行部分神经元的提取（提取的比例需要比默认的比例少一半，以补偿Attention不进行压缩而没达到的整体压缩率），大小模型中的Attention层保持不变。
- powerinfer：与默认的小模型生成相同，区别是只在env初始化或env切换时触发一次小模型生成
- edgeta：与默认的小模型生成相同，区别是只在env初始化或env切换时触发一次小模型生成
