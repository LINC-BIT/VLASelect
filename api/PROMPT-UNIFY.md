我希望在api/vla_model_interface_examples/vla_adapter_impl_verify.sh的基础上，使这一个脚本就能支持对small model scaling方法和knowledge exchange方法的切换。例如：
```bash
MWE=1 bash api/vla_model_interface_examples/vla_adapter_impl_verify.sh --scaling-method knowledge_distillation
MWE=1 bash api/vla_model_interface_examples/vla_adapter_impl_verify.sh --knowledge-exchange-granularity block
```
目前这两个参数不会同时传递，每次只会指定scaling method或者knowledge exchange粒度。

small model scaling的相关代码在api/small_model_scaling_interface_examples。
knowledge exchange的相关代码在api/knowledge_exchange_interface_examples。

这两类方法在之后都可能会扩展出更多的选项。

然后我希望，如果指定了--scaling-method，那么其运行结果都存放到api/results/scaling_methods/<scaling_method>下面。我希望，你创建一个画图脚本，当我使用api/vla_model_interface_examples/vla_adapter_impl_verify.sh运行完所有的scaling methods之后，该画图脚本可以读取api/results/scaling_methods中的数据，就能够画出一个统一的折线图。其x轴是时间，y轴是训练精度，每条线代表每个scaling method。

对于--knowledge-exchange-granularity，同理，如果指定了，运行结果都存放到api/results/knowledge_exchange/<granularity>下面。同样，也创建一个统一的画图脚本，对api/results/knowledge_exchange中的数据进行画图。

我希望api/unified_online_rl.py不要进行eval，太费时间了，全程只关注训练精度即可。MWE=1时将训练时间控制在5分钟准时结束。

最后，对目前的knowledge_distillation方法做一下修改：其进行distill的目标小模型的参数应该被随机初始化后，再进行distill，这样更符合原始的知识蒸馏方法。（目前是继承了大模型的部分权重，应该对其进行随机初始化）。
