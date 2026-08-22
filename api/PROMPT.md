针对不同的模型，我现在的算法代码是分开写的，不好维护。例如，针对VLA-Adapter的启动脚本位于eval/train/vla_adapter_new/ours/run_online_rl_cl.sh，针对TinyVLA的启动脚本位于eval/train/tinyvla/ours/run_online_rl_cl.sh，相应的训练python文件是分开的。

我现在希望你能够抽象出一个统一的API接口，其是一个抽象类，名为`VLAModelInterface`。用户只要集成该抽象类，实现其中的接口，就能够将一个新的模型接入我们的方法。

现在，请你完成下列工作：
- 读取eval/train/vla_adapter_new/ours/run_online_rl_cl.sh和eval/train/tinyvla/ours/run_online_rl_cl.sh，分析其对应的训练python文件的共同点、和为了兼容不同模型的不同实现
- 从中抽取出一个抽象类`VLAModelInterface`，用于定义为了兼容不同模型的不同实现，写入`api/vla_model_interface.py`，并用英文注释写明该抽象类中每个需要实现的函数需要怎么实现、功能是什么
- 建立`api/vla_model_interface_examples`文件夹，下面有：
    - 两个文件`api/vla_model_interface_examples/vla_adapter_impl.py`和`api/vla_model_interface_examples/tinyvla_impl.py`，分别是继承了`VLAModelInterface`，用于实现两个模型接入的逻辑。
    - 对应的还有两个启动脚本`api/vla_model_interface_examples/vla_adapter_impl_verify.sh`和`api/vla_model_interface_examples/tinyvla_impl_verify.sh`，分别是用来验证两个模型接入后进行在线训练的效果。这两个启动脚本运行后，与eval/train/vla_adapter_new/ours/run_online_rl_cl.sh的功能完全一致，区别只是在于内部代码的实现逻辑不同，新脚本是利用抽象类来做实现的。即，用于训练的环境、超参数等设置，都沿用eval/train/vla_adapter_new/ours/run_online_rl_cl.sh的；把TinyVLA也接入eval/train/vla_adapter_new/ours/run_online_rl_cl.sh的运行环境和设置。这样，才能体现，通过统一的抽象类接口，能够将不同的模型接入同一个训练环境中进行训练和比较。
    - 同时，提供`MWE=1 bash ./<test script>`的选项，即MWE=1时每个启动脚本可以在5分钟内运行完成（参考README.md）
    - 除此之外，还可以有其它的辅助文件。原则就是，你不能改动除了api文件夹下的其它任何文件！如果需要改动，则将其复制到api文件夹中改动并引用。坚决不能改动api文件夹之外的任何文件！请保证api文件夹下的脚本和程序能够独立运行，不依赖其它文件夹。

在实现之前，你先告诉我，你计划使`VLAModelInterface`中包含哪些需要用户实现的接口，一一列举出来，待我确认后再进行最终的实现。
