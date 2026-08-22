api/vla_model_interface_examples/vla_adapter_impl_verify.sh现在已支持block和layer粒度。

现在请再支持Attention head粒度：
- 对于attention中的QKV层，将同一个head的神经元（即连续的数个神经元，数量由一个head的dim决定）打包在一起，在生成小模型的过程中要么都移除要么都保留
- 对于其它层，使用层粒度

