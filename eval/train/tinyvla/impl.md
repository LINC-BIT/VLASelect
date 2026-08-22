模仿train/vla_adapter_new中ours/online_rl_cl.py的实现方式（即一个python文件+一个启动脚本文件），将其迁移至train/edgevla中、用于在OpenCabinetDrawerEasyLevel0-v1等Env序列和模型EdgeVLA上进行训练。这两者的区别，你可以通过比较`train/edgevla/model_impl/online_rl_open_cabinet_drawer.py`和`train/vla_adapter_new/model_impl/workload_verify/online_rl_hold_cube_in_hand.py`得到，应该区别不大。你还可以参考train/edgevla/ppo_gen的实现方式。

## 实现要求

- 迁移后启动脚本的超参数与train/edgevla/model_impl/run_online_rl_open_cabinet_drawer.sh一致，`--envs-id`写为`"['OpenCabinetDrawerCabinet1021Default-v1','OpenCabinetDrawerCabinet1016ScaleUp1p3-v1','OpenCabinetDrawerCabinet1027Default-v1','OpenCabinetDrawerCabinet1016ScaleUp1p3-v1','OpenCabinetDrawerCabinet1032Default-v1','OpenCabinetDrawerCabinet1033ScaleUp1p3-v1','OpenCabinetDrawerCabinet1027Default-v1','OpenCabinetDrawerCabinet1021Default-v1','OpenCabinetDrawerCabinet1032Default-v1','OpenCabinetDrawerCabinet1033ScaleUp1p3-v1']"`（一样也没关系，主要是为了测试代码是否能正常运行），env更新时间点`--env-change-time-points`写为`"[31,62,96,131,151,163,207,247,271,300]"`。
- 实现完成后，你需要运行冒烟测试，直到运行完成无误、测试通过；
- 你如果直接运行启动脚本进行测试的话，需要注意LAUNCH_DIRECT的设置，设置为0可能导致你读取不到相应输出；
- 不允许修改和删除已有代码；
- 不需要做任何git提交。
