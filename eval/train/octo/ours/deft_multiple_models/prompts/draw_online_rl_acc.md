# 可视化online_rl的成功率

编写一个python脚本，输入若干个online RL的日志目录（例如`ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-080331-ours`）：
- 直接在代码中使用json数组指定日志目录，例如`data_logs = {'method1': 'xxx', 'method2': 'xxx'}`，key为方法名、value为日志目录

然后画一张折线图：
- x轴是相对训练开始的时间（单位为分钟）
- y轴是所有agent的平均`success_end`
- 图中有若干条线，每条线分别对应一个日志目录
- 将图片保存到输入的第一个日志目录中
