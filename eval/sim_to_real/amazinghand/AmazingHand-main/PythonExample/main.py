from base import Runner, RobotInterface, Kinematics
from pathlib import Path
from PIL import Image
import time

# sudo chmod 666 /dev/ttyUSB0
# 10                     s  f
# 开灯 + 抓鼠标            7  3
# 开灯 + 抓水瓶            8  2
# 开灯 + 抓游戏手柄         8  2
# 开灯 + 抓电源插头         9  1
# 关灯 + 抓鼠标            6  4
# 关灯 + 抓水瓶            8  2
# 关灯 + 抓游戏手柄         7  3
# 开灯 + 抓鼠标 + 手机灯    7  3
# 开灯 + 抓水瓶 + 手机灯    9  1
# 开灯 + 抓游戏手柄 + 手机灯 7  3

if __name__ == "__main__":
    # URDF_PATH = str(Path(__file__).parent.resolve() / "urdf/dofbot.urdf")
    # EE_LINK = "gripper_tcp"  # !!! replace with your actual end-effector link name
    MODEL_PATH = str(Path(__file__).parent.resolve() / "models")
    COM = "/dev/ttyUSB0"
    freq = 10
    # goal_pos = [-0.44, 0.1, 0.2]  # Example goal position, adjust as needed
    # o_pose = [-0.615, 0, 0.005, 0.707107, 0, 0, -0.707107]
    robot_interface = RobotInterface(COM, freq)
    runner = Runner(MODEL_PATH, robot_interface, freq)
    input()
    runner.run(step=50)
    # kin = Kinematics(URDF_PATH, EE_LINK)
    # for i in range(10000):
    #     qpos_rad = robot_interface.get_joint_angles()
    #     tcp_pose, tcp_to_goal_pos = kin.fk(qpos_rad, goal_pos, robot_interface.o_pose)
    #     robot_interface.step()
    #     # print(tcp_pose)
    #     print(qpos_rad)
    #     time.sleep(0.1)
        # input()
    # Image.fromarray(robot_interface.get_image()[0]).save('1.jpg')