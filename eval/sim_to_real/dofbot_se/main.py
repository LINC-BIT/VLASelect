from base import Runner, RobotInterface, Kinematics
from pathlib import Path
from PIL import Image
import time

# sudo chmod 666 /dev/ttyUSB0

if __name__ == "__main__":
    URDF_PATH = str(Path(__file__).parent.resolve() / "urdf/dofbot.urdf")
    EE_LINK = "gripper_tcp"  # !!! replace with your actual end-effector link name
    MODEL_PATH = str(Path(__file__).parent.resolve() / "model/best_agent.pt")
    COM = "/dev/ttyUSB0"
    freq = 2
    goal_pos = [-0.44, 0.0, 0.2]  # Example goal position, adjust as needed
    o_pose = [-0.615, 0, 0.005, 0.707107, 0, 0, -0.707107]
    robot_interface = RobotInterface(COM, o_pose, freq)
    runner = Runner(URDF_PATH, EE_LINK, MODEL_PATH, robot_interface, goal_pos, freq)
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

# 10                   s    f
# 开灯+蓝色块            8    2  
# 开灯+红色块            7    3
# 开灯+绿色块            8    2
# 开灯+黄色块            9    1
# 关灯+蓝色块            7    3
# 关灯+红色块            6    4
# 关灯+绿色块            6    4
# 开灯+手机灯+蓝色块      8    2
# 开灯+手机灯+红色块      8    2
# 开灯+手机灯+绿色块      8    2