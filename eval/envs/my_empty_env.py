import numpy as np
import sapien
import torch

from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig, SceneConfig
from mani_skill.utils.building import actors
import pandas as pd
import os
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

@register_env("MyEmpty-v1", max_episode_steps=200000)
class MyEmptyEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    """
    This is just a dummy environment for showcasing robots in a empty scene
    """

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.data = {}

    @property
    def _default_sensor_configs(self):
        # 更近、更低、看向机器人中心
        pose = sapien_utils.look_at([0.5, -0.5, 0.5], [0.0, 0.0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose,
                width=128,
                height=128,
                fov=np.pi / 4,   # 缩小视野，让机器人更大
                near=0.01,
                far=100
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0, -0.7, 0.4], [0.0, 0.0, 0.1])
        # pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera",
            pose,
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=100
        )
        # return self.agent._sensor_configs[0]

    def _load_agent(self, options: dict):
        if self.robot_uids == "amazinghand_right_cpu" or self.robot_uids == "amazinghand_right":
            q = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([0, -np.pi / 2, -np.pi / 2]), convention="XYZ"))
            o_pose = sapien.Pose(
                p=[0, 0, 0.2], 
                q=q
            )
            super()._load_agent(options, o_pose)

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)
        self.cube = actors.build_cube(
            self.scene,
            half_size=0.015,
            color=[0, 1, 0, 1],
            name="cube1",
            body_type="dynamic",
            # body_type="kinematic",
            initial_pose=sapien.Pose([-0.0196, -0.2843,  0.0124]),
        )
        self.cube.disable_gravity = True
        self.cube.set_mass(0.1)

        if not hasattr(self.agent, "marker1"):
            self.marker1 = actors.build_sphere(
                self.scene,
                radius=0.005,
                color=[1, 0, 0, 1],
                name="marker1",
                body_type="kinematic",
                
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            self.marker1.set_disable_gravity(True)

        if not hasattr(self.agent, "marker3"):
            self.marker3 = actors.build_sphere(
                self.scene,
                radius=0.005,
                color=[1, 0, 0, 1],
                name="marker3",
                body_type="kinematic",
                
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            self.marker3.set_disable_gravity(True)

        if not hasattr(self.agent, "marker2"):
            self.marker2 = actors.build_sphere(
                self.scene,
                radius=0.005,
                color=[1, 0, 0, 1],
                name="marker2",
                body_type="kinematic",
                
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            self.marker2.set_disable_gravity(True)

        if not hasattr(self.agent, "marker4"):
            self.marker4 = actors.build_sphere(
                self.scene,
                radius=0.005,
                color=[1, 0, 0, 1],
                name="marker4",
                body_type="kinematic",
                
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            self.marker4.set_disable_gravity(True)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 创建 TCP 可视化球
        if hasattr(self, "marker1"):
            self.marker1.set_pose(self.agent.ee_link.pose)  # 跟随 TCP

        if hasattr(self, "marker2"):
            self.marker2.set_pose(self.agent.ee_link_1.pose)  # 跟随 TCP

        if hasattr(self, "marker3"):
            self.marker3.set_pose(self.agent.ee_link_2.pose)  # 跟随 TCP

        if hasattr(self, "marker4"):
            self.marker4.set_pose(self.agent.ee_link_3.pose)  # 跟随 TCP
        # if hasattr(self.agent, "tcp_pose"):
            # if not hasattr(self, "tcp_marker"):
            #     self.tcp_marker = actors.build_sphere(
            #         self.scene,
            #         radius=0.005,
            #         color=[1, 0, 0, 1],
            #         name="tcp_marker",
            #         body_type="kinematic",
                    
            #         add_collision=False,
            #         initial_pose=self.agent.tcp_pose.raw_pose,
            #     )
            #     self.tcp_marker.set_disable_gravity(True)
            #     self.marker1 = actors.build_sphere(
            #         self.scene,
            #         radius=0.005,
            #         color=[1, 0, 0, 1],
            #         name="marker1",
            #         body_type="kinematic",
                    
            #         add_collision=False,
            #         initial_pose=sapien.Pose(p=self.agent._drive_pos(1)),
            #     )
            #     self.marker1.set_disable_gravity(True)

            #     self.marker2 = actors.build_sphere(
            #         self.scene,
            #         radius=0.005,
            #         color=[1, 0, 0, 1],
            #         name="marker2",
            #         body_type="kinematic",
                    
            #         add_collision=False,
            #         initial_pose=sapien.Pose(p=self.agent._drive_pos(2)),
            #     )
            #     self.marker2.set_disable_gravity(True)

            #     # self.cube.set_pose(self.agent.tcp_pose.raw_pose)

            #     self.root_marker = actors.build_sphere(
            #         self.scene,
            #         radius=0.03,
            #         color=[0, 0, 1, 1],
            #         name="root_marker",
            #         body_type="kinematic",
            #         add_collision=False,
            #         initial_pose=sapien.Pose([0, 0, 0]),
            #     )
            #     self.root_marker.set_disable_gravity(True)
            # else:
            #     # reset 只更新位置
            #     self.tcp_marker.set_pose(self.agent.tcp_pose)
            #     self.marker1.set_pose(sapien.Pose(p=self.agent._drive_pos(1)))
            #     self.marker2.set_pose(sapien.Pose(p=self.agent._drive_pos(2)))
            #     # self.cube.set_pose(self.agent.tcp_pose.raw_pose)
        pass

    # @property
    # def _default_sim_config(self):
    #     return SimConfig(
    #         scene_config=SceneConfig(      
    #             solver_position_iterations=40, 
    #             solver_velocity_iterations=10,
    #             # contact_offset=0.02,
    #         ),
    #         # gpu_memory_config=GPUMemoryConfig(
    #         #     max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
    #         #     max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
    #         #     found_lost_pairs_capacity=2**26,
    #         # )
    #     )

    def evaluate(self):
        # is_grasped = self.agent.is_grasping(self.cube)
        # is_robot_static = self.agent.is_static(0.2)
        # return {
        #     "is_robot_static": is_robot_static,
        #     "is_grasped": is_grasped,
        # }
        return {}
    
    def _get_obs_extra(self, info: dict):
        # 每步更新 TCP 球位置
        
        if hasattr(self, "tcp_marker"):
            self.tcp_marker.set_pose(self.agent.tcp_pose)  # 跟随 TCP

        return dict(tcp_pose=self.agent.tcp_pose)
    
    @property
    def _default_sim_config(self):
        return SimConfig(
            scene_config=SceneConfig(      
                solver_position_iterations=40, 
                solver_velocity_iterations=20,
                contact_offset=0.002,
            ),
            sim_freq=200,
            # gpu_memory_config=GPUMemoryConfig(
            #     max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
            #     max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
            #     found_lost_pairs_capacity=2**26,
            # )
        )

    def _before_control_step(self):
        if hasattr(self, "tcp_marker"):
            self.tcp_marker.set_pose(self.agent.tcp_pose)  # 跟随 TCP
            self.marker1.set_pose(sapien.Pose(p=self.agent._drive_pos(1)))
            self.marker2.set_pose(sapien.Pose(p=self.agent._drive_pos(2)))
        # contacts = self.scene.get_contacts()

        if hasattr(self, "marker1"):
            self.marker1.set_pose(self.agent.ee_link.pose)  # 跟随 TCP

        if hasattr(self, "marker2"):
            self.marker2.set_pose(self.agent.ee_link_1.pose)  # 跟随 TCP

        if hasattr(self, "marker3"):
            self.marker3.set_pose(self.agent.ee_link_2.pose)  # 跟随 TCP

        if hasattr(self, "marker4"):
            self.marker4.set_pose(self.agent.ee_link_3.pose)  # 跟随 TCP

        # self.cube.set_pose(self.agent.tcp_pose.raw_pose)

        # target = [
        #     x.get_drive_target()
        #     for x in self.agent.right_drive._objs
        # ][0]
        # current = self.agent.right_drive.pose_in_parent.inv() * self.agent.right_drive.pose_in_child

        # print(target, current)

        # if contacts:
        #     print("碰撞检测到以下 link 对：")
        #     for contact in contacts:
        #         link1 = getattr(contact.bodies[0], "name", str(contact.bodies[0]))
        #         link2 = getattr(contact.bodies[1], "name", str(contact.bodies[1]))
        #         if "gripper" in link1 or "gripper" in link2:
        #             print(f"  {link1} <-> {link2}")

        # print(self.agent.tcp_pose)
        # print(self.agent.is_grasping(self.cube))
        # p_f_right, p_p_right, p_f_left, p_p_left = self.agent._get_gripper_constraint()
        # self.agent._get_constraint_p()
        # print("p_f_right = ", p_f_right.cpu().tolist())
        # print("p_p_right = ", p_p_right.cpu().tolist())
        # print("p_f_left = ", p_f_left.cpu().tolist())
        # print("p_p_left = ", p_p_left.cpu().tolist())

        # if self.agent.is_grasping(self.tcp_marker):
        #     self.tcp_marker.set_disable_gravity(False)
        # robot: sapien.Articulation
        # left_joints = [
        #     self.agent.robot.joints_map[f"rlink_joint2"],
        # ]
        # right_joints = [
        #     self.agent.robot.joints_map[f"llink_joint2"],
        # ]

        # 获取当前角度
        # q_left = [j.qpos for j in left_joints]   # list of floats
        # q_right = [j.qpos for j in right_joints]

        # 如果左右是对称的，通常需要乘 -1（机械对称）
        # q_right_sym = [-x for x in q_right]

        # 计算角度差
        # for i, (ql, qr) in enumerate(zip(q_left, q_right_sym), 1):
        #     print(f"Joint {i}: Left={ql}, Right_sym={qr}, diff={ql-qr}")

        # pass

    def get_regression_data(self):
        names = ['A', 'B', 'roll', 'yaw', 'distal', 'proximal']
        def get_data(finger_name, joints):
            if finger_name not in self.data:
                self.data[finger_name] = {name: [] for name in names}
            for name, joint_name in zip(names, joints):
                joint = self.agent.robot.joints_map[joint_name]
                self.data[finger_name][name].append(joint.qpos[0].cpu().item())

        # finger 1:
        joints = ["revolute_5_1", "revolute_6_1", "revolute_3_2", "revolute_1_1", "cylindrical_1_5", "revolute_2_1"]
        get_data('finger1', joints)
        
        # finger 2:
        joints = ["revolute_5_3", "revolute_6_3", "revolute_3_4", "revolute_1_4", "cylindrical_1_11", "revolute_2_3"]
        get_data('finger2', joints)

        # finger 3:
        joints = ["revolute_5_0", "revolute_6_0", "revolute_3_1", "revolute_1_0", "cylindrical_1_2", "revolute_2_0"]
        get_data('finger3', joints)

        # finger 4:
        joints = ["revolute_5_2", "revolute_6_2", "revolute_3_3", "revolute_1_2", "cylindrical_1_8", "revolute_2_2"]
        get_data('finger4', joints)

    def _after_simulation_step(self):
        super()._after_simulation_step()
        self.get_regression_data()
        
    def save_data(self, filepath):
        for finger_name, finger_data in self.data.items():
            df = pd.DataFrame(finger_data)
            df.to_csv(os.path.join(filepath, f"{finger_name}.csv"), index=False)