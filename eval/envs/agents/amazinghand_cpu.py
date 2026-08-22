"""Adapted from https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/cartpole.py"""

import os
from typing import Any, Optional, Union
from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization, rewards
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import (
    Array,
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)
from mani_skill.agents.registration import register_agent

@register_agent()
class AH_RIGHT(BaseAgent):
    uid = "amazinghand_right_cpu"
    urdf_path = Path(__file__).parent.resolve() / f'urdfs/amazinghand_right/right_hand_final.urdf'
    disable_self_collisions = True
    load_multiple_collisions = False

    all_active_joints = ['revolute_1_1', 'revolute_6_1', 'revolute_5_1', 'revolute_1_4', 'revolute_6_3', 'revolute_5_3', 'revolute_1_0', 'revolute_6_0', 'revolute_5_0', 'revolute_2_1', 'revolute_3_2', 'revolute_1_2', 'revolute_6_2', 'revolute_5_2', 'revolute_2_3', 'revolute_3_4', 'revolute_2_0', 'revolute_3_1', 'ball_1_17', 'ball_1_12', 'revolute_2_2', 'revolute_3_3', 'ball_1_37', 'ball_1_32', 'ball_1_7', 'ball_1_2', 'cylindrical_1_5', 'ball_1_18', 'ball_1_13', 'ball_1_27', 'ball_1_22', 'cylindrical_1_11', 'ball_1_38', 'ball_1_33', 'cylindrical_1_2', 'ball_1_8', 'ball_1_3', 'ball_1_19', 'ball_1_14', 'cylindrical_1_8', 'ball_1_28', 'ball_1_23', 'ball_1_39', 'ball_1_34', 'ball_1_9', 'ball_1_4', 'ball_1_29', 'ball_1_24']

    servo_joints = ["revolute_5_0", "revolute_5_1", "revolute_5_2", "revolute_5_3", "revolute_6_0", "revolute_6_1", "revolute_6_2", "revolute_6_3"]

    servo_damping = 5e2
    servo_stiffness = 5e3
    servo_friction = 1e-2

    def __init__(
        self,
        scene,
        control_freq: int,
        control_mode: Optional[str] = None,
        agent_idx: Optional[str] = None,
        initial_pose: Optional[Union[sapien.Pose, Pose]] = None,
        build_separate: bool = False,
    ):
        self.passive_joints = [x for x in self.all_active_joints if x not in self.servo_joints]
        self.touch_links = {}
        super().__init__(scene, control_freq, control_mode, agent_idx, initial_pose, build_separate)
        
    @property
    def _controller_configs(self):
        servo_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.servo_joints,
            lower=-0.2,
            upper=0.2,
            damping=self.servo_damping,
            stiffness=self.servo_stiffness,
            friction=self.servo_friction,
            use_delta=True,
        )
        servo_pd_joint_pos = PDJointPosControllerConfig(
            self.servo_joints,
            lower=None, 
            upper=None,
            damping=self.servo_damping,
            stiffness=self.servo_stiffness,
            friction=self.servo_friction,
        )
        rest = PassiveControllerConfig(self.passive_joints, damping=0, friction=0)
        return dict(
            pd_joint_delta_pos=dict(
                servo=servo_pd_joint_delta_pos, rest=rest, balance_passive_force=False
            ),
            pd_joint_pos=dict(
                servo=servo_pd_joint_pos, rest=rest, balance_passive_force=False
            )
        )

    def get_proprioception(self):
        qpos = torch.stack([joint.qpos for joint in self.controller.joints if joint.name in self.servo_joints], dim=-1)
        qvel = torch.stack([joint.qvel for joint in self.controller.joints if joint.name in self.servo_joints], dim=-1)
        return dict(qpos=qpos, qvel=qvel)

    def _after_init(self):
        touch_links = ["distal_shell", "proximal_shell", "distal_shell_1", "proximal_shell_1", "distal_shell_2", "proximal_shell_2", "distal_shell_3", "proximal_shell_3", "r_palm_shell"]
        
        for link_name in touch_links:
            link = sapien_utils.get_obj_by_name(self.robot.get_links(), link_name)
            self.touch_links[link_name] = link

        self.ee_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "ee_link"
        )

        self.ee_link_1 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "ee_link_1"
        )

        self.ee_link_2 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "ee_link_2"
        )

        self.ee_link_3 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "ee_link_3"
        )

    def _drive_pos(self, id=1):
        rotule_lever_5_p = [-0.00057801, 0.01520963, -0.00386368]
        rotule_lever_5 = self.robot.links_map["rotule_lever_5"]

        pose = rotule_lever_5.pose

        local_pose = sapien.Pose(rotule_lever_5_p)
        world_pose = pose * local_pose

        world_p = world_pose.p[0]
        print(world_p)
        return world_p

    def _set_fixed_drive(self, drive, stiff = 1e6, damp = 1):
        for d in drive._objs:
            # 1️⃣ 锁平移
            d.set_limit_x(0, 0)
            d.set_limit_y(0, 0)
            d.set_limit_z(0, 0)

            # 2️⃣ 锁旋转
            d.set_limit_twist(0, 0)
            d.set_limit_cone(0, 0)   # swing

            # 3️⃣ 加刚度（防漂）
            d.set_drive_property_x(stiff, damp)
            d.set_drive_property_y(stiff, damp)
            d.set_drive_property_z(stiff, damp)

            d.set_drive_property_swing(stiff, damp, 3.4028234663852886e38, "force")
            d.set_drive_property_twist(stiff, damp, 3.4028234663852886e38, "force")

            # 4️⃣ 锁当前相对位姿
            d.set_drive_target(sapien.Pose())

    def _set_drive(self, drive, type="ball", swing=3.14 / 2, twist=3.14 / 2, stiff=5, damp=2, force_limit=50):
        for d in drive._objs:
            d.set_limit_x(0, 0)   # 锁平移
            d.set_limit_y(0, 0)
            d.set_limit_z(0, 0)
            if type == 'ball':
                # d.set_limit_cone(0.3, 0.3)
                # d.set_limit_twist(-0.05, 0.05)
                d.set_drive_property_swing(stiff, damp, force_limit, "force")
                d.set_drive_property_twist(stiff, damp, force_limit, "force")
                pass
            # d.set_limit_twist(-twist, twist)
                    
        # drive.set_drive_property_x(1e6, 1e5)
        # drive.set_drive_property_y(1e6, 1e5)
        # drive.set_drive_property_z(1e6, 1e5)

    def _get_drive_pos(self, link, world_pos):
        # 获取 link 上 world_pos 的 local 坐标
        pose = link.pose
        world_pose = sapien.Pose(p=world_pos)
        local_pose = pose.inv() * world_pose
        return local_pose

    def _get_constraint_p(self):
        # target_name = "rotule_ball_15"
        # link_a_name = "link_3"
        # link_b_name = "rotule_lever_14"

        # target_name = "rotule_lever"
        # link_a_name = "rotule_lever_1"
        # link_b_name = "rotule_lever"

        target_name = "parallel_pin_2_x_16_7"
        link_a_name = "proximal_3"
        link_b_name = "distal_3"

        # ball_offset = sapien.Pose([0, 0, 0.0029])
        # target_world_p = (self.robot.links_map[target_name].pose * ball_offset).p[0]

        target_world_p = self.robot.links_map[target_name].pose.p[0]

        link_a = self.robot.links_map[link_a_name]
        link_b = self.robot.links_map[link_b_name]

        link_a_pose = self._get_drive_pos(link_a, target_world_p)
        link_b_pose = self._get_drive_pos(link_b, target_world_p)

        print(f"{target_name}  (target) real position:", target_world_p)
        print(f"target_p in {link_a_name} local position:", link_a_pose)
        print(f"target_p in {link_b_name} local position:", link_b_pose)

    def _after_loading_articulation(self):
        # for finger 1:
   
        link_1 = self.robot.links_map["link_1"]
        rotule_lever_5 = self.robot.links_map["rotule_lever_5"]
        link_1_pose = [-0.0099,  0.0036, -0.0051,  0.9998,  0.0190, -0.0000, -0.0000]
        rotule_lever_5_pose = [-8.9407e-08,  1.5250e-02,  1.1921e-07,  7.0692e-01, -2.1556e-02, -7.0658e-01,  2.3369e-02]

        self.finger1_drive1 = self.scene.create_drive(
            link_1, sapien.Pose(p=link_1_pose[:3], q=link_1_pose[3:]), rotule_lever_5, sapien.Pose(p=rotule_lever_5_pose[:3], q=rotule_lever_5_pose[3:])
        )
        self._set_drive(self.finger1_drive1)

        rotule_lever_7 = self.robot.links_map["rotule_lever_7"]
        link_1_pose = [ 0.0062,  0.0036, -0.0051,  0.9998,  0.0190, -0.0000, -0.0000]
        rotule_lever_7_pose = [-1.7881e-07,  1.5250e-02,  1.4901e-07,  7.0669e-01,  2.3370e-02, -7.0681e-01, -2.1564e-02]

        self.finger1_drive2 = self.scene.create_drive(
            link_1, sapien.Pose(p=link_1_pose[:3], q=link_1_pose[3:]), rotule_lever_7, sapien.Pose(p=rotule_lever_7_pose[:3], q=rotule_lever_7_pose[3:])
        )
        self._set_drive(self.finger1_drive2)

        proximal_1 = self.robot.links_map["proximal_1"]
        distal_1 = self.robot.links_map["distal_1"]
        proximal_1_p = [0.0029, 0.0520, 0.0000]
        distal_1_p = [0.0059,  0.0028, -0.0053]

        self.finger1_drive3 = self.scene.create_drive(
            proximal_1, sapien.Pose(proximal_1_p), distal_1, sapien.Pose(distal_1_p)
        )
        self._set_drive(self.finger1_drive3, type="pin")

    #     # for finger 2:

        link_2 = self.robot.links_map["link_2"]
        rotule_lever_9 = self.robot.links_map["rotule_lever_9"]
        link_2_pose = [0.0061,  0.0036, -0.0051,  0.9929,  0.0189, -0.0022,  0.1175]
        rotule_lever_9_pose = [-3.5763e-07,  1.5250e-02,  2.3842e-07,  7.4422e-01, -5.6168e-02, -6.6238e-01,  6.5035e-02]

        self.finger2_drive1 = self.scene.create_drive(
            link_2, sapien.Pose(p=link_2_pose[:3], q=link_2_pose[3:]), rotule_lever_9, sapien.Pose(p=rotule_lever_9_pose[:3], q=rotule_lever_9_pose[3:])
        )
        self._set_drive(self.finger2_drive1)

        rotule_lever_11 = self.robot.links_map["rotule_lever_11"]
        link_2_pose = [-0.0100,  0.0036, -0.0051,  0.9929,  0.0189, -0.0022,  0.1175]
        rotule_lever_11_pose = [-3.2783e-07,  1.5250e-02,  8.9407e-08,  6.9210e-01, -1.0553e-01, -7.0625e-01,  1.0523e-01]

        self.finger2_drive2 = self.scene.create_drive(
            link_2, sapien.Pose(p=link_2_pose[:3], q=link_2_pose[3:]), rotule_lever_11, sapien.Pose(p=rotule_lever_11_pose[:3], q=rotule_lever_11_pose[3:])
        )
        self._set_drive(self.finger2_drive2)

        proximal_2 = self.robot.links_map["proximal_2"]
        distal_2 = self.robot.links_map["distal_2"]
        proximal_2_p = [2.8500e-03,  5.2000e-02,  2.9802e-08]
        distal_2_p = [0.0059,  0.0028, -0.0053]

        self.finger2_drive3 = self.scene.create_drive(
            proximal_2, sapien.Pose(proximal_2_p), distal_2, sapien.Pose(distal_2_p)
        )
        self._set_drive(self.finger2_drive3, type="pin")

    #     # for finger 3:

        link = self.robot.links_map["link"]
        rotule_lever_2 = self.robot.links_map["rotule_lever_2"]
        link_pose = [6.1500e-03,  3.5957e-03, -5.0508e-03,  9.9845e-01,  1.8983e-02, -9.9489e-04,  5.2327e-02]
        rotule_lever_2_pose = [-3.5763e-07,  1.5250e-02,  1.6391e-07,  7.4687e-01, -1.2728e-02, -6.6465e-01,  1.6224e-02]

        self.finger3_drive1 = self.scene.create_drive(
            link, sapien.Pose(p=link_pose[:3], q=link_pose[3:]), rotule_lever_2, sapien.Pose(p=rotule_lever_2_pose[:3], q=rotule_lever_2_pose[3:])
        )
        self._set_drive(self.finger3_drive1)

        rotule_lever = self.robot.links_map["rotule_lever"]
        link_pose = [-9.9500e-03,  3.5958e-03, -5.0508e-03,  9.9845e-01,  1.8983e-02, -9.9489e-04,  5.2327e-02]
        rotule_lever_pose = [-2.9802e-07,  1.5250e-02,  1.4901e-08,  6.9817e-01, -5.9063e-02, -7.1098e-01,  5.9793e-02]

        self.finger3_drive2 = self.scene.create_drive(
            link, sapien.Pose(p=link_pose[:3], q=link_pose[3:]), rotule_lever, sapien.Pose(p=rotule_lever_pose[:3], q=rotule_lever_pose[3:])
        )
        self._set_drive(self.finger3_drive2)

        proximal = self.robot.links_map["proximal"]
        distal = self.robot.links_map["distal"]
        proximal_p = [2.8500e-03,  5.2000e-02,  2.9802e-08]
        distal_p = [0.0059,  0.0028, -0.0053]

        self.finger3_drive3 = self.scene.create_drive(
            proximal, sapien.Pose(proximal_p), distal, sapien.Pose(distal_p)
        )
        self._set_drive(self.finger3_drive3, type="pin")

    #     # for finger 4:

        link_3 = self.robot.links_map["link_3"]
        rotule_lever_12 = self.robot.links_map["rotule_lever_12"]
        link_3_pose = [0.0061,  0.0036, -0.0051,  0.1251, -0.1204,  0.7095, -0.6830]
        rotule_lever_12_pose = [6.7055e-08,  1.5250e-02, -2.9802e-08,  5.4900e-01,  6.5103e-01, -4.4707e-01,  2.7364e-01]

        self.finger4_drive1 = self.scene.create_drive(
            link_3, sapien.Pose(p=link_3_pose[:3], q=link_3_pose[3:]), rotule_lever_12, sapien.Pose(p=rotule_lever_12_pose[:3], q=rotule_lever_12_pose[3:])
        )
        self._set_drive(self.finger4_drive1)

        rotule_lever_14 = self.robot.links_map["rotule_lever_14"]
        link_3_pose = [-0.0100,  0.0036, -0.0051,  0.1251, -0.1204,  0.7095, -0.6830]
        rotule_lever_14_pose = [1.9744e-07,  1.5250e-02,  7.4506e-09,  4.8609e-01,  6.5006e-01, -5.1477e-01,  2.7594e-01]

        self.finger4_drive2 = self.scene.create_drive(
            link_3, sapien.Pose(p=link_3_pose[:3], q=link_3_pose[3:]), rotule_lever_14, sapien.Pose(p=rotule_lever_14_pose[:3], q=rotule_lever_14_pose[3:])
        )
        self._set_drive(self.finger4_drive2)

        proximal_3 = self.robot.links_map["proximal_3"]
        distal_3 = self.robot.links_map["distal_3"]
        proximal_3_p = [2.8500e-03, 5.2000e-02, -1.8626e-08]
        distal_3_p = [0.0059, 0.0028, -0.0053]

        self.finger4_drive3 = self.scene.create_drive(
            proximal_3, sapien.Pose(proximal_3_p), distal_3, sapien.Pose(distal_3_p)
        )
        self._set_drive(self.finger4_drive3, type="pin")

    def is_grasping(
        self,
        object_actor,        # target object (sapien.Actor)
        min_fingers=2,
        force_threshold=0.5, # contact force must be above this to count
        oppose_threshold=-0.3,
    ):
        """
        判断是否成功 grasp

        Returns:
            grasp (bool)
        """
        scene = self.scene
        hand_links = self.touch_links.values()  # 只关注 touch_links 定义的部分

        contact_forces = []
        active_links = []

        # =========================
        # 1. 获取每个 link 的接触力
        # =========================
        for link in hand_links:
            # 注意：返回的是作用在 link 上的力
            f = scene.get_pairwise_contact_forces(link, object_actor)

            f_norm = torch.linalg.norm(f)

            if f_norm > force_threshold:
                contact_forces.append(f)
                active_links.append(link)

            num_fingers = len(active_links)

            if num_fingers < min_fingers:
                return False

            # =========================
            # 2. 对向力检测（关键）
            # =========================
            opposing = False

            for i in range(len(contact_forces)):
                for j in range(i + 1, len(contact_forces)):
                    fi = contact_forces[i]
                    fj = contact_forces[j]

                    ni = fi / (torch.linalg.norm(fi) + 1e-8)
                    nj = fj / (torch.linalg.norm(fj) + 1e-8)

                    if torch.dot(ni, nj) < oppose_threshold:
                        opposing = True
                        break

                if opposing:
                    break

            if not opposing:
                return False
            
            # =========================
            # 成功 grasp
            # =========================
            return True
        
    def get_joint_qpos(self, name):
        for joint in self.controller.joints:
            if joint.name == name:
                return joint.qpos
        raise ValueError(f"Joint {name} not found")
    
        