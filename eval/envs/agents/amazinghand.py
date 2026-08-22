"""Adapted from https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/cartpole.py"""

import os
from typing import Any, Optional, Union
from pathlib import Path

from mani_skill.agents.controllers.base_controller import ControllerConfig
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs import Articulation
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
import torch.nn as nn
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from math import sqrt

@register_agent()
class AH_RIGHT(BaseAgent):
    uid = "amazinghand_right"
    urdf_path = Path(__file__).parent.resolve() / f'urdfs/amazinghand_right/right_hand_final_simple.urdf'
    disable_self_collisions = False
    load_multiple_collisions = False

    all_active_joints = ['revolute_1_1', 'revolute_6_1', 'revolute_5_1', 'revolute_1_4', 'revolute_6_3', 'revolute_5_3', 'revolute_1_0', 'revolute_6_0', 'revolute_5_0', 'revolute_2_1', 'revolute_3_2', 'revolute_1_2', 'revolute_6_2', 'revolute_5_2', 'revolute_2_3', 'revolute_3_4', 'revolute_2_0', 'revolute_3_1', 'revolute_2_2', 'revolute_3_3', 'cylindrical_1_5', 'cylindrical_1_11', 'cylindrical_1_2', 'cylindrical_1_8']

    servo_joints = ["revolute_5_0", "revolute_5_1", "revolute_5_2", "revolute_5_3", "revolute_6_0", "revolute_6_1", "revolute_6_2", "revolute_6_3"]
    joint_names_1 = ["revolute_5_1", "revolute_6_1", "revolute_3_2", "revolute_1_1", "cylindrical_1_5", "revolute_2_1"]
    joint_names_2 = ["revolute_5_3", "revolute_6_3", "revolute_3_4", "revolute_1_4", "cylindrical_1_11", "revolute_2_3"]
    joint_names_3 = ["revolute_5_0", "revolute_6_0", "revolute_3_1", "revolute_1_0", "cylindrical_1_2", "revolute_2_0"]
    joint_names_4 = ["revolute_5_2", "revolute_6_2", "revolute_3_3", "revolute_1_2", "cylindrical_1_8", "revolute_2_2"]

    finger_shell_1 = ["distal_shell_1", "proximal_shell_1"]
    finger_shell_2 = ["distal_shell", "proximal_shell"]
    finger_shell_3 = ["distal_shell_2", "proximal_shell_2"]
    finger_shell_4 = ["distal_shell_3", "proximal_shell_3"]

    servo_stiffness = 200
    servo_damping = 2 * sqrt(servo_stiffness)
    servo_friction = 0
    force_limit = 0.01

    keyframes = dict(
        start=Keyframe(
            qpos=np.array(
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ),
            pose=sapien.Pose(),
        )
    )

    urdf_config = dict(
        _materials=dict(
            skin=dict(static_friction=1., dynamic_friction=1., restitution=0.0)
        ),
        link={
            l : dict(
                material="skin", patch_radius=0.1, min_patch_radius=0.1
            )
            for l in ["distal_shell", "proximal_shell", "distal_shell_1", "proximal_shell_1", "distal_shell_2", "proximal_shell_2", "distal_shell_3", "proximal_shell_3", "r_palm_shell"]
        }
    )

    def __init__(
        self,
        scene,
        control_freq: int,
        control_mode: Optional[str] = None,
        agent_idx: Optional[str] = None,
        initial_pose: Optional[Union[sapien.Pose, Pose]] = None,
        build_separate: bool = False,
    ):
        self.touch_links = {}
        super().__init__(scene, control_freq, control_mode, agent_idx, initial_pose, build_separate)
        
    @property
    def _controller_configs(self):
        def get_finger_controllers(joint_names, finger_idx, finger_shells):
            joint_map = {name : joint for name, joint in zip(names, joint_names)}
            finger_pd_joint_delta_pos = PDJointPosControllerForAHConfig(
                joint_names,
                lower=-0.4,
                upper=0.4,
                damping=self.servo_damping,
                stiffness=self.servo_stiffness,
                friction=self.servo_friction,
                force_limit=self.force_limit,
                use_delta=True,
                finger_id=finger_idx,
                finger_shells=finger_shells,
                # interpolate=True,
                **joint_map
            )
            finger_pd_joint_pos = PDJointPosControllerForAHConfig(
                joint_names,
                lower=None, 
                upper=None,
                damping=self.servo_damping,
                stiffness=self.servo_stiffness,
                friction=self.servo_friction,
                finger_id=finger_idx,
                force_limit=self.force_limit,
                finger_shells=finger_shells,
                # interpolate=True,
                **joint_map
            )
            return finger_pd_joint_delta_pos, finger_pd_joint_pos

        # names = ['A', 'B', 'roll', 'pitch', 'distal', 'proximal']
        names = ['A', 'B', 'roll', 'pitch']
        # finger1:
        finger_id = 1
        finger1_pd_joint_delta_pos, finger1_pd_joint_pos = get_finger_controllers(self.joint_names_1[:4], finger_id, finger_shells=self.finger_shell_1)
        # finger2:
        finger_id = 2
        finger2_pd_joint_delta_pos, finger2_pd_joint_pos = get_finger_controllers(self.joint_names_2[:4], finger_id, finger_shells=self.finger_shell_2)
        # finger3:
        finger_id = 3
        finger3_pd_joint_delta_pos, finger3_pd_joint_pos = get_finger_controllers(self.joint_names_3[:4], finger_id, finger_shells=self.finger_shell_3)
        # finger4:
        finger_id = 4
        finger4_pd_joint_delta_pos, finger4_pd_joint_pos = get_finger_controllers(self.joint_names_4[:4], finger_id, finger_shells=self.finger_shell_4)

        joint_names = self.joint_names_1[:4] + self.joint_names_2[:4] + self.joint_names_3[:4] + self.joint_names_4[:4]
        passive_joints = [x for x in self.all_active_joints if x not in joint_names]
        rest = PassiveControllerConfig(passive_joints, damping=0, friction=self.servo_friction, force_limit=self.force_limit)

        return dict(
            pd_joint_delta_pos=dict(
                finger1=finger1_pd_joint_delta_pos, 
                finger2=finger2_pd_joint_delta_pos, 
                finger3=finger3_pd_joint_delta_pos, 
                finger4=finger4_pd_joint_delta_pos, 
                rest=rest, 
                balance_passive_force=False
            ),
            pd_joint_pos=dict(
                finger1=finger1_pd_joint_pos, 
                finger2=finger2_pd_joint_pos, 
                finger3=finger3_pd_joint_pos, 
                finger4=finger4_pd_joint_pos, 
                rest=rest, 
                balance_passive_force=False
            )
        )

    def get_proprioception(self):
        qpos = torch.stack([joint.qpos for joint in self.controller.joints if joint.name in self.servo_joints], dim=-1)
        qvel = torch.stack([joint.qvel for joint in self.controller.joints if joint.name in self.servo_joints], dim=-1)
        roll = torch.stack([
            self.robot.joints_map[self.joint_names_1[2]].qpos,
            self.robot.joints_map[self.joint_names_2[2]].qpos,
            self.robot.joints_map[self.joint_names_3[2]].qpos,
            self.robot.joints_map[self.joint_names_4[2]].qpos,
        ], dim=-1)
        pitch = torch.stack([
            self.robot.joints_map[self.joint_names_1[3]].qpos,
            self.robot.joints_map[self.joint_names_2[3]].qpos,
            self.robot.joints_map[self.joint_names_3[3]].qpos,
            self.robot.joints_map[self.joint_names_4[3]].qpos,
        ], dim=-1)
        # return dict(qpos=qpos, qvel=qvel, roll=roll, pitch=pitch)
        # if not torch.isfinite(roll).all():
        #     roll = torch.ones_like(roll, device=self.device) * -1.4
        # if not torch.isfinite(pitch).all():
        #     pitch = torch.ones_like(pitch, device=self.device) * 0.7
        roll[torch.isnan(roll)] = -1.4
        pitch[torch.isnan(pitch)] = 0.8
        roll = torch.clamp(roll, -1.4, 0)
        pitch = torch.clamp(pitch, -0.8, 0.8)
        return dict(roll=roll, pitch=pitch)

    def _after_init(self):
        self.touch_links_names = ["distal_shell", "proximal_shell", "distal_shell_1", "proximal_shell_1", "distal_shell_2", "proximal_shell_2", "distal_shell_3", "proximal_shell_3", "r_palm_shell"]
        
        for link_name in self.touch_links_names:
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

    def _set_drive(self, drive, type="ball", swing=3.14 / 2, twist=3.14 / 2, stiff=30, damp=5, force_limit=50):
        for d in drive._objs:
            d.set_limit_x(0, 0)   # 锁平移
            d.set_limit_y(0, 0)
            d.set_limit_z(0, 0)
            if type == 'ball':
                # d.set_limit_cone(0.3, 0.3)
                # d.set_limit_twist(-0.05, 0.05)
                # d.set_drive_property_swing(stiff, damp, force_limit, "force")
                # d.set_drive_property_twist(stiff, damp, force_limit, "force")
                pass
            # d.set_limit_twist(-twist, twist)
                    
        # drive.set_drive_property_x(1e6, 0)
        # drive.set_drive_property_y(1e6, 0)
        # drive.set_drive_property_z(1e6, 0)

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
   
        # link_1 = self.robot.links_map["link_1"]
        # rotule_lever_5 = self.robot.links_map["rotule_lever_5"]
        # link_1_pose = [-0.0099,  0.0036, -0.0051,  0.9998,  0.0190, -0.0000, -0.0000]
        # rotule_lever_5_pose = [-8.9407e-08,  1.5250e-02,  1.1921e-07,  7.0692e-01, -2.1556e-02, -7.0658e-01,  2.3369e-02]

        # self.finger1_drive1 = self.scene.create_drive(
        #     link_1, sapien.Pose(p=link_1_pose[:3], q=link_1_pose[3:]), rotule_lever_5, sapien.Pose(p=rotule_lever_5_pose[:3], q=rotule_lever_5_pose[3:])
        # )
        # self._set_drive(self.finger1_drive1)

        # rotule_lever_7 = self.robot.links_map["rotule_lever_7"]
        # link_1_pose = [ 0.0062,  0.0036, -0.0051,  0.9998,  0.0190, -0.0000, -0.0000]
        # rotule_lever_7_pose = [-1.7881e-07,  1.5250e-02,  1.4901e-07,  7.0669e-01,  2.3370e-02, -7.0681e-01, -2.1564e-02]

        # self.finger1_drive2 = self.scene.create_drive(
        #     link_1, sapien.Pose(p=link_1_pose[:3], q=link_1_pose[3:]), rotule_lever_7, sapien.Pose(p=rotule_lever_7_pose[:3], q=rotule_lever_7_pose[3:])
        # )
        # self._set_drive(self.finger1_drive2)

        proximal_1 = self.robot.links_map["proximal_1"]
        distal_1 = self.robot.links_map["distal_1"]
        proximal_1_p = [0.0029, 0.0520, 0.0000]
        distal_1_p = [0.0059,  0.0028, -0.0053]

        self.finger1_drive3 = self.scene.create_drive(
            proximal_1, sapien.Pose(proximal_1_p), distal_1, sapien.Pose(distal_1_p)
        )
        self._set_drive(self.finger1_drive3, type="pin")

    #     # for finger 2:

        # link_2 = self.robot.links_map["link_2"]
        # rotule_lever_9 = self.robot.links_map["rotule_lever_9"]
        # link_2_pose = [0.0061,  0.0036, -0.0051,  0.9929,  0.0189, -0.0022,  0.1175]
        # rotule_lever_9_pose = [-3.5763e-07,  1.5250e-02,  2.3842e-07,  7.4422e-01, -5.6168e-02, -6.6238e-01,  6.5035e-02]

        # self.finger2_drive1 = self.scene.create_drive(
        #     link_2, sapien.Pose(p=link_2_pose[:3], q=link_2_pose[3:]), rotule_lever_9, sapien.Pose(p=rotule_lever_9_pose[:3], q=rotule_lever_9_pose[3:])
        # )
        # self._set_drive(self.finger2_drive1)

        # rotule_lever_11 = self.robot.links_map["rotule_lever_11"]
        # link_2_pose = [-0.0100,  0.0036, -0.0051,  0.9929,  0.0189, -0.0022,  0.1175]
        # rotule_lever_11_pose = [-3.2783e-07,  1.5250e-02,  8.9407e-08,  6.9210e-01, -1.0553e-01, -7.0625e-01,  1.0523e-01]

        # self.finger2_drive2 = self.scene.create_drive(
        #     link_2, sapien.Pose(p=link_2_pose[:3], q=link_2_pose[3:]), rotule_lever_11, sapien.Pose(p=rotule_lever_11_pose[:3], q=rotule_lever_11_pose[3:])
        # )
        # self._set_drive(self.finger2_drive2)

        proximal_2 = self.robot.links_map["proximal_2"]
        distal_2 = self.robot.links_map["distal_2"]
        proximal_2_p = [2.8500e-03,  5.2000e-02,  2.9802e-08]
        distal_2_p = [0.0059,  0.0028, -0.0053]

        self.finger2_drive3 = self.scene.create_drive(
            proximal_2, sapien.Pose(proximal_2_p), distal_2, sapien.Pose(distal_2_p)
        )
        self._set_drive(self.finger2_drive3, type="pin")

    #     # for finger 3:

        # link = self.robot.links_map["link"]
        # rotule_lever_2 = self.robot.links_map["rotule_lever_2"]
        # link_pose = [6.1500e-03,  3.5957e-03, -5.0508e-03,  9.9845e-01,  1.8983e-02, -9.9489e-04,  5.2327e-02]
        # rotule_lever_2_pose = [-3.5763e-07,  1.5250e-02,  1.6391e-07,  7.4687e-01, -1.2728e-02, -6.6465e-01,  1.6224e-02]

        # self.finger3_drive1 = self.scene.create_drive(
        #     link, sapien.Pose(p=link_pose[:3], q=link_pose[3:]), rotule_lever_2, sapien.Pose(p=rotule_lever_2_pose[:3], q=rotule_lever_2_pose[3:])
        # )
        # self._set_drive(self.finger3_drive1)

        # rotule_lever = self.robot.links_map["rotule_lever"]
        # link_pose = [-9.9500e-03,  3.5958e-03, -5.0508e-03,  9.9845e-01,  1.8983e-02, -9.9489e-04,  5.2327e-02]
        # rotule_lever_pose = [-2.9802e-07,  1.5250e-02,  1.4901e-08,  6.9817e-01, -5.9063e-02, -7.1098e-01,  5.9793e-02]

        # self.finger3_drive2 = self.scene.create_drive(
        #     link, sapien.Pose(p=link_pose[:3], q=link_pose[3:]), rotule_lever, sapien.Pose(p=rotule_lever_pose[:3], q=rotule_lever_pose[3:])
        # )
        # self._set_drive(self.finger3_drive2)

        proximal = self.robot.links_map["proximal"]
        distal = self.robot.links_map["distal"]
        proximal_p = [2.8500e-03,  5.2000e-02,  2.9802e-08]
        distal_p = [0.0059,  0.0028, -0.0053]

        self.finger3_drive3 = self.scene.create_drive(
            proximal, sapien.Pose(proximal_p), distal, sapien.Pose(distal_p)
        )
        self._set_drive(self.finger3_drive3, type="pin")

    #     # for finger 4:

        # link_3 = self.robot.links_map["link_3"]
        # rotule_lever_12 = self.robot.links_map["rotule_lever_12"]
        # link_3_pose = [0.0061,  0.0036, -0.0051,  0.1251, -0.1204,  0.7095, -0.6830]
        # rotule_lever_12_pose = [6.7055e-08,  1.5250e-02, -2.9802e-08,  5.4900e-01,  6.5103e-01, -4.4707e-01,  2.7364e-01]

        # self.finger4_drive1 = self.scene.create_drive(
        #     link_3, sapien.Pose(p=link_3_pose[:3], q=link_3_pose[3:]), rotule_lever_12, sapien.Pose(p=rotule_lever_12_pose[:3], q=rotule_lever_12_pose[3:])
        # )
        # self._set_drive(self.finger4_drive1)

        # rotule_lever_14 = self.robot.links_map["rotule_lever_14"]
        # link_3_pose = [-0.0100,  0.0036, -0.0051,  0.1251, -0.1204,  0.7095, -0.6830]
        # rotule_lever_14_pose = [1.9744e-07,  1.5250e-02,  7.4506e-09,  4.8609e-01,  6.5006e-01, -5.1477e-01,  2.7594e-01]

        # self.finger4_drive2 = self.scene.create_drive(
        #     link_3, sapien.Pose(p=link_3_pose[:3], q=link_3_pose[3:]), rotule_lever_14, sapien.Pose(p=rotule_lever_14_pose[:3], q=rotule_lever_14_pose[3:])
        # )
        # self._set_drive(self.finger4_drive2)

        proximal_3 = self.robot.links_map["proximal_3"]
        distal_3 = self.robot.links_map["distal_3"]
        proximal_3_p = [2.8500e-03, 5.2000e-02, -1.8626e-08]
        distal_3_p = [0.0059, 0.0028, -0.0053]

        self.finger4_drive3 = self.scene.create_drive(
            proximal_3, sapien.Pose(proximal_3_p), distal_3, sapien.Pose(distal_3_p)
        )
        self._set_drive(self.finger4_drive3, type="pin")

        finger1_links = ['distal_shell', 'proximal_shell']
        finger2_links = ['distal_shell_1', 'proximal_shell_1']
        finger3_links = ['distal_shell_2', 'proximal_shell_2']
        finger4_links = ['distal_shell_3', 'proximal_shell_3']

        for link_name in finger1_links:
            link = self.robot.links_map[link_name]
            link.set_collision_group_bit(group=2, bit_idx=28, bit=1)

        for link_name in finger2_links:
            link = self.robot.links_map[link_name]
            link.set_collision_group_bit(group=2, bit_idx=29, bit=1)
        
        for link_name in finger3_links:
            link = self.robot.links_map[link_name]
            link.set_collision_group_bit(group=2, bit_idx=30, bit=1)

        for link_name in finger4_links:
            link = self.robot.links_map[link_name]
            link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def is_grasping(self, object_actor, min_contacts=2, force_thresh=0.0, vel_thresh=0.05, ret_num=False):
        # print(self.robot.qpos)
        scene = self.scene
        hand_links = list(self.touch_links.values())

        contact_forces = []
        thumb_touch = torch.zeros((self.scene.num_envs), dtype=torch.bool, device=self.device)
        finger1_touch = torch.zeros((self.scene.num_envs), dtype=torch.bool, device=self.device)
        finger2_touch = torch.zeros((self.scene.num_envs), dtype=torch.bool, device=self.device)
        finger3_touch = torch.zeros((self.scene.num_envs), dtype=torch.bool, device=self.device)
        palm_touch = torch.zeros((self.scene.num_envs), dtype=torch.bool, device=self.device)

        for id, link in enumerate(hand_links):
            f = scene.get_pairwise_contact_forces(link, object_actor)  # (B,3)
            if self.touch_links_names[id] in ['distal_shell_3', 'proximal_shell_3']:
                thumb_touch = thumb_touch | (torch.linalg.norm(f, dim=-1) > force_thresh)
            elif self.touch_links_names[id] in ['distal_shell', 'proximal_shell']:
                finger1_touch = finger1_touch | (torch.linalg.norm(f, dim=-1) > force_thresh)
            elif self.touch_links_names[id] in ['distal_shell_1', 'proximal_shell_1']:
                finger2_touch = finger2_touch | (torch.linalg.norm(f, dim=-1) > force_thresh)
            elif self.touch_links_names[id] in ['distal_shell_2', 'proximal_shell_2']:
                finger3_touch = finger3_touch | (torch.linalg.norm(f, dim=-1) > force_thresh)
            elif self.touch_links_names[id] in ['r_palm_shell']:
                palm_touch = palm_touch | (torch.linalg.norm(f, dim=-1) > force_thresh)

            contact_forces.append(f)

        contact_forces = torch.stack(contact_forces, dim=0)  # (F,B,3)

        force_norm = torch.linalg.norm(contact_forces, dim=-1)  # (F,B)

        contact_mask = force_norm > force_thresh

        num_contacts = contact_mask.sum(dim=0)  # (B,)
            
        grasp = (num_contacts >= min_contacts) & thumb_touch & (finger1_touch | finger2_touch | finger3_touch | palm_touch)

        # print(num_contacts)
        # print(thumb_touch, finger1_touch, finger2_touch, finger3_touch, grasp)
        
        if ret_num:
            return grasp, num_contacts
        
        return grasp
    
    def is_touching(self, object, min_force=0.):
        scene = self.scene
        hand_links = list(self.touch_links.values())
        touch_flag = torch.zeros((self.scene.num_envs), dtype=torch.bool, device=self.device)
        for id, link in enumerate(hand_links):
            f = scene.get_pairwise_contact_forces(link, object).norm(dim=-1) > min_force
            touch_flag = touch_flag | f
        return touch_flag


    def _touch_other_fingers(self):
        def is_finger_touching(finger_links, other_finger_links, thres=0.):
            for link in finger_links:
                for other_link in other_finger_links:
                    f = self.scene.get_pairwise_contact_forces(link, other_link)
                    contact = torch.linalg.norm(f, dim=-1) > thres
                    if contact.any():
                        return contact
            return torch.zeros((self.scene.num_envs,), dtype=torch.bool, device=self.device)
        
        finger1_shell_links = [self.robot.links_map[name] for name in self.finger_shell_1]
        finger2_shell_links = [self.robot.links_map[name] for name in self.finger_shell_2]
        finger3_shell_links = [self.robot.links_map[name] for name in self.finger_shell_3]
        finger4_shell_links = [self.robot.links_map[name] for name in self.finger_shell_4]

        contact1 = is_finger_touching(finger1_shell_links, finger2_shell_links)
        contact2 = is_finger_touching(finger2_shell_links, finger3_shell_links)
        contact3 = is_finger_touching(finger3_shell_links, finger4_shell_links)
        contact4 = is_finger_touching(finger4_shell_links, finger1_shell_links)
        contact5 = is_finger_touching(finger1_shell_links, finger3_shell_links)
        contact6 = is_finger_touching(finger2_shell_links, finger4_shell_links)
        return contact1 | contact2 | contact3 | contact4 | contact5 | contact6
        
    def is_static(self, thres=0.2):
        qvel = self.robot.get_qvel()
        return torch.max(torch.abs(qvel), 1)[0] <= thres

        
class F_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def encode(self, x):
        return torch.cat([
            x,
            torch.sin(x),
            torch.cos(x),
            torch.sin(2*x),
            torch.cos(2*x)
        ], dim=-1)

    def forward(self, x):
        x = self.encode(x)
        return self.net(x)
    
class G_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),   # ⭐ 改这里
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

from dataclasses import dataclass, field
import glob

class PDJointPosControllerForAH(PDJointPosController):
    config: "PDJointPosControllerForAHConfig"
    def __init__(self, config: ControllerConfig, articulation: Articulation, control_freq: int, sim_freq: int | None = None, scene: ManiSkillScene = None):
        super().__init__(config, articulation, control_freq, sim_freq, scene)
        self.finger_id = self.config.finger_id
        f_path = str(Path(__file__).parent.resolve() / f"models/amazinghand_right/finger{self.finger_id}_f.pth")
        g_path = str(Path(__file__).parent.resolve() / f"models/amazinghand_right/finger{self.finger_id}_g.pth")
        self.f = F_Model()
        self.g = G_Model()
        self.f.load_state_dict(torch.load(f_path, map_location="cpu"))
        self.g.load_state_dict(torch.load(g_path, map_location="cpu"))
        self.f = self.f.to(self.device)
        self.g = self.g.to(self.device)
        self.f.eval()
        self.g.eval()

        self.shell_links = [sapien_utils.get_obj_by_name(self.articulation.get_links(), name) for name in self.config.finger_shells]
        # self._desired_qpos = self._target_qpos.clone()

    def reset(self):

        super().reset()

        num_envs = self.scene.num_envs
        device = self.device

        # =====================================================
        # per-link contact force history
        # =====================================================
        self._prev_link_contact_force = {}

        for i, shell in enumerate(self.shell_links):

            self._prev_link_contact_force[i] = torch.zeros(
                (num_envs,),
                dtype=torch.float32,
                device=device
            )

        # =====================================================
        # desired target
        # =====================================================
        self._prev_qpos = self._target_qpos.clone()
        
    def _initialize_joints(self):
        super()._initialize_joints()
        A_idx = self.config.joint_names.index(self.config.A)
        B_idx = self.config.joint_names.index(self.config.B)
        roll_idx = self.config.joint_names.index(self.config.roll)
        pitch_idx = self.config.joint_names.index(self.config.pitch)
        # distal_idx = self.config.joint_names.index(self.config.distal)
        # proximal_idx = self.config.joint_names.index(self.config.proximal)
        self._control_joint_indices = torch.tensor([A_idx, B_idx], device=self.device, dtype=torch.int32)
        # self._passive_joint_indices = torch.tensor([roll_idx, pitch_idx, distal_idx, proximal_idx], device=self.device, dtype=torch.int32)
        self._passive_joint_indices = torch.tensor([roll_idx, pitch_idx], device=self.device, dtype=torch.int32)
        self._all_joint_indices = torch.tensor([A_idx, B_idx, roll_idx, pitch_idx], device=self.device, dtype=torch.int32)

    def _get_joint_limits(self):
        joint_limits = super()._get_joint_limits()
        # 只对 A、B 两个关节进行限制
        joint_limits = joint_limits[self._control_joint_indices.cpu().numpy()]
        if len(joint_limits.shape) == 1:
            joint_limits = joint_limits[None, :]
        return joint_limits

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._prev_qpos = self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos[:, self._control_joint_indices] = self._target_qpos[:, self._control_joint_indices] + action
            else:
                self._target_qpos[:, self._control_joint_indices] = self._start_qpos[:, self._control_joint_indices] + action
        else:
            # Compatible with mimic controllers. Need to clone here otherwise cannot do in-place replacements in the reset function
            self._target_qpos[:, self._control_joint_indices] = torch.broadcast_to(
                action, self._start_qpos[:, self._control_joint_indices].shape
            ).clone()

        f_input = self._target_qpos[:, self._control_joint_indices]

        with torch.no_grad():
            f_output = self.f(f_input)

        f_output[:, 0] = torch.clamp(f_output[:, 0], -1.3, 0)
        f_output[:, 1] = torch.clamp(f_output[:, 1], -0.6, 0.6)
        self._target_qpos[:, self._passive_joint_indices] = f_output

        # self._desired_qpos = self._target_qpos.clone()

        # self._target_qpos = _quantize_qpos(self._target_qpos, only_arm=True)
        
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def is_contact_force_reach_thres_and_increase(
        self,
        thres=0,
        increase_thres=0.000,
    ):

        num_envs = self.scene.num_envs
        device = self.device

        # =====================================================
        # 初始化 prev_force
        # =====================================================
        if not hasattr(self, "_prev_link_contact_force"):

            self._prev_link_contact_force = {}

            for i, shell in enumerate(self.shell_links):

                self._prev_link_contact_force[i] = torch.zeros(
                    (num_envs,),
                    dtype=torch.float32,
                    device=device
                )

        # =====================================================
        # 最终 flag
        # =====================================================
        final_flag = torch.zeros(
            (num_envs,),
            dtype=torch.bool,
            device=device
        )

        # =====================================================
        # 每个 shell 单独判断
        # =====================================================
        for i, shell in enumerate(self.shell_links):

            current_force = torch.linalg.norm(
                shell.get_net_contact_forces(),
                dim=-1
            )

            prev_force = self._prev_link_contact_force[i]

            # force increase
            force_delta = current_force - prev_force

            increase_flag = force_delta > increase_thres

            # over threshold
            thres_flag = current_force > thres

            # 当前 shell 是否危险
            shell_flag = thres_flag & increase_flag

            # 任意 shell 危险即可
            final_flag |= shell_flag

            # update prev force
            self._prev_link_contact_force[i] = current_force.clone()

        return final_flag        

    def before_simulation_step(self):
        self._step += 1
        
        if not self.config.interpolate:
            # =====================================================
            # 1. 当前关节位置
            # =====================================================
            current = self.qpos

            self.set_drive_targets(current)

            # print(current, desired)

            # =====================================================
            # 3. 接触检测
            # =====================================================
            contact = self.is_contact_force_reach_thres_and_increase()

            # =====================================================
            # 5. 接触后减速（核心）
            # =====================================================
            if contact.any():
                # 接触后只允许小幅移动
                # print("contact detected, reducing step size", torch.where(contact)[0])
                
                self._target_qpos = self._prev_qpos

                self.set_drive_targets(self._target_qpos)

            else:
                self._prev_qpos = current.clone()

        # Compute the next target via a linear interpolation
        if self.config.interpolate:
            targets = self._start_qpos + self._step_size * self._step
            self.set_drive_targets(targets)

@dataclass
class PDJointPosControllerForAHConfig(PDJointPosControllerConfig):
    controller_cls = PDJointPosControllerForAH
    finger_id:int = 1 # 1~4
    A:str = ''
    B:str = ''
    roll:str = ''
    pitch:str = ''
    finger_shells:list = field(default_factory=list)
    # distal:str = ''
    # proximal:str = ''