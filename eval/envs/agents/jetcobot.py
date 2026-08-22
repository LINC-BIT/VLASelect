from copy import deepcopy

import numpy as np
import sapien
import sapien.physx as physx
import torch
from pathlib import Path

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class JetCobot(BaseAgent):
    uid = "jetcobot"
    urdf_path = Path(__file__).parent.resolve() / f'urdfs/jetcobot/jetcobot_display_no_gripper_mass.urdf'
    # disable_self_collisions = True
    arm_joint_names = [
        '1_Joint', 
        '2_Joint', 
        '3_Joint', 
        '4_Joint', 
        '5_Joint', 
        '6_Joint',
    ]
    
    gripper_joint_names = [
        'gripper_controller',
        # 'gripper_base_to_gripper_left2', 
        'gripper_base_to_gripper_right3', 
        # 'gripper_base_to_gripper_right2', 
        'gripper_left3_to_gripper_left1', 
        'gripper_right3_to_gripper_right1'
    ]

    ee_link_name = "gripper_tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0.009, 0, 0.01], q=[1, 0, 0, 0]),  # 相对于 camera_link 的本地坐标
                width=640,
                height=480,
                fov=np.deg2rad(110),
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],      # 直接挂在 URDF 的 camera_link
            )
        ]

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-2.87,
            upper=2.87,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-2.87,
            upper=2.87,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True
        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-2.87,
            pos_upper=2.87,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-2.87,
            pos_upper=2.87,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            drive_mode="force",
            joint_names=self.gripper_joint_names,
            # lower = [-0.7, -0.8, -0.15, -0.5, -0.5, -0.5],
            # upper = [0.15, 0.5, 0.7, 0.8, 0.5, 0.5],

            lower = [-0.7, -0.15, -0.5, -0.5],
            upper = [0.15,  0.7, 0.5, 0.5],

            # lower = [-0.7, -0.15, -0.5, -0.5],
            # upper = [0.15,  0.7, 0.5, 0.5],

            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={
                # "gripper_base_to_gripper_left2": {
                #     "joint": "gripper_controller",
                #     "multiplier": 1.0,
                #     "offset": 0.0,
                # },
                "gripper_left3_to_gripper_left1": {
                    "joint": "gripper_controller",
                    "multiplier": -1.,
                    "offset": 0.0,
                },
                "gripper_base_to_gripper_right3": {
                    "joint": "gripper_controller",
                    "multiplier": -1.,
                    "offset": 0.0,
                },
                # "gripper_base_to_gripper_right2": {
                #     "joint": "gripper_controller",
                #     "multiplier": -1.0,
                #     "offset": 0.0,
                # },
                "gripper_right3_to_gripper_right1": {
                    "joint": "gripper_controller",
                    "multiplier": 1.,
                    "offset": 0.0,
                },
            },
            # mimic={
            #     # "gripper_base_to_gripper_left2": {
            #     #     "joint": "gripper_controller",
            #     #     "multiplier": 1.0,
            #     #     "offset": 0.0,
            #     # },
            #     "gripper_left3_to_gripper_left1": {
            #         "joint": "gripper_base_to_gripper_right3",
            #         "multiplier": 1.,
            #         "offset": 0.0,
            #     },
            #     "gripper_controller": {
            #         "joint": "gripper_base_to_gripper_right3",
            #         "multiplier": -1.,
            #         "offset": 0.0,
            #     },
            #     # "gripper_base_to_gripper_right2": {
            #     #     "joint": "gripper_controller",
            #     #     "multiplier": -1.0,
            #     #     "offset": 0.0,
            #     # },
            #     "gripper_right3_to_gripper_right1": {
            #         "joint": "gripper_base_to_gripper_right3",
            #         "multiplier": -1.,
            #         "offset": 0.0,
            #     },
            # },
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
        )

        # controller_configs = dict(
        #     pd_joint_delta_pos=dict(
        #         arm=arm_pd_joint_delta_pos
        #     ),
        #     pd_joint_pos=dict(arm=arm_pd_joint_pos),
        #     pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos),
        #     pd_ee_delta_pose=dict(
        #         arm=arm_pd_ee_delta_pose
        #     ),
        #     pd_ee_target_delta_pos=dict(
        #         arm=arm_pd_ee_target_delta_pos
        #     ),
        #     pd_ee_target_delta_pose=dict(
        #         arm=arm_pd_ee_target_delta_pose
        #     ),
        #     pd_joint_target_delta_pos=dict(
        #         arm=arm_pd_joint_target_delta_pos
        #     ),
        # )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    
    def _after_init(self):
        # 最末端接触物体的 finger links
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "gripper_left1"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "gripper_right1"
        )
        
        gripper_base = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "gripper_base"
        )
        gripper_base.set_linear_damping

        # TCP（你之前已经定义 gripper_tcp）
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )
        

    def is_grasping(self, object: Actor, min_force=0.2, min_opposition=0.000, max_opening=0.07):
        l_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )

        # 1️⃣ 必须左右同时接触
        if l_forces.shape[0] == 0 or r_forces.shape[0] == 0:
            return torch.zeros(
                max(l_forces.shape[0], r_forces.shape[0]),
                dtype=torch.bool,
                device=l_forces.device,
            )

        # 2️⃣ 力大小
        l_mag = torch.linalg.norm(l_forces, dim=1)
        r_mag = torch.linalg.norm(r_forces, dim=1)

        l_ok = l_mag >= min_force
        r_ok = r_mag >= min_force

        # 3️⃣ 对向力判据（核心）
        grasp_axis = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]

        l_proj = torch.sum(l_forces * grasp_axis, dim=1)
        r_proj = torch.sum(r_forces * grasp_axis, dim=1)

        opposition_ok = torch.logical_and(l_proj > min_opposition, r_proj < -min_opposition)

        # 4️⃣ 夹爪接近闭合
        opening = torch.abs(
            self.finger1_link.pose.p[..., 1]
            - self.finger2_link.pose.p[..., 1]
        )
        closed_ok = opening <= max_opening
        # print(f'closed:{closed_ok}')
        # 5️⃣ 合并所有条件
        grasp_ok = torch.logical_and(
            torch.logical_and(l_ok, r_ok),
            torch.logical_and(opposition_ok, closed_ok)
        )

        return grasp_ok

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-len(self.gripper_joint_names)]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)
    
    
@register_agent()
class JetCobotFixed(BaseAgent):
    uid = "jetcobot_fixed"
    urdf_path = Path(__file__).parent.resolve() / f'urdfs/jetcobot/jetcobot.urdf'

    arm_joint_names = [
        '1_Joint', 
        '2_Joint', 
        '3_Joint', 
        '4_Joint', 
        '5_Joint', 
        '6_Joint',
    ]

    gripper_joint_names = [
    ]

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),  # 相对于 camera_link 的本地坐标
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],      # 直接挂在 URDF 的 camera_link
            )
        ]

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True
        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=-0.01,   # 保证夹薄物体时仍有夹紧力
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={
                "gripper_base_to_gripper_left2": {
                    "joint": "gripper_controller",
                    "multiplier": 1.0,
                    "offset": 0.0,
                },
                "gripper_left3_to_gripper_left1": {
                    "joint": "gripper_controller",
                    "multiplier": -1.0,
                    "offset": 0.0,
                },
                "gripper_base_to_gripper_right3": {
                    "joint": "gripper_controller",
                    "multiplier": -1.0,
                    "offset": 0.0,
                },
                "gripper_base_to_gripper_right2": {
                    "joint": "gripper_controller",
                    "multiplier": -1.0,
                    "offset": 0.0,
                },
                "gripper_right3_to_gripper_right1": {
                    "joint": "gripper_controller",
                    "multiplier": 1.0,
                    "offset": 0.0,
                },
            },
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-len(self.gripper_joint_names)]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        pass