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
class DOFBOT_Pro(BaseAgent):
    uid = "dofbot_pro"
    urdf_path = Path(__file__).parent.resolve() / f'urdfs/dofbot_pro/DOFBOT_Pro-V24.urdf'
    disable_self_collisions = True
    arm_joint_names = [
        'Arm1_Joint', 
        'Arm2_Joint', 
        'Arm3_Joint', 
        'Arm4_Joint', 
        'Arm5_Joint'
    ]

    gripper_joint_names = [
        'grip_joint', 
        'llink_joint1', 
        'rlink_joint3', 
        'llink_joint3', 
        'rlink_joint2', 
        'llink_joint2'
    ]

    ee_link_name = "Gripping_point_Link"

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
                mount=self.robot.links_map["DaBai_DCW2_Link"],      # 直接挂在 URDF 的 camera_link
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
            lower=-1.6,   # 保证夹薄物体时仍有夹紧力
            upper=0.0,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={
                "llink_joint1": {
                    "joint": "grip_joint",
                    "multiplier": 1.0,
                    "offset": 0.0,
                },
                "rlink_joint3": {
                    "joint": "grip_joint",
                    "multiplier": -1.0,
                    "offset": 0.0,
                },
                "llink_joint3": {
                    "joint": "grip_joint",
                    "multiplier": 1.0,
                    "offset": 0.0,
                },
                "rlink_joint2": {
                    "joint": "grip_joint",
                    "multiplier": 1.0,
                    "offset": 0.0,
                },
                "llink_joint2": {
                    "joint": "grip_joint",
                    "multiplier": -1.0,
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
            self.robot.get_links(), "llink2"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "rlink2"
        )

        # JetCobot 没有 pad，可以直接复用 finger
        self.finger1pad_link = self.finger1_link
        self.finger2pad_link = self.finger2_link

        # TCP（你之前已经定义 gripper_tcp）
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )

        if l_contact_forces.shape[0] == 0 or r_contact_forces.shape[0] == 0:
            return torch.zeros(
                l_contact_forces.shape[0], dtype=torch.bool, device=l_contact_forces.device
            )

        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # JetCobot：X 轴是开合方向
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 0]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 0]

        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)

        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )

        return torch.logical_and(lflag, rflag)

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