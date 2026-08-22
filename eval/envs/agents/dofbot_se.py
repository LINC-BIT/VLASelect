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
from mani_skill.agents.controllers.base_controller import DictController

@register_agent()
class DOFBOT_SE(BaseAgent):
    uid = "dofbot_se"
    urdf_path = Path(__file__).parent.resolve() / f'urdfs/dofbot_se/dofbot.urdf'
    disable_self_collisions = True
    load_multiple_collisions = False
    arm_joint_names = [
        'arm_joint1', 
        'arm_joint2', 
        'arm_joint3', 
        'arm_joint4', 
        'arm_joint5'
    ]

    passive_gripper_joint_names = [  
        'llink_joint3',
        'rlink_joint3', 
        'llink_joint2', 
        'rlink_joint2', 
    ]

    finger_gripper_joint_names = [
        'llink_joint1',
        'grip_joint', 
    ]

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2., dynamic_friction=2., restitution=0.0)
        ),
        link=dict(
            llink2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            rlink2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        start=Keyframe(
            qpos=np.array(
                [0.0, 0.365, -np.pi / 2, -np.pi / 2, 0, -np.pi / 4, -np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4, -np.pi / 4]
            ),
            pose=sapien.Pose(),
        )
    )

    # urdf_config = dict(
    #     _materials=dict(
    #         gripper=dict(static_friction=1.5, dynamic_friction=1.5, restitution=0.0)
    #     ),
    #     link=dict(
    #         llink2=dict(
    #             material="gripper", patch_radius=0.02, min_patch_radius=0.02
    #         ),
    #         rlink2=dict(
    #             material="gripper", patch_radius=0.02, min_patch_radius=0.02
    #         ),
    #     ),
    # )

    ee_link_name = "gripper_tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_friction = 0.1
    arm_force_limit = 100

    gripper_stiffness = 1e5
    gripper_damping = 1e3
    gripper_friction = 0.05
    gripper_force_limit = 0.1

    resize_scale = 0.3125

    intrinsic_matrix = np.array([
        [956.26581799 * resize_scale, 0., 320.60169269 * resize_scale],
        [0., 956.70914603 * resize_scale, 286.20894899 * resize_scale],
        [0., 0., 1.,]
    ])

    distortion_params = np.array(
        [-0.4612075, 0.55314965, -0.00183297, -0.00125828, -1.11466445]
    )

    # arm_stiffness = 1e4
    # arm_damping = 1e3
    # arm_friction = 0.1
    # arm_force_limit = 100

    # gripper_stiffness = 1e5
    # gripper_damping = 2000
    # gripper_force_limit = 0.1
    # gripper_friction = 1

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0.00, 0, 0.01], q=[0.7071068, 0, -0.7071068, 0]),  # 相对于 camera_link 的本地坐标
                # width=int(640 / 3.75),
                # height=int(480 / 3.75),
                width=int(640 * self.resize_scale),
                height=int(480 * self.resize_scale),
                intrinsic=self.intrinsic_matrix.tolist(),
                # fov=np.pi / 180 * 23.2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["mono_link"],      # 直接挂在 URDF 的 camera_link
            )
        ]

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerForDofbotConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            friction=self.arm_friction,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerForDofbotConfig(
            self.arm_joint_names,
            lower=-0.2,
            upper=0.2,
            stiffness=self.arm_stiffness,
            friction=self.arm_friction,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True
        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.2,
            pos_upper=0.2,
            stiffness=self.arm_stiffness,
            friction=self.arm_friction,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=True,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.2,
            pos_upper=0.2,
            rot_lower=-0.2,
            rot_upper=0.2,
            stiffness=self.arm_stiffness,
            friction=self.arm_friction,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=True,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        passive_gripper_joints = PassiveControllerConfig(
            joint_names=self.passive_gripper_joint_names,
            damping=0,
            friction=0,
        )

        gripper_pd_joint_pos = PDJointPosMimicControllerForDofbotConfig(
            joint_names=self.finger_gripper_joint_names,
            lower=None, 
            upper=None,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            mimic={
                "llink_joint1": {
                    "joint": "grip_joint",
                    "multiplier": -1.0,
                    "offset": 0.0,
                },
            }
        )

        gripper_pd_joint_delta_pos = PDJointPosMimicControllerForDofbotConfig(
            joint_names=self.finger_gripper_joint_names,
            lower=-0.2, 
            upper=0.2,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            use_delta=True,
            mimic={
                "llink_joint1": {
                    "joint": "grip_joint",
                    "multiplier": -1.0,
                    "offset": 0.0,
                },
            }
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper_active=gripper_pd_joint_pos, gripper_passive=passive_gripper_joints
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper_active=gripper_pd_joint_pos, gripper_passive=passive_gripper_joints),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper_active=gripper_pd_joint_pos, gripper_passive=passive_gripper_joints),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper_active=gripper_pd_joint_pos, gripper_passive=passive_gripper_joints
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper_active=gripper_pd_joint_delta_pos, gripper_passive=passive_gripper_joints
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper_active=gripper_pd_joint_delta_pos, gripper_passive=passive_gripper_joints
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper_active=gripper_pd_joint_delta_pos, gripper_passive=passive_gripper_joints
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    
    def _after_init(self):
        self.left_tip_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "llink2"
        )
        self.right_tip_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "rlink2"
        )

        # direction links (决定开合方向)
        self.left_dir_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "llink1"
        )
        self.right_dir_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "rlink1"
        )

        # TCP（你之前已经定义 gripper_tcp）
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.base_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "base_link"
        )

    def _drive_pos(self, id):
        if id == 1:
            outer_knuckle = next(
                j for j in self.robot.get_active_joints() if j.name == "grip_joint"
            )
            outer_finger = next(
                j for j in self.robot.get_active_joints() if j.name == "rlink_joint2"
            )
            inner_knuckle = next(
                j for j in self.robot.get_active_joints() if j.name == "rlink_joint3"
            )

            pad = outer_finger.get_child_link()
            lif = inner_knuckle.get_child_link()
            p_w = (
                outer_finger.get_global_pose().p
                + inner_knuckle.get_global_pose().p
                - outer_knuckle.get_global_pose().p
            )[0]
            return p_w.cpu().tolist()
        else:
            outer_knuckle = next(
                j for j in self.robot.get_active_joints() if j.name == "llink_joint1"
            )
            outer_finger = next(
                j for j in self.robot.get_active_joints() if j.name == "llink_joint2"
            )
            inner_knuckle = next(
                j for j in self.robot.get_active_joints() if j.name == "llink_joint3"
            )
            pad = outer_finger.get_child_link()
            lif = inner_knuckle.get_child_link()
            p_w = (
                outer_finger.get_global_pose().p
                + inner_knuckle.get_global_pose().p
                - outer_knuckle.get_global_pose().p
            )[0]
            return p_w.cpu().tolist()

    def _get_gripper_constraint(self):
        outer_knuckle = next(
            j for j in self.robot.get_active_joints() if j.name == "grip_joint"
        )
        outer_finger = next(
            j for j in self.robot.get_active_joints() if j.name == "rlink_joint2"
        )
        inner_knuckle = next(
            j for j in self.robot.get_active_joints() if j.name == "rlink_joint3"
        )

        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()
        p_w = (
            outer_finger.get_global_pose().p
            + inner_knuckle.get_global_pose().p
            - outer_knuckle.get_global_pose().p
        )[0]

        

        T_pw = pad.pose.inv().to_transformation_matrix()[0]
        T_fw = lif.pose.inv().to_transformation_matrix()[0]
        p_f_right = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p_right = T_pw[:3, :3] @ p_w + T_pw[:3, 3]

        R1 = lif.pose.to_transformation_matrix()[0][:3, :3]
        t1 = lif.pose.to_transformation_matrix()[0][:3, 3]

        R2 = pad.pose.to_transformation_matrix()[0][:3, :3]
        t2 = pad.pose.to_transformation_matrix()[0][:3, 3]

        # p1 = R1 @ torch.Tensor(p_f_right) + t1
        # p2 = R2 @ torch.Tensor(p_p_right) + t2

        # print(R1.T @ R2)

        outer_knuckle = next(
            j for j in self.robot.get_active_joints() if j.name == "llink_joint1"
        )
        outer_finger = next(
            j for j in self.robot.get_active_joints() if j.name == "llink_joint2"
        )
        inner_knuckle = next(
            j for j in self.robot.get_active_joints() if j.name == "llink_joint3"
        )
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()
        p_w = (
            outer_finger.get_global_pose().p
            + inner_knuckle.get_global_pose().p
            - outer_knuckle.get_global_pose().p
        )[0]

        # print(p_w)

        T_pw = pad.pose.inv().to_transformation_matrix()[0]
        T_fw = lif.pose.inv().to_transformation_matrix()[0]
        p_f_left = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p_left = T_pw[:3, :3] @ p_w + T_pw[:3, 3]

        return p_f_right, p_p_right, p_f_left, p_p_left

    def _after_loading_articulation(self):
        # j1->      ?
        # |         |
        # j2        j3
        # the next 4 magic arrays come from https://github.com/haosulab/cvpr-tutorial-2022/blob/master/debug/robotiq.py which was
        # used to precompute these poses for drive creation
        p_f_right = [0.029999971389770508, -0.0007542740786448121, 2.7008354663848877e-08]
        p_p_right = [0.018000006675720215, 0.00812501460313797, -7.82310962677002e-08]
        p_f_left = [0.02999994345009327, 0.0006510481471195817, 3.259629011154175e-08]
        p_p_left = [0.018000036478042603, -0.00787498988211155, -4.330649971961975e-08]

        # for name in self.passive_gripper_joint_names:
        #     joint = self.robot.active_joints_map[name]
        #     joint.set_drive_target(0)
        #     joint.set_drive_properties(stiffness=1e5, damping=1e3)

        # for name in self.finger_gripper_joint_names:
        #     joint = self.robot.active_joints_map[name]
        #     joint.set_drive_target(0)
        #     joint.set_drive_properties(stiffness=1e5, damping=1e3)

        # # p_f_right, p_p_right, p_f_left, p_p_left = self._get_gripper_constraint()
        # # print("p_f_right = ", p_f_right.cpu().tolist())
        # # print("p_p_right = ", p_p_right.cpu().tolist())
        # # print("p_f_left = ", p_f_left.cpu().tolist())
        # # print("p_p_left = ", p_p_left.cpu().tolist())
        
        outer_finger = self.robot.active_joints_map["rlink_joint2"]
        inner_knuckle = self.robot.active_joints_map["rlink_joint3"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        # R1 = lif.pose.to_transformation_matrix()[0][:3, :3]
        # t1 = lif.pose.to_transformation_matrix()[0][:3, 3]

        # R2 = pad.pose.to_transformation_matrix()[0][:3, :3]
        # t2 = pad.pose.to_transformation_matrix()[0][:3, 3]

        # p1 = R1 @ torch.Tensor(p_f_right) + t1
        # p2 = R2 @ torch.Tensor(p_p_right) + t2

        # print(p1 - p2)

        self.right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right)
        )
        self.right_drive.set_limit_x(0, 0)
        self.right_drive.set_limit_y(0, 0)
        self.right_drive.set_limit_z(0, 0)

        outer_finger = self.robot.active_joints_map["llink_joint2"]
        inner_knuckle = self.robot.active_joints_map["llink_joint3"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        self.left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left)
        )
        self.left_drive.set_limit_x(0, 0)
        self.left_drive.set_limit_y(0, 0)
        self.left_drive.set_limit_z(0, 0)
        
        # self.right_drive.set_drive_property_x(stiffness=1e8, damping=0)
        # self.right_drive.set_drive_property_y(stiffness=1e8, damping=0)
        # self.right_drive.set_drive_property_z(stiffness=1e8, damping=0)

        # self.left_drive.set_drive_property_x(stiffness=1e8, damping=0)
        # self.left_drive.set_drive_property_y(stiffness=1e8, damping=0)
        # self.left_drive.set_drive_property_z(stiffness=1e8, damping=0)
        
        # [
        #     x.set_limit_twist(0, 0)
        #     for x in self.right_drive._objs
        # ]

        # [
        #     x.set_limit_twist(0, 0)
        #     for x in self.left_drive._objs
        # ]
        

        # [
        #     x.set_drive_property_swing(1e5, 1e2)
        #     for x in right_drive._objs
        # ]
        
        # [
        #     x.set_drive_property_twist(1e8, 0)
        #     for x in self.right_drive._objs
        # ]

        # [
        #     x.set_drive_property_swing(1e5, 1e2)
        #     for x in left_drive._objs
        # ]

        # [
        #     x.set_drive_property_twist(1e8, 0)
        #     for x in self.left_drive._objs
        # ]

        # disable impossible collisions here instead of just using the SRDF as there are too many
        # and using the SRDF will cause the robot to assign too many collision groups

        # disable all collisions between gripper related links
        gripper_links = [
            "rlink1",
            "rlink2",
            "rlink3",
            "llink1",
            "llink2",
            "llink3",
            "arm_link5",  # not gripper link but is adjacent to the gripper part
        ]
        for link_name in gripper_links:
            link = self.robot.links_map[link_name]
            link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def is_touching(self, object: Actor, min_force=0.5):
        """
        Check if gripper is touching an object.

        Force detection:
            llink2 / rlink2
        """

        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.left_tip_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.right_tip_link, object
        )

        lforce = torch.linalg.norm(l_contact_forces, dim=1)
        rforce = torch.linalg.norm(r_contact_forces, dim=1)

        lflag = lforce >= min_force
        rflag = rforce >= min_force

        return torch.logical_and(lflag, rflag)

    @property
    def start_action(self):
        if isinstance(self.controller, DictController):
            action = self.controller.from_qpos(self.keyframes['start'].qpos)
        else:
            action = self.keyframes['start'].qpos
        return action

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """
        Check if gripper is grasping an object.

        Force detection:
            llink2 / rlink2

        Direction estimation:
            llink1 / rlink1
        """

        # -----------------------------
        # 1. contact forces (tip links)
        # -----------------------------
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.left_tip_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.right_tip_link, object
        )

        lforce = torch.linalg.norm(l_contact_forces, dim=1)
        rforce = torch.linalg.norm(r_contact_forces, dim=1)

        # -----------------------------
        # 2. opening direction (link1)
        # -----------------------------
        ldirection = self.left_dir_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.right_dir_link.pose.to_transformation_matrix()[..., :3, 1]

        # -----------------------------
        # 3. contact angle
        # -----------------------------
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)

        # -----------------------------
        # 4. grasp condition
        # -----------------------------
        lflag = torch.logical_and(
            lforce >= min_force,
            torch.rad2deg(langle) <= max_angle,
        )

        rflag = torch.logical_and(
            rforce >= min_force,
            torch.rad2deg(rangle) <= max_angle,
        )

        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-6]
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

def _quantize_qpos(qpos, only_gripper=False, only_arm=False):

    q_deg = qpos * 180.0 / torch.pi
    
    is_np = False
    if isinstance(q_deg, np.ndarray):
        is_np = True
        q_deg = torch.from_numpy(q_deg)
    
    if only_arm:
        q_deg_main = q_deg
        q_deg_main = q_deg_main.to(torch.int).to(torch.float32)
        q_deg = q_deg_main
    elif only_gripper:
        q_deg_gripper = q_deg
        q_deg_gripper = q_deg_gripper * 2.0
        q_deg_gripper = q_deg_gripper.to(torch.int).to(torch.float32)
        q_deg_gripper = q_deg_gripper / 2.0
        q_deg = q_deg_gripper

    q_rad = q_deg * torch.pi / 180.0
    if is_np:
        q_rad = q_rad.numpy()

    return q_rad

from mani_skill.utils.structs.types import Array
from dataclasses import dataclass

class PDJointPosControllerForDofbot(PDJointPosController):
    config: "PDJointPosControllerForDofbotConfig"
    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + action
            else:
                self._target_qpos = self._start_qpos + action
        else:
            # Compatible with mimic controllers. Need to clone here otherwise cannot do in-place replacements in the reset function
            self._target_qpos = torch.broadcast_to(
                action, self._start_qpos.shape
            ).clone()

        self._target_qpos = _quantize_qpos(self._target_qpos, only_arm=True)
        
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

class PDJointPosMimicControllerForDofbot(PDJointPosMimicController):
    config: "PDJointPosMimicControllerForDofbotConfig"
    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos[:, self.control_joint_indices] += action
            else:
                self._target_qpos[:, self.control_joint_indices] = (
                    self._start_qpos[:, self.control_joint_indices] + action
                )
        else:
            self._target_qpos[:, self.control_joint_indices] = action
        self._target_qpos[:, self.mimic_joint_indices] = (
            self._target_qpos[:, self.mimic_control_joint_indices]
            * self._multiplier[None, :]
            + self._offset[None, :]
        )
        self._target_qpos = _quantize_qpos(self._target_qpos, only_gripper=True)
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

@dataclass
class PDJointPosControllerForDofbotConfig(PDJointPosControllerConfig):
    controller_cls = PDJointPosControllerForDofbot

@dataclass
class PDJointPosMimicControllerForDofbotConfig(PDJointPosMimicControllerConfig):
    controller_cls = PDJointPosMimicControllerForDofbot

if __name__ == '__main__':
    DOFBOT_SE.get_state()