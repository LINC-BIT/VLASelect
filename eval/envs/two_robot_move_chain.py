import os
from typing import Any, Tuple

import numpy as np
import sapien
import torch
from sapien.pysapien import physx

os.environ.setdefault("MS_ASSET_DIR", "/home/Maniskill/.maniskill")

import envs.agents.ur10e_panda_gripper  # noqa: F401
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.agents.robots.xarm6.xarm6_robotiq import XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


def _yaw_quat(theta: float):
    return [float(np.cos(theta / 2.0)), 0.0, 0.0, float(np.sin(theta / 2.0))]


def _axis_to_y_quat():
    theta = np.pi / 2
    return [float(np.cos(theta / 2.0)), 0.0, 0.0, float(np.sin(theta / 2.0))]


def _axis_to_z_quat():
    theta = -np.pi / 2
    return [float(np.cos(theta / 2.0)), 0.0, float(np.sin(theta / 2.0)), 0.0]


def _disable_internal_collision(link_builder):
    # Put all chain links into the same articulation self-collision filter group.
    link_builder.collision_groups = [1, 1, 2, 0]


def _sample_yaw_quat_from_yaw(yaw: torch.Tensor):
    q = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=torch.float32)
    q[:, 0] = torch.cos(yaw / 2.0)
    q[:, 3] = torch.sin(yaw / 2.0)
    return q


@register_env("TwoRobotMoveChain-v1", max_episode_steps=200)
class TwoRobotMoveChainEnv(BaseEnv):
    SUPPORTED_ROBOTS = [
        ("xarm6_robotiq", "panda_wristcam"),
        ("ur10e_panda_gripper", "panda_wristcam"),
    ]
    agent: MultiAgent[Tuple[XArm6Robotiq, PandaWristCam]]
    SUPPORTED_REWARD_MODES = ["none", "normalized_dense"]
    top_approach_stage_bonus = 2.5
    dual_grasp_stage_bonus = 4.5
    lifted_stage_bonus = 9.0
    placed_stage_bonus = 19.0

    def __init__(
        self,
        *args,
        robot_uids=("xarm6_robotiq", "panda_wristcam"),
        use_rigid_bar=True,
        robot_init_qpos_noise=0.02,
        left_robot_xy=(-0.70, 0.02),
        right_robot_xy=(0.93, 0.02),
        left_robot_z=None,
        right_robot_z=-0.1,
        left_robot_yaw=0.0,
        right_robot_yaw=np.pi,
        chain_num_links=11,
        chain_link_length=0.05,
        chain_link_half_width=0.012,
        rigid_bar_grasp_inset=0.04,
        chain_top_grasp_z_offset=0.02,
        chain_contact_grasp_z_offset=0.004,
        chain_joint_limit=np.pi / 2,
        chain_joint_damping=0.08,
        chain_stretch_limit=0.012,
        chain_stretch_damping=0.12,
        chain_density=200.0,
        chain_static_friction=0.1,
        chain_dynamic_friction=0.08,
        chain_restitution=0.0,
        sticky_grasp_enabled=False,
        sticky_grasp_release_gap=0.07,
        chain_root_xy=(-0.25, -0.18),
        chain_root_height=0.08,
        chain_root_xy_random=(0.03, 0.02),
        chain_yaw_random_deg=15.0,
        chain_bend_random=0.12,
        chain_s_shape_random=0.05,
        obstacle_half_size=(0.34, 0.03, 0.01),
        obstacle_xy=(0.0, 0.02),
        groove_offset_x=0.18,
        groove_y=0.14,
        goal_xy_random=(0.02, 0.02),
        goal_thresh=0.06,
        fail_goal_dist_thresh=1.0,
        obs_sanitize_clip=10.0,
        goal_center_thresh=0.08,
        goal_align_cos_thresh=0.92,
        goal_span_ratio_thresh=0.75,
        goal_wall_clearance=0.03,
        **kwargs,
    ):
        self.use_rigid_bar = bool(use_rigid_bar)
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        if (
            robot_uids[0] == "ur10e_panda_gripper"
            and tuple(float(v) for v in left_robot_xy) == (-0.70, 0.02)
        ):
            left_robot_xy = (-0.82, 0.02)
        self.left_robot_xy = tuple(float(v) for v in left_robot_xy)
        self.right_robot_xy = tuple(float(v) for v in right_robot_xy)
        if left_robot_z is None:
            left_robot_z = -0.10 if robot_uids[0] == "ur10e_panda_gripper" else 0.0
        self.left_robot_z = float(left_robot_z)
        self.right_robot_z = float(right_robot_z)
        self.left_robot_yaw = float(left_robot_yaw)
        self.right_robot_yaw = float(right_robot_yaw)
        self.chain_num_links = int(chain_num_links)
        self.chain_link_length = float(chain_link_length)
        self.chain_link_half_width = float(chain_link_half_width)
        self.rigid_bar_grasp_inset = float(rigid_bar_grasp_inset)
        self.chain_top_grasp_z_offset = float(chain_top_grasp_z_offset)
        self.chain_contact_grasp_z_offset = float(chain_contact_grasp_z_offset)
        self.chain_joint_limit = float(chain_joint_limit)
        self.chain_joint_damping = float(chain_joint_damping)
        self.chain_stretch_limit = float(chain_stretch_limit)
        self.chain_stretch_damping = float(chain_stretch_damping)
        self.chain_density = float(chain_density)
        self.chain_static_friction = float(chain_static_friction)
        self.chain_dynamic_friction = float(chain_dynamic_friction)
        self.chain_restitution = float(chain_restitution)
        self.sticky_grasp_enabled = bool(sticky_grasp_enabled)
        self.sticky_grasp_release_gap = float(sticky_grasp_release_gap)
        self.chain_root_xy = tuple(float(v) for v in chain_root_xy)
        self.chain_root_height = float(chain_root_height)
        self.chain_root_xy_random = tuple(float(v) for v in chain_root_xy_random)
        self.chain_yaw_random = np.deg2rad(float(chain_yaw_random_deg))
        self.chain_bend_random = float(chain_bend_random)
        self.chain_s_shape_random = float(chain_s_shape_random)
        self.obstacle_half_size = tuple(float(v) for v in obstacle_half_size)
        self.obstacle_xy = tuple(float(v) for v in obstacle_xy)
        self.chain_spawn_clearance = self.chain_link_half_width + 0.01
        self.groove_offset_x = float(groove_offset_x)
        self.groove_y = float(groove_y)
        self.goal_xy_random = tuple(float(v) for v in goal_xy_random)
        self.goal_thresh = float(goal_thresh)
        self.fail_goal_dist_thresh = float(fail_goal_dist_thresh)
        self.obs_sanitize_clip = float(obs_sanitize_clip)
        self.goal_center_thresh = float(goal_center_thresh)
        self.goal_align_cos_thresh = float(goal_align_cos_thresh)
        self.goal_span_ratio_thresh = float(goal_span_ratio_thresh)
        self.goal_wall_clearance = float(goal_wall_clearance)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.0, -0.82, 1.02], [0.0, 0.02, 0.12])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 3, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.0, -0.82, 1.02], [0.0, 0.02, 0.12])
        return CameraConfig("render_camera", pose, 512, 512, np.pi / 3, 0.01, 100)

    @property
    def left_agent(self) -> XArm6Robotiq:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def _left_robot_pose(self):
        return sapien.Pose(
            p=[self.left_robot_xy[0], self.left_robot_xy[1], self.left_robot_z],
            q=_yaw_quat(self.left_robot_yaw),
        )

    def _right_robot_pose(self):
        return sapien.Pose(
            p=[self.right_robot_xy[0], self.right_robot_xy[1], self.right_robot_z],
            q=_yaw_quat(self.right_robot_yaw),
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, [self._left_robot_pose(), self._right_robot_pose()])

    def _build_chain_articulation(self):
        if self.use_rigid_bar:
            builder = self.scene.create_articulation_builder()
            builder.set_name("move_bar")
            builder.disable_self_collisions = True
            chain_material = physx.PhysxMaterial(
                self.chain_static_friction,
                self.chain_dynamic_friction,
                self.chain_restitution,
            )
            builder.initial_pose = sapien.Pose(
                p=[self.chain_root_xy[0], self.chain_root_xy[1], self.chain_root_height]
            )

            total_length = self.chain_num_links * self.chain_link_length
            endpoint_radius = self.chain_link_half_width * 1.2
            endpoint_offset = max(total_length / 2 - self.rigid_bar_grasp_inset, 0.0)
            mid_name = f"segment_{self.chain_num_links // 2}"

            root = builder.create_link_builder()
            root.set_name(mid_name)
            root.set_joint_name("root_joint")
            _disable_internal_collision(root)
            root.set_joint_properties(
                "undefined",
                [],
                pose_in_parent=sapien.Pose(),
                pose_in_child=sapien.Pose(),
            )
            root.add_box_collision(
                pose=sapien.Pose(),
                half_size=[
                    total_length / 2,
                    self.chain_link_half_width,
                    self.chain_link_half_width,
                ],
                material=chain_material,
                density=self.chain_density,
            )
            root.add_box_visual(
                pose=sapien.Pose(),
                half_size=[
                    total_length / 2,
                    self.chain_link_half_width,
                    self.chain_link_half_width,
                ],
                material=sapien.render.RenderMaterial(base_color=[0.9, 0.55, 0.2, 1.0]),
            )

            left_end = builder.create_link_builder(parent=root)
            left_end.set_name("segment_0")
            left_end.set_joint_name("left_endpoint_joint")
            _disable_internal_collision(left_end)
            left_end.set_joint_properties(
                "fixed",
                [],
                pose_in_parent=sapien.Pose(p=[-endpoint_offset, 0.0, 0.0]),
                pose_in_child=sapien.Pose(),
            )
            left_end.add_sphere_collision(
                radius=endpoint_radius,
                material=chain_material,
                density=self.chain_density,
            )
            left_end.add_sphere_visual(
                radius=endpoint_radius,
                material=sapien.render.RenderMaterial(base_color=[0.2, 0.55, 0.95, 1.0]),
            )

            right_end = builder.create_link_builder(parent=root)
            right_end.set_name(f"segment_{self.chain_num_links - 1}")
            right_end.set_joint_name("right_endpoint_joint")
            _disable_internal_collision(right_end)
            right_end.set_joint_properties(
                "fixed",
                [],
                pose_in_parent=sapien.Pose(p=[endpoint_offset, 0.0, 0.0]),
                pose_in_child=sapien.Pose(),
            )
            right_end.add_sphere_collision(
                radius=endpoint_radius,
                material=chain_material,
                density=self.chain_density,
            )
            right_end.add_sphere_visual(
                radius=endpoint_radius,
                material=sapien.render.RenderMaterial(base_color=[0.2, 0.85, 0.45, 1.0]),
            )

            self.chain = builder.build(fix_root_link=False)
            self.chain_left_end = self.chain.links_map["segment_0"]
            self.chain_right_end = self.chain.links_map[f"segment_{self.chain_num_links - 1}"]
            self.chain_mid_link = self.chain.links_map[mid_name]
            self.chain_dof = int(self.chain.max_dof)
            return

        builder = self.scene.create_articulation_builder()
        builder.set_name("move_chain")
        builder.disable_self_collisions = True
        chain_material = physx.PhysxMaterial(
            self.chain_static_friction,
            self.chain_dynamic_friction,
            self.chain_restitution,
        )
        builder.initial_pose = sapien.Pose(
            p=[self.chain_root_xy[0], self.chain_root_xy[1], self.chain_root_height]
        )

        root = builder.create_link_builder()
        root.set_name("segment_0")
        root.set_joint_name("root_joint")
        _disable_internal_collision(root)
        root.set_joint_properties(
            "undefined",
            [],
            pose_in_parent=sapien.Pose(),
            pose_in_child=sapien.Pose(),
        )
        segment_pose = sapien.Pose(p=[self.chain_link_length / 2, 0.0, 0.0])
        root.add_box_collision(
            pose=segment_pose,
            half_size=[
                self.chain_link_length / 2,
                self.chain_link_half_width,
                self.chain_link_half_width,
            ],
            material=chain_material,
            density=self.chain_density,
        )
        root.add_box_visual(
            pose=segment_pose,
            half_size=[
                self.chain_link_length / 2,
                self.chain_link_half_width,
                self.chain_link_half_width,
            ],
            material=sapien.render.RenderMaterial(base_color=[0.9, 0.55, 0.2, 1.0]),
        )

        parent = root
        z_axis_q = _axis_to_z_quat()
        y_axis_q = _axis_to_y_quat()
        for idx in range(1, self.chain_num_links):
            stretch_x = builder.create_link_builder(parent=parent)
            stretch_x.set_name(f"stretch_x_{idx}")
            stretch_x.set_joint_name(f"joint_x_{idx}")
            _disable_internal_collision(stretch_x)
            stretch_x.set_joint_properties(
                "prismatic",
                [[0.0, self.chain_stretch_limit]],
                pose_in_parent=sapien.Pose(
                    p=[self.chain_link_length, 0.0, 0.0],
                ),
                pose_in_child=sapien.Pose(),
                damping=self.chain_stretch_damping,
            )

            bend_z = builder.create_link_builder(parent=stretch_x)
            bend_z.set_name(f"bend_z_{idx}")
            bend_z.set_joint_name(f"joint_z_{idx}")
            _disable_internal_collision(bend_z)
            bend_z.set_joint_properties(
                "revolute",
                [[-self.chain_joint_limit, self.chain_joint_limit]],
                pose_in_parent=sapien.Pose(
                    q=z_axis_q,
                ),
                pose_in_child=sapien.Pose(q=z_axis_q),
                damping=self.chain_joint_damping,
            )

            link = builder.create_link_builder(parent=bend_z)
            link.set_name(f"segment_{idx}")
            link.set_joint_name(f"joint_y_{idx}")
            _disable_internal_collision(link)
            link.set_joint_properties(
                "revolute",
                [[-self.chain_joint_limit, self.chain_joint_limit]],
                pose_in_parent=sapien.Pose(q=y_axis_q),
                pose_in_child=sapien.Pose(q=y_axis_q),
                damping=self.chain_joint_damping,
            )
            link.add_box_collision(
                pose=segment_pose,
                half_size=[
                    self.chain_link_length / 2,
                    self.chain_link_half_width,
                    self.chain_link_half_width,
                ],
                material=chain_material,
                density=self.chain_density,
            )
            link.add_box_visual(
                pose=segment_pose,
                half_size=[
                    self.chain_link_length / 2,
                    self.chain_link_half_width,
                    self.chain_link_half_width,
                ],
                material=sapien.render.RenderMaterial(
                    base_color=[0.2, 0.45 + 0.05 * idx, 0.85 - 0.04 * idx, 1.0]
                ),
            )
            parent = link

        self.chain = builder.build(fix_root_link=False)
        self.chain_left_end = self.chain.links_map["segment_0"]
        self.chain_right_end = self.chain.links_map[f"segment_{self.chain_num_links - 1}"]
        self.chain_mid_link = self.chain.links_map[f"segment_{self.chain_num_links // 2}"]
        self.chain_dof = int(self.chain.max_dof)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=0.0)
        self.table_scene.build()
        self._build_chain_articulation()
        self.left_endpoint_sticky = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.right_endpoint_sticky = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.top_approach_achieved = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.dual_grasp_achieved = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.chain_lifted_over_wall = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self._saved_left_qpos = None
        self._saved_left_qvel = None
        self._saved_right_qpos = None
        self._saved_right_qvel = None

        self.obstacle = actors.build_box(
            self.scene,
            half_sizes=list(self.obstacle_half_size),
            color=[0.45, 0.45, 0.45, 1.0],
            name="obstacle_wall",
            body_type="static",
            initial_pose=sapien.Pose(
                p=[
                    self.obstacle_xy[0],
                    self.obstacle_xy[1],
                    self.obstacle_half_size[2],
                ]
            ),
        )
        self.goal_left = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0.1, 0.9, 0.1, 1.0],
            body_type="kinematic",
            add_collision=False,
            name="goal_left",
            initial_pose=sapien.Pose(
                p=[-self.groove_offset_x, self.groove_y, self.chain_root_height]
            ),
        )
        self.goal_right = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0.1, 0.9, 0.1, 1.0],
            body_type="kinematic",
            add_collision=False,
            name="goal_right",
            initial_pose=sapien.Pose(
                p=[self.groove_offset_x, self.groove_y, self.chain_root_height]
            ),
        )
        self._hidden_objects.append(self.goal_left)
        self._hidden_objects.append(self.goal_right)

    def _sample_chain_pose(self, batch_size: int):
        xy = torch.zeros((batch_size, 2), device=self.device)
        xy[:, 0] = self.chain_root_xy[0] + (
            torch.rand((batch_size,), device=self.device) * 2 - 1
        ) * self.chain_root_xy_random[0]
        xy[:, 1] = self.chain_root_xy[1] + (
            torch.rand((batch_size,), device=self.device) * 2 - 1
        ) * self.chain_root_xy_random[1]
        z = torch.full((batch_size, 1), self.chain_root_height, device=self.device)
        p = torch.cat([xy, z], dim=-1)
        yaw = (torch.rand((batch_size,), device=self.device) * 2 - 1) * self.chain_yaw_random
        q = _sample_yaw_quat_from_yaw(yaw)
        return Pose.create_from_pq(p, q)

    def _sample_chain_qpos(self, batch_size: int):
        qpos = torch.zeros((batch_size, self.chain_dof), device=self.device)
        if self.chain_dof == 0:
            return qpos
        pair_count = self.chain_num_links - 1
        t = torch.linspace(-1.0, 1.0, pair_count, device=self.device).unsqueeze(0)
        yaw_arc_amp = (torch.rand((batch_size, 1), device=self.device) * 2 - 1) * self.chain_bend_random
        yaw_s_amp = (torch.rand((batch_size, 1), device=self.device) * 2 - 1) * self.chain_s_shape_random
        pitch_amp = (torch.rand((batch_size, 1), device=self.device) * 2 - 1) * (
            0.5 * self.chain_bend_random
        )
        stretch = (
            torch.rand((batch_size, pair_count), device=self.device) * 0.35 * self.chain_stretch_limit
        )
        yaw_phase = torch.rand((batch_size, 1), device=self.device) * np.pi
        pitch_phase = torch.rand((batch_size, 1), device=self.device) * np.pi
        yaw = yaw_arc_amp * t + yaw_s_amp * torch.sin(np.pi * t + yaw_phase)
        pitch = pitch_amp * torch.sin(2 * np.pi * t + pitch_phase)
        qpos[:, 0::3] = stretch
        qpos[:, 1::3] = yaw
        qpos[:, 2::3] = pitch
        qpos[:, 0::3] += 0.002 * torch.rand_like(qpos[:, 0::3])
        qpos[:, 1::3] += 0.01 * torch.randn_like(qpos[:, 1::3])
        qpos[:, 2::3] += 0.01 * torch.randn_like(qpos[:, 2::3])
        qpos[:, 0::3] = torch.clamp(qpos[:, 0::3], min=0.0, max=self.chain_stretch_limit)
        qpos[:, 1::3] = torch.clamp(qpos[:, 1::3], min=-0.45, max=0.45)
        qpos[:, 2::3] = torch.clamp(qpos[:, 2::3], min=-0.45, max=0.45)
        return qpos

    def _chain_xy_intersects_wall(self, root_xy: torch.Tensor, root_yaw: torch.Tensor, qpos: torch.Tensor):
        batch_size = root_xy.shape[0]
        x_min = self.obstacle_xy[0] - self.obstacle_half_size[0] - self.chain_spawn_clearance
        x_max = self.obstacle_xy[0] + self.obstacle_half_size[0] + self.chain_spawn_clearance
        y_min = self.obstacle_xy[1] - self.obstacle_half_size[1] - self.chain_spawn_clearance
        y_max = self.obstacle_xy[1] + self.obstacle_half_size[1] + self.chain_spawn_clearance

        pos = root_xy.clone()
        theta = root_yaw.clone()
        intersects = torch.zeros((batch_size,), device=self.device, dtype=torch.bool)

        for seg_idx in range(self.chain_num_links):
            dir_xy = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
            center = pos + 0.5 * self.chain_link_length * dir_xy
            in_wall = (
                (center[:, 0] >= x_min)
                & (center[:, 0] <= x_max)
                & (center[:, 1] >= y_min)
                & (center[:, 1] <= y_max)
            )
            intersects = torch.logical_or(intersects, in_wall)
            if seg_idx < self.chain_num_links - 1 and qpos.shape[1] > 0:
                step_len = self.chain_link_length + qpos[:, 3 * seg_idx]
                pos = pos + step_len.unsqueeze(1) * dir_xy
                theta = theta + qpos[:, 3 * seg_idx + 1]
            else:
                pos = pos + self.chain_link_length * dir_xy
        return intersects

    def _sample_valid_chain_init(self, batch_size: int, max_tries: int = 24):
        root_xy = torch.zeros((batch_size, 2), device=self.device, dtype=torch.float32)
        root_yaw = torch.zeros((batch_size,), device=self.device, dtype=torch.float32)
        qpos = torch.zeros((batch_size, self.chain_dof), device=self.device, dtype=torch.float32)
        valid = torch.zeros((batch_size,), device=self.device, dtype=torch.bool)

        for _ in range(max_tries):
            remaining = ~valid
            if not torch.any(remaining):
                break
            rem_count = int(remaining.sum().item())

            sample_pose = self._sample_chain_pose(rem_count)
            sample_qpos = self._sample_chain_qpos(rem_count)
            sample_xy = sample_pose.p[:, :2]
            sample_yaw = 2.0 * torch.atan2(sample_pose.q[:, 3], sample_pose.q[:, 0])
            sample_valid = ~self._chain_xy_intersects_wall(sample_xy, sample_yaw, sample_qpos)

            take = remaining.nonzero(as_tuple=False).squeeze(-1)
            root_xy[take] = sample_xy
            root_yaw[take] = sample_yaw
            qpos[take] = sample_qpos
            valid[take] = sample_valid

        if not torch.all(valid):
            invalid = (~valid).nonzero(as_tuple=False).squeeze(-1)
            root_xy[invalid, 0] = self.chain_root_xy[0]
            root_xy[invalid, 1] = self.chain_root_xy[1]
            root_yaw[invalid] = 0.0
            qpos[invalid] = 0.0

        root_z = torch.full((batch_size, 1), self.chain_root_height, device=self.device, dtype=torch.float32)
        root_p = torch.cat([root_xy, root_z], dim=1)
        if self.use_rigid_bar:
            total_length = self.chain_num_links * self.chain_link_length
            center_offset = torch.stack(
                [
                    torch.cos(root_yaw) * (0.5 * total_length),
                    torch.sin(root_yaw) * (0.5 * total_length),
                    torch.zeros_like(root_yaw),
                ],
                dim=1,
            )
            root_p = root_p + center_offset
        root_pose = Pose.create_from_pq(
            root_p,
            _sample_yaw_quat_from_yaw(root_yaw),
        )
        return root_pose, qpos

    def _sample_goal_positions(self, batch_size: int):
        left = torch.tensor(
            [-self.groove_offset_x, self.groove_y, self.chain_root_height],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(batch_size, 1)
        right = torch.tensor(
            [self.groove_offset_x, self.groove_y, self.chain_root_height],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(batch_size, 1)
        left[:, 0] += (torch.rand((batch_size,), device=self.device) * 2 - 1) * self.goal_xy_random[0]
        left[:, 1] += (torch.rand((batch_size,), device=self.device) * 2 - 1) * self.goal_xy_random[1]
        right[:, 0] += (torch.rand((batch_size,), device=self.device) * 2 - 1) * self.goal_xy_random[0]
        right[:, 1] += (torch.rand((batch_size,), device=self.device) * 2 - 1) * self.goal_xy_random[1]
        min_goal_y = (
            self.obstacle_xy[1]
            + self.obstacle_half_size[1]
            + self.goal_thresh
            + self.goal_wall_clearance
        )
        left[:, 1] = torch.clamp(left[:, 1], min=min_goal_y)
        right[:, 1] = torch.clamp(right[:, 1], min=min_goal_y)
        return left, right

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.left_endpoint_sticky[env_idx] = False
            self.right_endpoint_sticky[env_idx] = False
            self.top_approach_achieved[env_idx] = False
            self.dual_grasp_achieved[env_idx] = False
            self.chain_lifted_over_wall[env_idx] = False

            left_qpos = self.left_agent.robot.get_qpos().clone()
            right_qpos = self.right_agent.robot.get_qpos().clone()
            left_rest = torch.as_tensor(
                self.left_agent.keyframes["rest"].qpos,
                device=self.device,
                dtype=left_qpos.dtype,
            ).repeat(b, 1)
            if getattr(self.right_agent, "uid", None) == "panda_wristcam":
                right_rest = torch.tensor(
                    [0.0, 0.4, 0.0, -1.8, 0.0, 2.1, 0.7, 0.04, 0.04],
                    device=self.device,
                    dtype=right_qpos.dtype,
                ).repeat(b, 1)
            else:
                right_rest = torch.as_tensor(
                    self.right_agent.keyframes["rest"].qpos,
                    device=self.device,
                    dtype=right_qpos.dtype,
                ).repeat(b, 1)
            left_rest += (torch.rand_like(left_rest) * 2 - 1) * self.robot_init_qpos_noise
            right_rest += (torch.rand_like(right_rest) * 2 - 1) * self.robot_init_qpos_noise
            left_qpos[env_idx] = left_rest
            right_qpos[env_idx] = right_rest
            self.left_agent.robot.set_qpos(left_qpos[env_idx])
            self.right_agent.robot.set_qpos(right_qpos[env_idx])
            self.left_agent.robot.set_pose(self._left_robot_pose())
            self.right_agent.robot.set_pose(self._right_robot_pose())
            self.left_init_qpos = self.left_agent.robot.get_qpos().clone()
            self.right_init_qpos = self.right_agent.robot.get_qpos().clone()

            all_qpos = self.chain.get_qpos().clone()
            all_qvel = self.chain.get_qvel().clone()
            root_pose, qpos = self._sample_valid_chain_init(b)
            qvel = torch.zeros((b, self.chain_dof), device=self.device)
            all_root_pose = self.chain.pose.raw_pose.clone()
            all_root_pose[env_idx] = root_pose.raw_pose
            self.chain.set_pose(Pose.create(all_root_pose[env_idx]))
            all_qpos[env_idx] = qpos
            all_qvel[env_idx] = qvel
            self.chain.set_qpos(all_qpos[env_idx])
            self.chain.set_qvel(all_qvel[env_idx])
            self.chain.set_root_linear_velocity(
                torch.zeros((b, 3), device=self.device)
            )
            self.chain.set_root_angular_velocity(
                torch.zeros((b, 3), device=self.device)
            )

            goal_left_pose = self.goal_left.pose.raw_pose.clone()
            goal_right_pose = self.goal_right.pose.raw_pose.clone()
            left_goal, right_goal = self._sample_goal_positions(b)
            goal_left_pose[env_idx, :3] = left_goal
            goal_right_pose[env_idx, :3] = right_goal
            self.goal_left.set_pose(Pose.create(goal_left_pose[env_idx]))
            self.goal_right.set_pose(Pose.create(goal_right_pose[env_idx]))

    def _endpoint_positions(self):
        return (
            self.chain_left_end.pose.p,
            self.chain_right_end.pose.p,
            self.chain_mid_link.pose.p,
        )

    def _before_control_step(self):
        self._saved_left_qpos = None
        self._saved_left_qvel = None
        self._saved_right_qpos = None
        self._saved_right_qvel = None

    def _after_control_step(self):
        left_grasp = self.left_agent.is_grasping(self.chain_left_end)
        right_grasp = self.right_agent.is_grasping(self.chain_right_end)

        if not self.sticky_grasp_enabled:
            return

        left_gap = self._agent_gripper_gap(self.left_agent)
        right_gap = self._agent_gripper_gap(self.right_agent)

        self.left_endpoint_sticky = torch.logical_or(self.left_endpoint_sticky, left_grasp)
        self.right_endpoint_sticky = torch.logical_or(self.right_endpoint_sticky, right_grasp)
        self.left_endpoint_sticky = torch.logical_and(
            self.left_endpoint_sticky,
            left_gap <= self.sticky_grasp_release_gap,
        )
        self.right_endpoint_sticky = torch.logical_and(
            self.right_endpoint_sticky,
            right_gap <= self.sticky_grasp_release_gap,
        )

        active = torch.logical_or(self.left_endpoint_sticky, self.right_endpoint_sticky)
        if not torch.any(active):
            return

        left_end, right_end, _, _ = self._endpoint_positions_and_invalid()
        left_tcp = self._agent_tcp_pose(self.left_agent).p
        right_tcp = self._agent_tcp_pose(self.right_agent).p
        delta = torch.zeros_like(left_end)

        if torch.any(self.left_endpoint_sticky):
            delta[self.left_endpoint_sticky] += (
                left_tcp[self.left_endpoint_sticky]
                - left_end[self.left_endpoint_sticky]
            )
        if torch.any(self.right_endpoint_sticky):
            delta[self.right_endpoint_sticky] += (
                right_tcp[self.right_endpoint_sticky]
                - right_end[self.right_endpoint_sticky]
            )

        moved = torch.linalg.norm(delta, dim=1) > 1e-8
        if not torch.any(moved):
            return

        root_pose = self.chain.pose.raw_pose.clone()
        root_pose[moved, :3] = root_pose[moved, :3] + delta[moved]
        self.chain.set_pose(Pose.create(root_pose))

    def _goal_positions(self):
        return self.goal_left.pose.p, self.goal_right.pose.p

    def _goal_line_stats(self):
        goal_left, goal_right = self._goal_positions()
        goal_left = self._sanitize_tensor(goal_left)
        goal_right = self._sanitize_tensor(goal_right)
        goal_mid = self._sanitize_tensor(0.5 * (goal_left + goal_right))
        goal_vec = goal_right - goal_left
        goal_span = torch.linalg.norm(goal_vec, axis=1)
        goal_dir = goal_vec / torch.clamp(goal_span.unsqueeze(1), min=1e-6)
        return goal_left, goal_right, goal_mid, goal_dir, goal_span

    def _chain_line_stats(self, left_end: torch.Tensor, right_end: torch.Tensor):
        chain_vec = right_end - left_end
        chain_span = torch.linalg.norm(chain_vec, axis=1)
        chain_dir = chain_vec / torch.clamp(chain_span.unsqueeze(1), min=1e-6)
        return chain_dir, chain_span

    def _sanitize_tensor(self, x, clip=None):
        if clip is None:
            clip = self.obs_sanitize_clip
        x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
        return torch.clamp(x, min=-clip, max=clip)

    def _endpoint_positions_and_invalid(self):
        left_end, right_end, mid = self._endpoint_positions()
        invalid = (
            ~torch.isfinite(left_end).all(dim=1)
            | ~torch.isfinite(right_end).all(dim=1)
            | ~torch.isfinite(mid).all(dim=1)
            | ~torch.isfinite(self.chain.get_qvel()).all(dim=1)
        )
        return (
            self._sanitize_tensor(left_end),
            self._sanitize_tensor(right_end),
            self._sanitize_tensor(mid),
            invalid,
        )

    def _agent_tcp_pose(self, agent):
        if hasattr(agent, "tcp_pose"):
            return agent.tcp_pose
        if hasattr(agent, "tcp") and hasattr(agent.tcp, "pose"):
            return agent.tcp.pose
        raise AttributeError(f"{type(agent).__name__} does not expose a TCP pose")

    def _agent_arm_qvel(self, agent):
        qvel = agent.robot.get_qvel()
        arm_joint_names = getattr(agent, "arm_joint_names", None)
        if arm_joint_names is not None:
            return qvel[..., : len(arm_joint_names)]
        return qvel

    def _agent_gripper_gap(self, agent):
        if hasattr(agent, "finger1_link") and hasattr(agent, "finger2_link"):
            return torch.linalg.norm(
                agent.finger1_link.pose.p - agent.finger2_link.pose.p, dim=1
            )
        return torch.full((self.num_envs,), 1e6, device=self.device)

    def _normalize_vec(self, v: torch.Tensor, eps: float = 1e-6):
        return v / torch.clamp(torch.linalg.norm(v, axis=1, keepdims=True), min=eps)

    def _agent_grip_axis(self, agent):
        if hasattr(agent, "finger1_link") and hasattr(agent, "finger2_link"):
            axis = agent.finger1_link.pose.p - agent.finger2_link.pose.p
            return self._normalize_vec(axis)
        tcp_pose = self._agent_tcp_pose(agent)
        rot = tcp_pose.to_transformation_matrix()[..., :3, :3]
        return self._normalize_vec(rot[..., :, 1])

    def _projected_axis_alignment(
        self,
        left_axis: torch.Tensor,
        right_axis: torch.Tensor,
        chain_dir: torch.Tensor,
    ):
        left_proj = left_axis - torch.sum(left_axis * chain_dir, dim=1, keepdims=True) * chain_dir
        right_proj = right_axis - torch.sum(right_axis * chain_dir, dim=1, keepdims=True) * chain_dir
        left_proj_norm = torch.linalg.norm(left_proj, axis=1)
        right_proj_norm = torch.linalg.norm(right_proj, axis=1)
        valid = (left_proj_norm > 1e-4) & (right_proj_norm > 1e-4)
        left_proj = self._normalize_vec(left_proj)
        right_proj = self._normalize_vec(right_proj)
        proj_align = torch.abs(torch.sum(left_proj * right_proj, dim=1))
        raw_align = torch.abs(torch.sum(left_axis * right_axis, dim=1))
        return torch.where(valid, proj_align, raw_align)

    def _finger_contact_score(self, finger_a, finger_b, chain_link):
        fa = self.scene.get_pairwise_contact_forces(finger_a, chain_link)
        fb = self.scene.get_pairwise_contact_forces(finger_b, chain_link)
        return torch.linalg.norm(fa, axis=1) + torch.linalg.norm(fb, axis=1)

    def _chain_motion_norm(self):
        lin = torch.linalg.norm(self._sanitize_tensor(self.chain_mid_link.linear_velocity), axis=1)
        ang = torch.linalg.norm(self._sanitize_tensor(self.chain_mid_link.angular_velocity), axis=1)
        return lin + 0.3 * ang

    def _best_goal_match(self):
        _, _, mid, _ = self._endpoint_positions_and_invalid()
        _, _, goal_mid, _, _ = self._goal_line_stats()
        return self._sanitize_tensor(
            torch.linalg.norm(mid - goal_mid, axis=1),
            clip=self.obs_sanitize_clip,
        )

    def evaluate(self):
        left_end, right_end, mid, invalid = self._endpoint_positions_and_invalid()
        _, _, goal_mid, goal_dir, goal_span = self._goal_line_stats()
        chain_dir, chain_span = self._chain_line_stats(left_end, right_end)
        center_dist = torch.linalg.norm(mid - goal_mid, axis=1)
        align_score = torch.abs(torch.sum(chain_dir * goal_dir, dim=1))
        span_ratio = chain_span / torch.clamp(goal_span, min=1e-6)
        placed = (
            (center_dist <= self.goal_center_thresh)
            & (align_score >= self.goal_align_cos_thresh)
            & (span_ratio >= self.goal_span_ratio_thresh)
        )
        static = self._chain_motion_norm() < 0.2
        fail = torch.logical_or(left_end[:, 2] < 0.0, right_end[:, 2] < 0.0)
        fail = torch.logical_or(fail, mid[:, 2] < 0.0)
        fail = torch.logical_or(fail, center_dist > self.fail_goal_dist_thresh)
        fail = torch.logical_or(fail, invalid)
        return {
            "success": torch.logical_and(torch.logical_and(placed, static), ~fail),
            "fail": fail,
            "is_chain_placed": placed,
            "is_chain_static": static,
            "goal_center_dist": center_dist,
            "goal_align_score": align_score,
            "goal_span_ratio": span_ratio,
        }

    def _get_obs_extra(self, info: dict):
        left_end, right_end, mid, _ = self._endpoint_positions_and_invalid()
        goal_left, goal_right, goal_mid, goal_dir, goal_span = self._goal_line_stats()
        left_tcp_pose = self._agent_tcp_pose(self.left_agent)
        right_tcp_pose = self._agent_tcp_pose(self.right_agent)
        left_grip_axis = self._sanitize_tensor(self._agent_grip_axis(self.left_agent))
        right_grip_axis = self._sanitize_tensor(self._agent_grip_axis(self.right_agent))
        obstacle_top = torch.tensor(
            [self.obstacle_xy[0], self.obstacle_xy[1], self.obstacle_half_size[2] * 2],
            device=self.device,
            dtype=left_end.dtype,
        ).repeat(self.num_envs, 1)
        obstacle_top = self._sanitize_tensor(obstacle_top)
        obs = dict(
            left_arm_tcp=self._sanitize_tensor(left_tcp_pose.raw_pose),
            right_arm_tcp=self._sanitize_tensor(right_tcp_pose.raw_pose),
            left_arm_tcp_q=self._sanitize_tensor(left_tcp_pose.q),
            right_arm_tcp_q=self._sanitize_tensor(right_tcp_pose.q),
            left_grip_axis=left_grip_axis,
            right_grip_axis=right_grip_axis,
            left_to_right_tcp=self._sanitize_tensor(right_tcp_pose.p - left_tcp_pose.p),
        )
        if "state" in self.obs_mode:
            obs.update(
                chain_left_end=left_end,
                chain_right_end=right_end,
                chain_mid_pos=mid,
                goal_left_pos=goal_left,
                goal_right_pos=goal_right,
                goal_mid_pos=goal_mid,
                goal_dir=goal_dir,
                goal_span=goal_span.unsqueeze(1),
                obstacle_top_pos=obstacle_top,
                left_tcp_to_chain_left=self._sanitize_tensor(left_end - left_tcp_pose.p),
                right_tcp_to_chain_right=self._sanitize_tensor(right_end - right_tcp_pose.p),
                chain_mid_to_goal_mid=self._sanitize_tensor(goal_mid - mid),
                chain_left_to_goal_left=self._sanitize_tensor(goal_left - left_end),
                chain_right_to_goal_right=self._sanitize_tensor(goal_right - right_end),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        left_end, right_end, mid, _ = self._endpoint_positions_and_invalid()
        goal_left, goal_right, goal_mid, goal_dir, goal_span = self._goal_line_stats()
        left_tcp_pose = self._agent_tcp_pose(self.left_agent)
        right_tcp_pose = self._agent_tcp_pose(self.right_agent)
        chain_dir, chain_span = self._chain_line_stats(left_end, right_end)
        left_grip_axis = self._agent_grip_axis(self.left_agent)
        right_grip_axis = self._agent_grip_axis(self.right_agent)

        left_reach = 1 - torch.tanh(
            4 * torch.linalg.norm(left_end - left_tcp_pose.p, axis=1)
        )
        right_reach = 1 - torch.tanh(
            4 * torch.linalg.norm(right_end - right_tcp_pose.p, axis=1)
        )
        left_xy_align = 1.0 - torch.tanh(
            8.0 * torch.linalg.norm(left_tcp_pose.p[:, :2] - left_end[:, :2], axis=1)
        )
        right_xy_align = 1.0 - torch.tanh(
            8.0 * torch.linalg.norm(right_tcp_pose.p[:, :2] - right_end[:, :2], axis=1)
        )
        left_top_target = left_end.clone()
        right_top_target = right_end.clone()
        left_top_target[:, 2] += self.chain_top_grasp_z_offset
        right_top_target[:, 2] += self.chain_top_grasp_z_offset
        left_top_target_reach = 1.0 - torch.tanh(
            10.0 * torch.linalg.norm(left_tcp_pose.p - left_top_target, axis=1)
        )
        right_top_target_reach = 1.0 - torch.tanh(
            10.0 * torch.linalg.norm(right_tcp_pose.p - right_top_target, axis=1)
        )
        left_top_approach = left_top_target_reach
        right_top_approach = right_top_target_reach
        left_grip_perp = 1.0 - torch.abs(torch.sum(left_grip_axis * chain_dir, dim=1))
        right_grip_perp = 1.0 - torch.abs(torch.sum(right_grip_axis * chain_dir, dim=1))
        left_pregrasp = left_top_approach * left_grip_perp
        right_pregrasp = right_top_approach * right_grip_perp
        top_stage_dense_reward = 0.55 * (left_top_approach + right_top_approach)
        top_stage_dense_reward += 0.45 * (
            left_grip_perp * left_top_approach + right_grip_perp * right_top_approach
        )
        top_approach_done = (
            (left_top_approach > 0.85)
            & (right_top_approach > 0.85)
            & (left_grip_perp > 0.75)
            & (right_grip_perp > 0.75)
        )
        self.top_approach_achieved = torch.logical_or(self.top_approach_achieved, top_approach_done)

        left_contact_z_align = 1.0 - torch.tanh(
            18.0
            * torch.abs(
                left_tcp_pose.p[:, 2]
                - (left_end[:, 2] + self.chain_contact_grasp_z_offset)
            )
        )
        right_contact_z_align = 1.0 - torch.tanh(
            18.0
            * torch.abs(
                right_tcp_pose.p[:, 2]
                - (right_end[:, 2] + self.chain_contact_grasp_z_offset)
            )
        )
        left_descend_grasp = left_xy_align * left_contact_z_align * left_grip_perp
        right_descend_grasp = right_xy_align * right_contact_z_align * right_grip_perp
        reward = 0.35 * (left_reach + right_reach)
        reward += 0.75 * (left_descend_grasp + right_descend_grasp)

        left_grasp = self.left_agent.is_grasping(self.chain_left_end)
        right_grasp = self.right_agent.is_grasping(self.chain_right_end)
        dual_grasp = torch.logical_and(left_grasp, right_grasp)
        single_grasp = torch.logical_xor(left_grasp, right_grasp)
        left_only = left_grasp & (~right_grasp)
        right_only = right_grasp & (~left_grasp)

        reward += 0.9 * left_grasp.float() * (0.2 + 0.8 * left_pregrasp)
        reward += 0.9 * right_grasp.float() * (0.2 + 0.8 * right_pregrasp)
        reward -= 0.45 * left_grasp.float() * (1.0 - left_pregrasp)
        reward -= 0.45 * right_grasp.float() * (1.0 - right_pregrasp)

        obstacle_clearance = self.obstacle_half_size[2] * 2 + 0.03

        lift_height = torch.minimum(torch.minimum(left_end[:, 2], right_end[:, 2]), mid[:, 2])
        lift_reward = 1 - torch.tanh(
            5 * torch.clamp(obstacle_clearance - lift_height, min=0.0)
        )
        left_tcp_lift = 1 - torch.tanh(
            5 * torch.clamp(obstacle_clearance - left_tcp_pose.p[:, 2], min=0.0)
        )
        right_tcp_lift = 1 - torch.tanh(
            5 * torch.clamp(obstacle_clearance - right_tcp_pose.p[:, 2], min=0.0)
        )
        tcp_lift_reward = torch.minimum(left_tcp_lift, right_tcp_lift)
        grip_twist_align = self._projected_axis_alignment(
            left_grip_axis,
            right_grip_axis,
            chain_dir,
        )
        balanced_lift = torch.exp(-12.0 * torch.abs(left_end[:, 2] - right_end[:, 2]))
        chain_static_reward = 1 - torch.tanh(self._chain_motion_norm())
        left_static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self._agent_arm_qvel(self.left_agent), axis=1)
        )
        right_static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self._agent_arm_qvel(self.right_agent), axis=1)
        )

        single_wait_static = torch.zeros_like(reward)
        single_wait_static[left_only] = 0.5 * (
            chain_static_reward[left_only] + left_static_reward[left_only]
        )
        single_wait_static[right_only] = 0.5 * (
            chain_static_reward[right_only] + right_static_reward[right_only]
        )
        single_catchup_reward = torch.zeros_like(reward)
        single_catchup_reward[left_only] = right_reach[left_only]
        single_catchup_reward[right_only] = left_reach[right_only]
        premature_lift = torch.clamp(lift_height - 0.03, min=0.0)
        center_dist = torch.linalg.norm(mid - goal_mid, axis=1)
        premature_move = 1 - torch.tanh(3.0 * center_dist)

        reward += 1.2 * single_grasp.float() * single_wait_static
        reward += 1.4 * single_grasp.float() * single_catchup_reward
        reward -= 1.0 * single_grasp.float() * premature_lift
        reward -= 0.6 * single_grasp.float() * premature_move

        self.dual_grasp_achieved = torch.logical_or(self.dual_grasp_achieved, dual_grasp)

        grasp_stage_dense_reward = reward + 0.4 * dual_grasp.float() * grip_twist_align
        grasp_stage_dense_reward += 0.8 * dual_grasp.float() * balanced_lift
        grasp_stage_dense_reward = torch.where(
            self.dual_grasp_achieved,
            torch.full_like(grasp_stage_dense_reward, self.dual_grasp_stage_bonus),
            grasp_stage_dense_reward,
        )
        grasp_stage_reward = torch.where(
            self.top_approach_achieved,
            torch.full_like(top_stage_dense_reward, self.top_approach_stage_bonus)
            + grasp_stage_dense_reward,
            top_stage_dense_reward,
        )

        lifted = torch.logical_and(dual_grasp, lift_height > obstacle_clearance)
        self.chain_lifted_over_wall = torch.logical_or(self.chain_lifted_over_wall, lifted)

        lift_stage_reward = (
            2.8 * lift_reward
            + 1.8 * tcp_lift_reward
            + 0.8 * grip_twist_align
            + 0.8 * balanced_lift
        )
        lift_stage_reward = torch.where(
            self.chain_lifted_over_wall,
            torch.full_like(lift_stage_reward, self.lifted_stage_bonus),
            lift_stage_reward,
        )

        move_reward = 1 - torch.tanh(3.0 * center_dist)
        align_reward = torch.abs(torch.sum(chain_dir * goal_dir, dim=1))
        span_ratio = chain_span / torch.clamp(goal_span, min=1e-6)
        span_reward = torch.clamp(span_ratio, min=0.0, max=1.0)
        move_stage_reward = (
            move_reward
            + align_reward
            + span_reward
            + 1.0 * grip_twist_align
            + 0.8 * balanced_lift
        )

        near_target = torch.logical_and(center_dist < self.goal_center_thresh * 2.0, lifted)
        settle_reward = (chain_static_reward + left_static_reward + right_static_reward) / 3
        settle_stage_reward = (
            1.5 * move_reward
            + 1.0 * align_reward
            + 1.0 * span_reward
            + 1.0 * grip_twist_align
            + 0.8 * balanced_lift
            + settle_reward
        )

        placed = info["is_chain_placed"]
        reward = grasp_stage_reward
        reward = torch.where(
            self.dual_grasp_achieved,
            self.top_approach_stage_bonus + self.dual_grasp_stage_bonus + lift_stage_reward,
            reward,
        )
        reward = torch.where(
            self.chain_lifted_over_wall,
            self.top_approach_stage_bonus
            + self.dual_grasp_stage_bonus
            + self.lifted_stage_bonus
            + move_stage_reward,
            reward,
        )
        reward = torch.where(
            near_target,
            self.top_approach_stage_bonus
            + self.dual_grasp_stage_bonus
            + self.lifted_stage_bonus
            + 5.0
            + settle_stage_reward,
            reward,
        )
        reward = torch.where(
            placed,
            torch.full_like(reward, self.placed_stage_bonus) + settle_reward,
            reward,
        )
        reward[info["success"]] = 21
        reward[info["fail"]] = -2
        return self._sanitize_tensor(reward, clip=21)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 21
