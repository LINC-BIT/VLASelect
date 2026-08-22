from typing import Any, Tuple

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import envs.agents.ur10e_panda_gripper  # noqa: F401
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("TwoRobotStackCubeUR10e-v1", max_episode_steps=100)
class TwoRobotStackCubeUR10eEnv(BaseEnv):
    """
    Copy of ManiSkill's two_robot_stack task with the right panda replaced by
    a UR10e arm equipped with the panda gripper.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "ur10e_panda_gripper")]
    agent: MultiAgent[Tuple[Panda, BaseAgent]]

    goal_radius = 0.06

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "ur10e_panda_gripper"),
        robot_init_qpos_noise=0.02,
        cube_scale=1.0,
        top_object_type="cube",
        top_object_scale=None,
        goal_radius=0.06,
        cube_x_center=0.0,
        cube_x_half_range=0.05,
        left_cube_y_center=-0.15,
        left_cube_y_half_range=0.05,
        right_cube_y_center=0.15,
        right_cube_y_half_range=0.05,
        goal_x_center=0.0,
        goal_x_half_range=0.05,
        goal_y=-0.10,
        **kwargs,
    ):
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.cube_scale = float(cube_scale)
        self.top_object_type = str(top_object_type)
        self.top_object_scale = float(cube_scale if top_object_scale is None else top_object_scale)
        self.goal_radius = float(goal_radius)
        self.cube_x_center = float(cube_x_center)
        self.cube_x_half_range = float(cube_x_half_range)
        self.left_cube_y_center = float(left_cube_y_center)
        self.left_cube_y_half_range = float(left_cube_y_half_range)
        self.right_cube_y_center = float(right_cube_y_center)
        self.right_cube_y_half_range = float(right_cube_y_half_range)
        self.goal_x_center = float(goal_x_center)
        self.goal_x_half_range = float(goal_x_half_range)
        self.goal_y = float(goal_y)
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
        pose = sapien_utils.look_at(eye=[0.3, 0.0, 0.6], target=[-0.1, 0.0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[1.1, 0.0, 0.75], target=[0.0, 0.0, 0.12])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def _left_robot_pose(self):
        return sapien.Pose([0.0, -0.75, 0.0], q=euler2quat(0, 0, np.pi / 2))

    def _right_robot_pose(self):
        return sapien.Pose([0.0, 0.75, 0.0], q=euler2quat(0, 0, -np.pi / 2))

    def _load_agent(self, options: dict):
        super()._load_agent(options, [self._left_robot_pose(), self._right_robot_pose()])

    def _load_scene(self, options: dict):
        base_cube_half_extent = 0.02
        top_object_half_extent = 0.02 * self.top_object_scale
        self.base_cube_half_size = common.to_tensor([base_cube_half_extent] * 3, device=self.device)
        self.top_object_half_extent = float(top_object_half_extent)
        if self.top_object_type == "cube":
            self.top_object_horizontal_extent = float(top_object_half_extent)
            self.top_object_vertical_extent = float(top_object_half_extent)
        elif self.top_object_type == "sphere":
            self.top_object_horizontal_extent = float(top_object_half_extent)
            self.top_object_vertical_extent = float(top_object_half_extent)
        elif self.top_object_type == "cylinder":
            self.top_object_horizontal_extent = float(top_object_half_extent)
            self.top_object_vertical_extent = float(top_object_half_extent)
        else:
            raise ValueError(f"Unsupported top_object_type={self.top_object_type}")
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        cube_a_color = np.array([12, 42, 160, 255]) / 255
        if self.top_object_type == "cube":
            self.cubeA = actors.build_cube(
                self.scene,
                half_size=top_object_half_extent,
                color=cube_a_color,
                name="cubeA",
                initial_pose=sapien.Pose(p=[1, 0, self.top_object_vertical_extent]),
            )
        elif self.top_object_type == "sphere":
            self.cubeA = actors.build_sphere(
                self.scene,
                radius=top_object_half_extent,
                color=cube_a_color,
                name="cubeA",
                initial_pose=sapien.Pose(p=[1, 0, self.top_object_vertical_extent]),
            )
        else:
            self.cubeA = actors.build_cylinder(
                self.scene,
                radius=top_object_half_extent,
                half_length=top_object_half_extent,
                color=cube_a_color,
                name="cubeA",
                initial_pose=sapien.Pose(p=[1, 0, self.top_object_vertical_extent]),
            )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=base_cube_half_extent,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[-1, 0, base_cube_half_extent]),
        )
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(),
        )

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> BaseAgent:
        return self.agent.agents[1]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            left_qpos = self.left_agent.robot.get_qpos().clone()
            right_qpos = self.right_agent.robot.get_qpos().clone()

            left_rest = torch.tensor(
                [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04],
                device=self.device,
                dtype=left_qpos.dtype,
            ).repeat(b, 1)
            right_rest = torch.as_tensor(
                [-1.95, -1.55, 2.4, -2.4, -1.5708, -2.0944, 0.04, 0.04],
                device=self.device,
                dtype=right_qpos.dtype,
            ).repeat(b, 1)
            left_rest += (torch.rand_like(left_rest) * 2 - 1) * self.robot_init_qpos_noise
            right_rest += (torch.rand_like(right_rest) * 2 - 1) * self.robot_init_qpos_noise
            left_rest[:, -2:] = 0.04
            right_rest[:, -2:] = 0.04

            left_qpos[env_idx] = left_rest
            right_qpos[env_idx] = right_rest
            self.left_agent.robot.set_qpos(left_qpos[env_idx])
            self.right_agent.robot.set_qpos(right_qpos[env_idx])
            self.left_agent.robot.set_pose(self._left_robot_pose())
            self.right_agent.robot.set_pose(self._right_robot_pose())

            cubeA_xyz = torch.zeros((b, 3), device=self.device)
            cubeA_xyz[:, 0] = (
                self.cube_x_center
                + (torch.rand((b,), device=self.device) * 2 - 1) * self.cube_x_half_range
            )
            cubeA_xyz[:, 1] = (
                self.left_cube_y_center
                + (torch.rand((b,), device=self.device) * 2 - 1) * self.left_cube_y_half_range
            )
            cubeA_xyz[:, 2] = self.top_object_vertical_extent

            cubeB_xyz = torch.zeros((b, 3), device=self.device)
            cubeB_xyz[:, 0] = (
                self.cube_x_center
                + (torch.rand((b,), device=self.device) * 2 - 1) * self.cube_x_half_range
            )
            cubeB_xyz[:, 1] = (
                self.right_cube_y_center
                + (torch.rand((b,), device=self.device) * 2 - 1) * self.right_cube_y_half_range
            )
            cubeB_xyz[:, 2] = self.base_cube_half_size[2]

            cubeA_q = random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            cubeB_q = random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=cubeA_xyz, q=cubeA_q))
            self.cubeB.set_pose(Pose.create_from_pq(p=cubeB_xyz, q=cubeB_q))

            target_region_xyz = torch.zeros((b, 3), device=self.device)
            target_region_xyz[:, 0] = (
                self.goal_x_center
                + (torch.rand((b,), device=self.device) * 2 - 1) * self.goal_x_half_range
            )
            target_region_xyz[:, 1] = self.goal_y
            target_region_xyz[:, 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.base_cube_half_size[:2]) + 0.005
        )
        expected_stack_height = float(self.base_cube_half_size[2]) + self.top_object_vertical_extent
        z_flag = torch.abs(offset[..., 2] - expected_stack_height) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        cubeB_to_goal_dist = torch.linalg.norm(
            self.cubeB.pose.p[:, :2] - self.goal_region.pose.p[..., :2],
            axis=1,
        )
        cubeB_placed = cubeB_to_goal_dist < self.goal_radius
        is_cubeA_grasped = self.left_agent.is_grasping(self.cubeA)
        is_cubeB_grasped = self.right_agent.is_grasping(self.cubeB)
        success = is_cubeA_on_cubeB & cubeB_placed & (~is_cubeA_grasped) & (~is_cubeB_grasped)
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeB_grasped": is_cubeB_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "cubeB_placed": cubeB_placed,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            stage = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
            stage[info["is_cubeA_grasped"] | info["is_cubeB_grasped"]] = 1
            stage[info["cubeB_placed"] & info["is_cubeA_grasped"]] = 2
            stage[info["is_cubeA_on_cubeB"] & info["cubeB_placed"]] = 3
            obs.update(
                goal_region_pos=self.goal_region.pose.p,
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                left_arm_tcp_to_cubeA_pos=self.cubeA.pose.p - self.left_agent.tcp.pose.p,
                right_arm_tcp_to_cubeB_pos=self.cubeB.pose.p - self.right_agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                stage=stage,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        cubeA_to_left_arm_tcp_dist = torch.linalg.norm(
            self.left_agent.tcp.pose.p - self.cubeA.pose.p,
            axis=1,
        )
        left_gripper_width = self.left_agent.robot.get_qlimits()[0, -2:, 1].sum().to(self.device)
        right_gripper_width = self.right_agent.robot.get_qlimits()[0, -2:, 1].sum().to(self.device)
        left_open_ratio = torch.sum(self.left_agent.robot.get_qpos()[:, -2:], axis=1) / left_gripper_width
        right_open_ratio = torch.sum(self.right_agent.robot.get_qpos()[:, -2:], axis=1) / right_gripper_width

        right_arm_push_pose = Pose.create_from_pq(
            p=self.cubeB.pose.p
            + torch.tensor([0, self.base_cube_half_size[0] + 0.005, 0], device=self.device)
        )
        right_arm_to_push_pose_dist = torch.linalg.norm(
            right_arm_push_pose.p - self.right_agent.tcp.pose.p,
            axis=1,
        )
        reach_reward = (
            (1 - torch.tanh(5 * cubeA_to_left_arm_tcp_dist))
            + (1 - torch.tanh(5 * right_arm_to_push_pose_dist))
        ) / 2

        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        reward = (reach_reward + info["is_cubeA_grasped"].float()) / 2

        place_stage_reached = info["is_cubeA_grasped"]
        cubeB_to_goal_dist = torch.linalg.norm(
            cubeB_pos[:, :2] - self.goal_region.pose.p[..., :2],
            axis=1,
        )
        place_reward = 1 - torch.tanh(5 * cubeB_to_goal_dist)
        stage_2_reward = place_reward + info["is_cubeA_grasped"].float()
        reward[place_stage_reached] = 2 + stage_2_reward[place_stage_reached] / 2

        cubeB_placed_and_cubeA_grasped = info["cubeB_placed"] & info["is_cubeA_grasped"]
        stack_target_z = cubeB_pos[:, 2] + float(self.base_cube_half_size[2]) + self.top_object_vertical_extent
        goal_xyz = torch.hstack(
            [cubeB_pos[:, :2], stack_target_z[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        cubeA_xy_to_goal_dist = torch.linalg.norm(goal_xyz[:, :2] - cubeA_pos[:, :2], axis=1)
        cubeA_z_to_goal_dist = torch.abs(goal_xyz[:, 2] - cubeA_pos[:, 2])
        top_place_reward = 1 - torch.tanh(5 * cubeA_to_goal_dist)
        right_arm_leave_reward = 1 - torch.tanh(
            5 * (self.right_agent.tcp.pose.p[:, 1] - 0.2).abs()
        )
        left_arm_leave_reward = 1 - torch.tanh(
            5 * (self.left_agent.tcp.pose.p[:, 1] + 0.2).abs()
        )
        right_release_reward = right_open_ratio.clone()
        right_release_reward[~info["is_cubeB_grasped"]] = 1.0
        stage_3_reward = top_place_reward * 2 + right_arm_leave_reward + right_release_reward
        reward[cubeB_placed_and_cubeA_grasped] = 4 + stage_3_reward[cubeB_placed_and_cubeA_grasped]

        # Once cubeA is nearly aligned over cubeB, start shaping Panda's release
        # before the strict stack condition is met so the policy does not learn to
        # keep squeezing until the very last instant.
        left_release_ready = (
            cubeB_placed_and_cubeA_grasped
            & (cubeA_xy_to_goal_dist < 0.04)
            & (cubeA_z_to_goal_dist < 0.015)
        )
        left_release_reward = left_open_ratio.clone()
        left_release_stage_reward = (
            top_place_reward * 2
            + right_arm_leave_reward
            + right_release_reward
            + 0.5 * left_release_reward
            + 0.5 * left_arm_leave_reward
        )
        reward[left_release_ready] = 4.5 + left_release_stage_reward[left_release_ready]

        cubes_placed = info["is_cubeA_on_cubeB"] & info["cubeB_placed"]
        ungrasp_reward_left = left_open_ratio.clone()
        ungrasp_reward_right = right_open_ratio.clone()
        ungrasp_reward_left[~info["is_cubeA_grasped"]] = 1.0
        ungrasp_reward_right[~info["is_cubeB_grasped"]] = 1.0
        stable_release_reward = (
            0.5 * ungrasp_reward_left
            + 0.5 * ungrasp_reward_right
            + 0.25 * left_arm_leave_reward
            + 0.25 * right_arm_leave_reward
        )
        reward[cubes_placed] = 8 + (2 * stable_release_reward)[cubes_placed]

        reward[info["success"]] = 10
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10
