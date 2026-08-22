import os
from typing import Any

import numpy as np
import sapien
import torch

os.environ.setdefault("MS_ASSET_DIR", "/home/Maniskill/.maniskill")

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


def _yaw_quat(theta: float):
    return [float(np.cos(theta / 2.0)), 0.0, 0.0, float(np.sin(theta / 2.0))]


@register_env("BallPickup-v1", max_episode_steps=100)
class BallPickupEnv(BaseEnv):
    """
    CoEnv DualFrankaGraspBall inspired task in ManiSkill style.

    A left Panda and a right robot jointly grasp the central ball, then lift it
    above the table to a target height. By default the right robot is xArm6.
    """

    SUPPORTED_ROBOTS = [
        ("panda_wristcam", "panda_wristcam"),
        ("panda_wristcam", "xarm6_robotiq"),
    ]
    SUPPORTED_REWARD_MODES = ["none", "normalized_dense"]
    agent: MultiAgent

    ball_radius = 0.11
    ball_mass = 0.1
    lift_success_delta = 0.12
    success_reward = 20.0
    static_ball_speed_thresh = 0.2
    wrist_side_offset = 0.18
    prealign_z_offset = 0.10
    side_descend_z_offset = 0.04
    facing_done_threshold = 0.82
    prealign_done_threshold = 0.65
    side_done_threshold = 0.72
    stage0_base = 3.5
    stage1_base = 3.2
    stage2_base = 4.0

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "xarm6_robotiq"),
        robot_init_qpos_noise=0.015,
        robot0_pose=(0.41, 0.0, np.pi / 2),
        robot1_pose=(-0.41, 0.0, np.pi / 2),
        ball_xy=(0.0, 0.53),
        ball_xy_noise=(0.03, 0.04),
        **kwargs,
    ):
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.robot0_pose_cfg = tuple(float(v) for v in robot0_pose)
        self.robot1_pose_cfg = tuple(float(v) for v in robot1_pose)
        self.ball_xy = tuple(float(v) for v in ball_xy)
        self.ball_xy_noise = tuple(float(v) for v in ball_xy_noise)
        self.ball = None
        self.cam_mount = None
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
        pose = sapien_utils.look_at([0.0, 1.12, 0.46], [0.0, 0.36, 0.12])
        return [
            CameraConfig(
                "base_camera",
                pose,
                128,
                128,
                np.pi / 2,
                0.01,
                100,
                mount=self.cam_mount,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.0, 1.18, 0.48], [0.0, 0.36, 0.12])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _robot_pose(self, cfg):
        return sapien.Pose(p=[cfg[0], cfg[1], 0.0], q=_yaw_quat(cfg[2]))

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            [
                self._robot_pose(self.robot0_pose_cfg),
                self._robot_pose(self.robot1_pose_cfg),
            ],
        )

    def _load_scene(self, options: dict):
        builder_mount = self.scene.create_actor_builder()
        builder_mount.set_initial_pose(sapien.Pose())
        self.cam_mount = builder_mount.build_kinematic("camera_mount")

        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=0.0)
        self.table_scene.build()

        self.ball = actors.build_sphere(
            self.scene,
            radius=self.ball_radius,
            color=[0.94, 0.30, 0.18, 1.0],
            name="pickup_ball",
            initial_pose=sapien.Pose(p=[0.0, 0.53, self.ball_radius]),
        )
        self.ball.mass = self.ball_mass

    @property
    def robot0(self):
        return self.agent.agents[0]

    @property
    def robot1(self):
        return self.agent.agents[1]

    def _rest_qpos_for_agent(self, agent, qpos_template: torch.Tensor):
        if "rest" in agent.keyframes:
            key = "rest"
        else:
            key = next(iter(agent.keyframes.keys()))
        base_qpos = np.asarray(agent.keyframes[key].qpos)
        if base_qpos.ndim > 1:
            base_qpos = base_qpos[0]
        return torch.as_tensor(base_qpos, device=self.device, dtype=qpos_template.dtype)

    def _is_grasping(self, agent):
        if hasattr(agent, "is_grasping"):
            return agent.is_grasping(self.ball)
        return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

    def _reach_reward(self, dist: torch.Tensor, scale: float):
        return 1 - torch.tanh(scale * dist)

    def _wrist_inward_axes(self):
        rot0 = quaternion_to_matrix(self.robot0.tcp.pose.q)
        rot1 = quaternion_to_matrix(self.robot1.tcp.pose.q)
        # The image-space "wrist facing inward" corresponds to left wrist -Y and right wrist +Y.
        inward0 = -rot0[:, :, 1]
        inward1 = rot1[:, :, 1]
        return inward0, inward1

    def _stage_metrics(self):
        ball_pos = self.ball.pose.p
        tcp0 = self.robot0.tcp.pose.p
        tcp1 = self.robot1.tcp.pose.p
        grasp0 = self._is_grasping(self.robot0)
        grasp1 = self._is_grasping(self.robot1)
        dual_grasp = torch.logical_and(grasp0, grasp1)

        align_target0 = ball_pos + torch.tensor(
            [self.wrist_side_offset, 0.0, self.prealign_z_offset],
            device=self.device,
            dtype=ball_pos.dtype,
        )
        align_target1 = ball_pos + torch.tensor(
            [-self.wrist_side_offset, 0.0, self.prealign_z_offset],
            device=self.device,
            dtype=ball_pos.dtype,
        )
        side_target0 = ball_pos + torch.tensor(
            [self.wrist_side_offset, 0.0, self.side_descend_z_offset],
            device=self.device,
            dtype=ball_pos.dtype,
        )
        side_target1 = ball_pos + torch.tensor(
            [-self.wrist_side_offset, 0.0, self.side_descend_z_offset],
            device=self.device,
            dtype=ball_pos.dtype,
        )

        inward0, inward1 = self._wrist_inward_axes()
        to_ball0 = ball_pos - tcp0
        to_ball1 = ball_pos - tcp1
        to_ball0 = to_ball0 / torch.clamp(torch.linalg.norm(to_ball0, axis=1, keepdim=True), min=1e-6)
        to_ball1 = to_ball1 / torch.clamp(torch.linalg.norm(to_ball1, axis=1, keepdim=True), min=1e-6)

        facing0 = torch.clamp((inward0 * to_ball0).sum(dim=1), min=0.0, max=1.0)
        facing1 = torch.clamp((inward1 * to_ball1).sum(dim=1), min=0.0, max=1.0)
        facing_mean = 0.5 * (facing0 + facing1)

        prealign0 = self._reach_reward(torch.linalg.norm(align_target0 - tcp0, axis=1), 5.0)
        prealign1 = self._reach_reward(torch.linalg.norm(align_target1 - tcp1, axis=1), 5.0)
        prealign_mean = 0.5 * (prealign0 + prealign1)

        side0 = self._reach_reward(torch.linalg.norm(side_target0 - tcp0, axis=1), 7.0)
        side1 = self._reach_reward(torch.linalg.norm(side_target1 - tcp1, axis=1), 7.0)
        side_mean = 0.5 * (side0 + side1)

        squeeze0 = self._reach_reward(torch.linalg.norm(ball_pos - tcp0, axis=1), 5.0)
        squeeze1 = self._reach_reward(torch.linalg.norm(ball_pos - tcp1, axis=1), 5.0)
        squeeze_mean = 0.5 * (squeeze0 + squeeze1)

        facing_done = torch.logical_and(
            facing_mean >= self.facing_done_threshold,
            prealign_mean >= self.prealign_done_threshold,
        )
        side_done = torch.logical_and(facing_done, side_mean >= self.side_done_threshold)

        return {
            "grasp0": grasp0,
            "grasp1": grasp1,
            "dual_grasp": dual_grasp,
            "facing_mean": facing_mean,
            "prealign_mean": prealign_mean,
            "side_mean": side_mean,
            "squeeze_mean": squeeze_mean,
            "facing_done": facing_done,
            "side_done": side_done,
        }

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            qpos0 = self.robot0.robot.get_qpos().clone()
            qpos1 = self.robot1.robot.get_qpos().clone()
            base_qpos0 = self._rest_qpos_for_agent(self.robot0, qpos0).unsqueeze(0).repeat(b, 1)
            base_qpos1 = self._rest_qpos_for_agent(self.robot1, qpos1).unsqueeze(0).repeat(b, 1)
            noise0 = (
                (torch.rand((b, base_qpos0.shape[1]), device=self.device) * 2 - 1)
                * self.robot_init_qpos_noise
            )
            noise1 = (
                (torch.rand((b, base_qpos1.shape[1]), device=self.device) * 2 - 1)
                * self.robot_init_qpos_noise
            )
            qpos0[env_idx] = base_qpos0 + noise0
            qpos1[env_idx] = base_qpos1 + noise1
            self.robot0.robot.set_qpos(qpos0[env_idx])
            self.robot1.robot.set_qpos(qpos1[env_idx])

            ball_xy = torch.zeros((b, 2), device=self.device)
            ball_xy[:, 0] = self.ball_xy[0] + (
                torch.rand((b,), device=self.device) * 2 - 1
            ) * self.ball_xy_noise[0]
            ball_xy[:, 1] = self.ball_xy[1] + (
                torch.rand((b,), device=self.device) * 2 - 1
            ) * self.ball_xy_noise[1]
            ball_xyz = torch.zeros((b, 3), device=self.device)
            ball_xyz[:, :2] = ball_xy
            ball_xyz[:, 2] = self.ball_radius
            self.ball.set_pose(Pose.create_from_pq(ball_xyz))

            cam_pose = sapien_utils.look_at([0.0, 1.12, 0.46], [0.0, 0.36, 0.12])
            cam_pose_p = torch.as_tensor(
                cam_pose.p,
                device=self.device,
                dtype=torch.float32,
            ).reshape(1, 3).repeat(b, 1)
            cam_pose_q = torch.as_tensor(
                cam_pose.q,
                device=self.device,
                dtype=torch.float32,
            ).reshape(1, 4).repeat(b, 1)
            self.cam_mount.set_pose(Pose.create_from_pq(cam_pose_p, cam_pose_q))

    def evaluate(self):
        stage_metrics = self._stage_metrics()
        dual_grasp = stage_metrics["dual_grasp"]
        lift_height = self.ball.pose.p[:, 2] - self.ball_radius
        success = torch.logical_and(dual_grasp, lift_height >= self.lift_success_delta)
        fail = self.ball.pose.p[:, 2] < -0.01
        return {
            "success": success,
            "fail": fail,
            "is_dual_grasp": dual_grasp,
            "lift_height": lift_height,
            "ball_speed": torch.linalg.norm(self.ball.get_linear_velocity(), axis=1),
            "facing_done": stage_metrics["facing_done"],
            "side_done": stage_metrics["side_done"],
        }

    def _stage_tensor(self):
        stage_metrics = self._stage_metrics()
        grasp_any = torch.logical_or(stage_metrics["grasp0"], stage_metrics["grasp1"])
        dual_grasp = stage_metrics["dual_grasp"]
        lift_height = self.ball.pose.p[:, 2] - self.ball_radius
        lifted_mid = lift_height > (self.lift_success_delta * 0.5)
        lifted_high = lift_height >= self.lift_success_delta

        stage = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        stage[stage_metrics["facing_done"]] = 1.0
        stage[stage_metrics["side_done"]] = 2.0
        stage[torch.logical_and(stage_metrics["side_done"], grasp_any)] = 3.0
        stage[dual_grasp] = 4.0
        stage[torch.logical_and(dual_grasp, lifted_mid)] = 5.0
        stage[torch.logical_and(dual_grasp, lifted_high)] = 6.0
        return stage

    def _get_obs_extra(self, info: dict):
        obs = dict(
            robot0_tcp=self.robot0.tcp.pose.raw_pose,
            robot1_tcp=self.robot1.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                ball_pose=self.ball.pose.raw_pose,
                robot0_tcp_to_ball_pos=self.ball.pose.p - self.robot0.tcp.pose.p,
                robot1_tcp_to_ball_pos=self.ball.pose.p - self.robot1.tcp.pose.p,
                stage=self._stage_tensor(),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        ball_speed = info["ball_speed"]
        lift_height = info["lift_height"]
        lift_progress = torch.clamp(lift_height / self.lift_success_delta, 0.0, 1.0)
        stage_metrics = self._stage_metrics()
        grasp_any = torch.logical_or(stage_metrics["grasp0"], stage_metrics["grasp1"])
        dual_grasp = stage_metrics["dual_grasp"]
        success = info["success"]
        fail = info["fail"]

        static_reward = 1 - torch.tanh(4 * ball_speed)
        stage0_dense = 1.6 * stage_metrics["facing_mean"] + 1.4 * stage_metrics["prealign_mean"]
        stage1_dense = 2.8 * stage_metrics["side_mean"]
        single_grasp_count = stage_metrics["grasp0"].float() + stage_metrics["grasp1"].float()
        stage2_dense = 2.6 * stage_metrics["squeeze_mean"] + 0.8 * single_grasp_count
        stage3_dense = 3.8 * lift_progress + 0.8 * static_reward

        stage0_active = ~stage_metrics["facing_done"]
        stage1_active = torch.logical_and(stage_metrics["facing_done"], ~stage_metrics["side_done"])
        stage2_active = torch.logical_and(stage_metrics["side_done"], ~dual_grasp)
        stage3_active = torch.logical_and(dual_grasp, ~success)

        reward = torch.zeros_like(stage0_dense)
        reward[stage0_active] = stage0_dense[stage0_active]
        reward[stage1_active] = self.stage0_base + stage1_dense[stage1_active]
        reward[stage2_active] = self.stage0_base + self.stage1_base + stage2_dense[stage2_active]
        reward[stage3_active] = (
            self.stage0_base
            + self.stage1_base
            + self.stage2_base
            + stage3_dense[stage3_active]
        )

        reward[success] = self.success_reward
        reward[fail] -= 2.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.success_reward
