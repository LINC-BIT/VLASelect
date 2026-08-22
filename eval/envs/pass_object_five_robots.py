import os
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

os.environ.setdefault("MS_ASSET_DIR", "/home/Maniskill/.maniskill")

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


def _yaw_quat(theta: float):
    return [float(np.cos(theta / 2.0)), 0.0, 0.0, float(np.sin(theta / 2.0))]


@register_env("PassObjectFiveRobots-v1", max_episode_steps=300)
class PassObjectFiveRobotsEnv(BaseEnv):
    agent_agent_collision_disable_bit = 28
    xarm6_cube_collision_disable_bit = 27
    hand_manual_delta_limit = 0.1
    SUPPORTED_ROBOTS = [
        (
            "panda_wristcam",
            "so100",
            "widowxai",
            "xarm6_robotiq",
            "fixed_inspire_hand_right",
        )
    ]
    agent: MultiAgent
    SUPPORTED_REWARD_MODES = ["none", "normalized_dense"]

    stage1_base_reward = 20.0
    stage2_base_reward = 42.0
    stage3_base_reward = 64.0
    stage4_base_reward = 86.0
    success_reward = 100.0
    FAIL_REASON_NONE = 0
    FAIL_REASON_DROP = 1
    FAIL_REASON_PUSH_TIMEOUT = 2
    FAIL_REASON_NONFINITE = 3
    push_offset = 0.055
    push_side_reach_radius = 0.02
    push_side_reach_bonus = 1.2
    push_tcp_z_offset = 0.008
    push_retract_radius = 0.18
    cube_half_size = 0.022
    cube_mass = 0.1
    cube_density = cube_mass / ((2.0 * cube_half_size) ** 3)
    handoff_lift_z = 0.14
    handoff_above_z = cube_half_size + 0.02
    handoff_above_radius = 0.055
    palm_above_z_offset = 0.05
    palm_above_radius = 0.055
    pregrasp_width_margin = 0.012
    handoff_place_radius = 0.15
    handoff_place_z_tol = 0.04
    handoff_place_static_thresh = 0.45
    handoff_place_face_cos = 0.85
    handoff_goal_miss_margin = 0.04
    handoff_goal_lateral_miss_start = 0.16
    handoff_goal_lateral_miss_margin = 0.12
    palm_target_local_offset = np.array([-0.015, 0.085, 0.022], dtype=np.float32)
    palm_success_radius = 0.055
    stage4_spawn_above_palm_z = 0.02
    stage4_spawn_palm_z_clearance = 0.04
    stage4_xarm_open_gripper_qpos = np.array([0.30, 0.30, 0.30, 0.30, -0.30, -0.30], dtype=np.float32)
    cube_static_thresh = 0.18
    xarm6_release_contact_thresh = 0.05
    table_goal_marker_radius = handoff_place_radius
    palm_goal_marker_radius = palm_success_radius
    above_marker_radius = 0.025
    inactive_pose_reward_coef = 0.35
    inactive_qpos_reward_coef = 0.15
    curriculum_inactive_pose_reward_coef = 0.90
    curriculum_inactive_qpos_reward_coef = 0.40
    curriculum_inactive_action_still_coef = 0.45
    so100_init_qpos = np.array([0.0, 0.0, 0.0, 0.18, 0.0, -1.1], dtype=np.float32)
    table_goal_stall_speed_thresh = 0.03
    table_goal_stall_moved_thresh = 0.05
    table_goal_stall_patience = 30
    table_goal_progress_eps = 0.005
    push_corridor_reward_coef = 0.45
    push_contact_force_thresh = 0.05
    push_contact_deadline_steps = 60
    auto_release_joint5_target = 1.80
    auto_release_joint6_target = 0.56
    auto_release_hold_steps = 8
    auto_release_trigger_radius = 0.03
    manual_stage4_hand_drive = False
    stage4_hand_wrist_action_scale = 0.05
    stage4_hand_finger_action_scale = 0.15

    def __init__(
        self,
        *args,
        robot_uids=(
            "panda_wristcam",
            "so100",
            "widowxai",
            "xarm6_robotiq",
            "fixed_inspire_hand_right",
        ),
        robot_init_qpos_noise=0.02,
        panda_xy=(-1.06, -0.81),
        so100_xy=(-0.77, -0.42),
        widowx_xy=(-0.66, -0.02),
        xarm6_xy=(-0.98, 0.29),
        hand_pose=(-0.37, 0.74, 0.03),
        table_yaw=0.0,
        so100_yaw=np.pi / 2,
        cube_xy=(-0.45, -0.82),
        cube_half_size=None,
        cube_mass=None,
        handoff_xy_01=(-0.45, -0.44),
        handoff_xy_12=(-0.45, -0.04),
        handoff_xy_23=(-0.45, 0.36),
        stage_mode="full",
        disable_agent_agent_collisions=True,
        disable_xarm6_cube_collisions=False,
        auto_stage3_rotate_release=False,
        **kwargs,
    ):
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.table_yaw = float(table_yaw)
        self.so100_yaw = float(so100_yaw)
        self.panda_xy = tuple(float(v) for v in panda_xy)
        self.so100_xy = tuple(float(v) for v in so100_xy)
        self.widowx_xy = tuple(float(v) for v in widowx_xy)
        self.xarm6_xy = tuple(float(v) for v in xarm6_xy)
        self.hand_pose = tuple(float(v) for v in hand_pose)
        self.cube_xy = tuple(float(v) for v in cube_xy)
        self.cube_half_size = (
            float(cube_half_size)
            if cube_half_size is not None
            else float(type(self).cube_half_size)
        )
        base_density = float(type(self).cube_density)
        cube_volume = (2.0 * self.cube_half_size) ** 3
        if cube_mass is None:
            self.cube_density = base_density
            self.cube_mass = self.cube_density * cube_volume
        else:
            self.cube_mass = float(cube_mass)
            self.cube_density = self.cube_mass / cube_volume
        self.handoff_above_z = self.cube_half_size + 0.02
        self.handoff_xy_01 = tuple(float(v) for v in handoff_xy_01)
        self.handoff_xy_12 = tuple(float(v) for v in handoff_xy_12)
        self.handoff_xy_23 = tuple(float(v) for v in handoff_xy_23)
        self.stage_mode = self._normalize_stage_mode(stage_mode)
        self.disable_agent_agent_collisions = bool(disable_agent_agent_collisions)
        self.disable_xarm6_cube_collisions = bool(disable_xarm6_cube_collisions)
        self.auto_stage3_rotate_release = bool(auto_stage3_rotate_release)
        self.curriculum_stage = None if self.stage_mode == "full" else int(self.stage_mode)
        self.stage_success_threshold = 5 if self.curriculum_stage is None else (self.curriculum_stage + 1)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @staticmethod
    def _normalize_stage_mode(stage_mode):
        if stage_mode is None:
            return "full"
        if isinstance(stage_mode, str):
            mode = stage_mode.strip().lower()
            if mode in ("", "full", "all"):
                return "full"
        else:
            mode = str(int(stage_mode))
        try:
            stage_id = int(mode)
        except ValueError as exc:
            raise ValueError(f"Unsupported stage_mode={stage_mode!r}") from exc
        if stage_id < 0 or stage_id > 4:
            raise ValueError(f"stage_mode must be one of 0,1,2,3,4,full, got {stage_mode!r}")
        return str(stage_id)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**21,
                max_rigid_contact_count=2**23,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.58, -0.02, 1.02], [-0.45, -0.02, 0.14])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 3, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.72, -0.02, 1.04], [-0.45, -0.02, 0.14])
        return CameraConfig("render_camera", pose, 512, 512, np.pi / 3, 0.01, 100)

    @property
    def panda_agent(self):
        return self.agent.agents[0]

    @property
    def so100_agent(self):
        return self.agent.agents[1]

    @property
    def widowx_agent(self):
        return self.agent.agents[2]

    @property
    def xarm6_agent(self):
        return self.agent.agents[3]

    @property
    def hand_agent(self):
        return self.agent.agents[4]

    def _arm_pose(self, xy):
        return sapien.Pose(p=[xy[0], xy[1], 0.0], q=_yaw_quat(self.table_yaw))

    def _so100_pose(self):
        return sapien.Pose(
            p=[self.so100_xy[0], self.so100_xy[1], 0.0],
            q=_yaw_quat(self.table_yaw + self.so100_yaw),
        )

    def _hand_pose(self):
        return sapien.Pose(
            p=list(self.hand_pose),
            q=euler2quat(np.pi / 2, 0.0, -np.pi / 2),
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            [
                self._arm_pose(self.panda_xy),
                self._so100_pose(),
                self._arm_pose(self.widowx_xy),
                self._arm_pose(self.xarm6_xy),
                self._hand_pose(),
            ],
        )
        if self.disable_agent_agent_collisions:
            self._disable_agent_agent_collisions()
        if self.disable_xarm6_cube_collisions:
            self._set_xarm6_cube_collisions_disabled(True)

    def _mask_multiagent_action_by_stage(self, action: dict[str, torch.Tensor]):
        if self.agent is None or not isinstance(self.agent, MultiAgent):
            return action
        agent_names = list(self.agent.agents_dict.keys())
        active_idx = torch.clamp(self.current_stage, min=0, max=4)
        masked_action = {}
        for agent_index, agent_name in enumerate(agent_names):
            value = action[agent_name]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
            if value.ndim == 1:
                masked_action[agent_name] = (
                    value
                    if int(active_idx[0].item()) == agent_index
                    else torch.zeros_like(value)
                )
                continue
            row_mask = (active_idx == agent_index).to(
                device=value.device, dtype=value.dtype
            ).view(-1, 1)
            masked_action[agent_name] = value * row_mask
        return masked_action

    def _step_action(self, action):
        if (
            isinstance(action, dict)
            and "control_mode" not in action
        ):
            action = common.to_tensor(action, device=self.device)
            hand_name = next(
                (
                    name
                    for name in self.agent.agents_dict.keys()
                    if "inspire" in name or name.startswith("fixed_inspire_hand")
                ),
                None,
            )
            if (
                (not self.manual_stage4_hand_drive)
                and hand_name is not None
                and hand_name in action
            ):
                hand_action = action[hand_name]
                scaled_hand_action = hand_action.clone()
                if scaled_hand_action.shape[-1] >= 2:
                    scaled_hand_action[..., :2] = (
                        scaled_hand_action[..., :2] * self.stage4_hand_wrist_action_scale
                    )
                if scaled_hand_action.shape[-1] > 2:
                    scaled_hand_action[..., 2:] = (
                        scaled_hand_action[..., 2:] * self.stage4_hand_finger_action_scale
                    )
                if hand_action.ndim == 1:
                    if int(torch.clamp(self.current_stage[0], min=0, max=4).item()) >= 4:
                        action[hand_name] = scaled_hand_action
                else:
                    stage4_mask = (self.current_stage >= 4).to(
                        device=hand_action.device, dtype=hand_action.dtype
                    ).view(-1, 1)
                    action[hand_name] = hand_action * (
                        1.0 - stage4_mask
                    )
                    action[hand_name] = action[hand_name] + scaled_hand_action * stage4_mask
            if self.stage_mode == "full":
                action = self._mask_multiagent_action_by_stage(action)
            if self.manual_stage4_hand_drive and hand_name is not None and hand_name in action:
                self._pending_hand_action = action[hand_name].clone()
        return super()._step_action(action)

    def _disable_agent_agent_collisions(self):
        # Put every robot link into a shared collision-filter group bit so
        # robot-robot contacts are ignored while robot-cube/table contacts stay on.
        for agent in self.agent.agents:
            for link in agent.robot.links:
                link.set_collision_group_bit(
                    group=2,
                    bit_idx=self.agent_agent_collision_disable_bit,
                    bit=1,
                )

    def _set_xarm6_cube_collisions_disabled(self, disabled: bool):
        bit = 1 if disabled else 0
        for link in self.xarm6_agent.robot.links:
            link.set_collision_group_bit(
                group=2,
                bit_idx=self.xarm6_cube_collision_disable_bit,
                bit=bit,
            )
        self.cube.set_collision_group_bit(
            group=2,
            bit_idx=self.xarm6_cube_collision_disable_bit,
            bit=bit,
        )

    def _build_cube(self):
        builder = self.scene.create_actor_builder()
        builder.set_initial_pose(
            sapien.Pose(p=[self.cube_xy[0], self.cube_xy[1], self.cube_half_size])
        )
        builder.add_box_visual(
            half_size=[self.cube_half_size] * 3,
            material=sapien.render.RenderMaterial(base_color=[0.92, 0.28, 0.2, 1.0]),
        )
        builder.add_box_collision(
            half_size=[self.cube_half_size] * 3,
            density=self.cube_density,
        )
        return builder.build(name="pass_cube")

    def _build_waypoint_marker(self, name: str, radius: float, color):
        return actors.build_sphere(
            self.scene,
            radius=radius,
            color=color,
            body_type="kinematic",
            add_collision=False,
            name=name,
            initial_pose=sapien.Pose(),
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=0.0)
        self.table_scene.build()
        self.cube = self._build_cube()
        self.handoff_target_01_marker = self._build_waypoint_marker(
            "handoff_target_01_marker", self.table_goal_marker_radius, [0.1, 0.9, 0.1, 0.55]
        )
        self.handoff_target_12_marker = self._build_waypoint_marker(
            "handoff_target_12_marker", self.table_goal_marker_radius, [0.1, 0.6, 1.0, 0.55]
        )
        self.handoff_target_23_marker = self._build_waypoint_marker(
            "handoff_target_23_marker", self.table_goal_marker_radius, [1.0, 0.7, 0.1, 0.55]
        )
        self.handoff_above_01_marker = self._build_waypoint_marker(
            "handoff_above_01_marker", self.above_marker_radius, [0.1, 0.9, 0.1, 0.95]
        )
        self.handoff_above_12_marker = self._build_waypoint_marker(
            "handoff_above_12_marker", self.above_marker_radius, [0.1, 0.6, 1.0, 0.95]
        )
        self.handoff_above_23_marker = self._build_waypoint_marker(
            "handoff_above_23_marker", self.above_marker_radius, [1.0, 0.7, 0.1, 0.95]
        )
        self.palm_target_marker = self._build_waypoint_marker(
            "palm_target_marker", self.palm_goal_marker_radius, [1.0, 0.1, 0.9, 0.65]
        )
        self.release_target_marker = self._build_waypoint_marker(
            "release_target_marker", self.palm_goal_marker_radius, [1.0, 0.2, 0.2, 0.75]
        )
        self.palm_above_marker = self._build_waypoint_marker(
            "palm_above_marker", self.above_marker_radius, [1.0, 0.1, 0.9, 0.95]
        )
        self.so100_tcp_marker = self._build_waypoint_marker(
            "so100_tcp_marker", 0.018, [1.0, 0.2, 0.2, 0.95]
        )
        self.current_stage = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._best_table_goal_dist = torch.full(
            (self.num_envs,), float("inf"), device=self.device, dtype=torch.float32
        )
        self._table_goal_stall_steps = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self._table_goal_stall_stage = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._push_contact_started = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self._push_contact_steps = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self._push_contact_stage = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._push_side_reached = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self._push_side_stage = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._auto_release_active = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self._auto_release_steps = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self._inactive_ref_qpos = None
        self._inactive_ref_tcp = None
        self._pending_hand_action = torch.zeros(
            (self.num_envs, 8), device=self.device, dtype=torch.float32
        )

    def _sample_robot_rest_qpos(self, agent):
        qpos = agent.robot.get_qpos().clone()
        agent_uid = getattr(agent, "uid", "")
        if agent_uid == "fixed_inspire_hand_right":
            key = "palm_up"
        elif agent_uid == "so100":
            base_qpos = self.so100_init_qpos
        elif "rest" in agent.keyframes:
            key = "rest"
        else:
            key = next(iter(agent.keyframes.keys()))
        if agent_uid != "so100":
            base_qpos = np.asarray(agent.keyframes[key].qpos)
            if base_qpos.ndim > 1:
                base_qpos = base_qpos[0]
        base_qpos = torch.as_tensor(base_qpos, device=self.device, dtype=qpos.dtype)
        return qpos, base_qpos

    def _curriculum_stage_start_xy(self, stage_id: int):
        if stage_id <= 0:
            return self.cube_xy
        if stage_id == 1:
            return self.handoff_xy_01
        if stage_id == 2:
            return self.handoff_xy_12
        if stage_id == 3:
            return self.handoff_xy_23
        raise ValueError(f"Unsupported curriculum stage for table start: {stage_id}")

    def _curriculum_stage_agent(self, stage_id: int):
        if stage_id <= 0:
            return self.panda_agent
        if stage_id == 1:
            return self.so100_agent
        if stage_id == 2:
            return self.widowx_agent
        if stage_id == 3:
            return self.xarm6_agent
        raise ValueError(f"Unsupported curriculum stage for agent lookup: {stage_id}")

    def _sample_xy_in_radius(self, center_xy, batch_size: int, radius: float):
        center_xy = torch.as_tensor(center_xy, device=self.device, dtype=torch.float32)
        theta = 2.0 * torch.pi * torch.rand((batch_size,), device=self.device)
        radial = float(radius) * torch.sqrt(torch.rand((batch_size,), device=self.device))
        offsets = torch.stack([radial * torch.cos(theta), radial * torch.sin(theta)], dim=1)
        return center_xy.unsqueeze(0) + offsets

    def _sample_xy_in_front_half_disk(self, center_xy, env_idx: torch.Tensor, agent, radius: float):
        batch_size = len(env_idx)
        center_xy = torch.as_tensor(center_xy, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(
            batch_size, 1
        )
        agent_xy = self._agent_tcp_pose(agent).p[env_idx, :2]
        front_dir = agent_xy - center_xy
        front_dir = front_dir / torch.linalg.norm(front_dir, dim=1, keepdim=True).clamp_min(1e-6)
        lateral_dir = torch.stack([-front_dir[:, 1], front_dir[:, 0]], dim=1)
        theta = (torch.rand((batch_size,), device=self.device) - 0.5) * torch.pi
        radial = float(radius) * torch.sqrt(torch.rand((batch_size,), device=self.device))
        offsets = (
            radial.unsqueeze(1) * torch.cos(theta).unsqueeze(1) * front_dir
            + radial.unsqueeze(1) * torch.sin(theta).unsqueeze(1) * lateral_dir
        )
        return center_xy + offsets

    def _curriculum_success_reward(self):
        if self.curriculum_stage is None:
            return self.success_reward
        next_stage_base_rewards = {
            0: self.stage1_base_reward,
            1: self.stage2_base_reward,
            2: self.stage3_base_reward,
            3: self.stage4_base_reward,
            4: self.success_reward,
        }
        return float(next_stage_base_rewards[self.curriculum_stage])

    def _sample_stage4_cube_positions(self, env_idx: torch.Tensor):
        _, _, _, palm_target = self._handoff_targets()
        palm_start = palm_target[env_idx].clone()
        palm_start[:, 2] += self.stage4_spawn_palm_z_clearance
        return palm_start

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            start_stage = 0 if self.curriculum_stage is None else self.curriculum_stage
            self.table_scene.initialize(env_idx)
            self.current_stage[env_idx] = start_stage
            self._best_table_goal_dist[env_idx] = float("inf")
            self._table_goal_stall_steps[env_idx] = 0
            self._table_goal_stall_stage[env_idx] = start_stage
            self._push_contact_started[env_idx] = False
            self._push_contact_steps[env_idx] = 0
            self._push_contact_stage[env_idx] = start_stage
            self._push_side_reached[env_idx] = False
            self._push_side_stage[env_idx] = start_stage
            self._auto_release_active[env_idx] = False
            self._auto_release_steps[env_idx] = 0

            for agent, pose in [
                (self.panda_agent, self._arm_pose(self.panda_xy)),
                (self.so100_agent, self._so100_pose()),
                (self.widowx_agent, self._arm_pose(self.widowx_xy)),
                (self.xarm6_agent, self._arm_pose(self.xarm6_xy)),
                (self.hand_agent, self._hand_pose()),
            ]:
                qpos, base_qpos = self._sample_robot_rest_qpos(agent)
                rest = base_qpos.unsqueeze(0).repeat(b, 1)
                apply_noise = self.robot_init_qpos_noise
                if start_stage == 4 and (
                    agent is self.xarm6_agent or agent is self.hand_agent
                ):
                    apply_noise = 0.0
                rest += (torch.rand_like(rest) * 2 - 1) * apply_noise
                qpos[env_idx] = rest
                agent_pose = pose
                agent.robot.set_qpos(qpos[env_idx])
                agent.robot.set_pose(agent_pose)

            cube_pose = self.cube.pose.raw_pose.clone()
            if start_stage <= 3:
                start_xy = self._curriculum_stage_start_xy(start_stage)
                if start_stage == 0:
                    sampled_xy = self._sample_xy_in_radius(start_xy, b, 0.015)
                else:
                    sampled_xy = self._sample_xy_in_front_half_disk(
                        start_xy,
                        env_idx,
                        self._curriculum_stage_agent(start_stage),
                        self.handoff_place_radius,
                    )
                cube_pose[env_idx, 0:2] = sampled_xy
                cube_pose[env_idx, 2] = self.cube_half_size
            else:
                cube_pose[env_idx, :3] = self._sample_stage4_cube_positions(env_idx)
            cube_pose[env_idx, 3] = 1.0
            cube_pose[env_idx, 4:] = 0.0
            self.cube.set_pose(Pose.create(cube_pose[env_idx]))
            self.cube.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.cube.set_angular_velocity(torch.zeros((b, 3), device=self.device))
            self._sync_waypoint_markers(env_idx)
            self._sync_tcp_markers(env_idx)
            self._sync_inactive_references(env_idx)

    def _sync_inactive_references(self, env_idx: torch.Tensor):
        if self._inactive_ref_qpos is None:
            self._inactive_ref_qpos = [
                agent.robot.get_qpos().clone() for agent in self.agent.agents
            ]
            self._inactive_ref_tcp = [
                self._agent_tcp_pose(agent).p.clone() for agent in self.agent.agents
            ]
        for i, agent in enumerate(self.agent.agents):
            self._inactive_ref_qpos[i][env_idx] = agent.robot.get_qpos()[env_idx]
            self._inactive_ref_tcp[i][env_idx] = self._agent_tcp_pose(agent).p[env_idx]

    def _set_marker_pose(self, marker, xyz: torch.Tensor, env_idx: torch.Tensor | None = None):
        marker_pose = marker.pose.raw_pose.clone()
        if env_idx is None:
            marker_pose[:, :3] = xyz
            marker_pose[:, 3] = 1.0
            marker_pose[:, 4:] = 0.0
            marker.set_pose(Pose.create(marker_pose))
        else:
            marker_pose[env_idx, :3] = xyz[env_idx]
            marker_pose[env_idx, 3] = 1.0
            marker_pose[env_idx, 4:] = 0.0
            prev_reset_mask = self._set_partial_reset_mask(env_idx)
            marker.set_pose(Pose.create(marker_pose[env_idx]))
            self._restore_reset_mask(prev_reset_mask)

    def _sync_waypoint_markers(self, env_idx: torch.Tensor | None = None):
        target01, target12, target23, palm_target = self._handoff_targets()
        table_z = torch.full((self.num_envs, 1), self.cube_half_size, device=self.device)
        above_z = torch.full((self.num_envs, 1), self.handoff_above_z, device=self.device)
        target01_xyz = torch.cat([target01, table_z], dim=1)
        target12_xyz = torch.cat([target12, table_z], dim=1)
        target23_xyz = torch.cat([target23, table_z], dim=1)
        above01_xyz = torch.cat([target01, above_z], dim=1)
        above12_xyz = torch.cat([target12, above_z], dim=1)
        above23_xyz = torch.cat([target23, above_z], dim=1)
        palm_above = palm_target.clone()
        palm_above[:, 2] += self.palm_above_z_offset
        release_target = palm_above.clone()
        hidden_palm_target = palm_target.clone()
        hidden_palm_target[:, 2] = -10.0
        hidden_palm_above = palm_above.clone()
        hidden_palm_above[:, 2] = -10.0

        for marker, xyz in [
            (self.handoff_target_01_marker, target01_xyz),
            (self.handoff_target_12_marker, target12_xyz),
            (self.handoff_target_23_marker, target23_xyz),
            (self.handoff_above_01_marker, above01_xyz),
            (self.handoff_above_12_marker, above12_xyz),
            (self.handoff_above_23_marker, above23_xyz),
            (self.palm_target_marker, hidden_palm_target),
            (self.release_target_marker, release_target),
            (self.palm_above_marker, hidden_palm_above),
        ]:
            self._set_marker_pose(marker, xyz, env_idx)

    def _sync_tcp_markers(self, env_idx: torch.Tensor | None = None):
        so100_tcp = self._agent_tcp_pose(self.so100_agent).p
        self._set_marker_pose(self.so100_tcp_marker, so100_tcp, env_idx)

    def _stage4_xarm_home_qpos(self, batch_size: int, dtype: torch.dtype):
        base_qpos = self._sample_robot_rest_qpos(self.xarm6_agent)[1]
        qpos = base_qpos.unsqueeze(0).repeat(batch_size, 1).to(dtype=dtype)
        qpos[:, 6:12] = torch.as_tensor(
            self.stage4_xarm_open_gripper_qpos,
            device=self.device,
            dtype=dtype,
        ).unsqueeze(0).repeat(batch_size, 1)
        return qpos

    def _reset_agent_controller_for_envs(self, agent, env_idx: torch.Tensor):
        prev_reset_mask = self.scene._reset_mask.clone()
        self.scene._reset_mask = torch.zeros_like(prev_reset_mask)
        self.scene._reset_mask[env_idx] = True
        agent.controller.reset()
        self.scene._reset_mask = prev_reset_mask

    def _set_partial_reset_mask(self, env_idx: torch.Tensor):
        prev_reset_mask = self.scene._reset_mask.clone()
        self.scene._reset_mask = torch.zeros_like(prev_reset_mask)
        self.scene._reset_mask[env_idx] = True
        return prev_reset_mask

    def _restore_reset_mask(self, prev_reset_mask: torch.Tensor):
        self.scene._reset_mask = prev_reset_mask

    def _gpu_apply_articulation_state_only(self):
        if not self.gpu_sim_enabled:
            return
        self.scene.px.gpu_apply_articulation_qpos()
        self.scene.px.gpu_apply_articulation_qvel()
        self.scene.px.gpu_apply_articulation_qf()
        self.scene.px.gpu_apply_articulation_root_pose()
        self.scene.px.gpu_apply_articulation_root_velocity()
        self.scene.px.gpu_update_articulation_kinematics()
        self.scene._gpu_fetch_all()

    def _hand_control_joint_indices(self):
        wrist_ctrl = self.hand_agent.controller.controllers["wrist"]
        fingers_ctrl = self.hand_agent.controller.controllers["fingers"]
        return torch.cat(
            [wrist_ctrl.active_joint_indices, fingers_ctrl.active_joint_indices], dim=0
        )

    def _apply_manual_hand_drive_on_gpu(self):
        if not self.manual_stage4_hand_drive:
            return
        if not self.gpu_sim_enabled:
            return
        if self._pending_hand_action is None:
            return
        active_mask = self.current_stage >= 4
        if not bool(active_mask.any()):
            return
        env_idx = torch.nonzero(active_mask, as_tuple=False).reshape(-1)
        robot = self.hand_agent.robot
        ctrl_idx = self._hand_control_joint_indices()
        qpos = robot.get_qpos().clone()
        qvel = robot.get_qvel().clone()
        qf = robot.get_qf().clone() if hasattr(robot, "get_qf") else None
        delta = torch.clamp(self._pending_hand_action[env_idx], min=-1.0, max=1.0)
        delta = delta * self.hand_manual_delta_limit
        next_qpos = qpos[env_idx].clone()
        next_qpos[:, ctrl_idx] = next_qpos[:, ctrl_idx] + delta
        qlimits = robot.get_qlimits()[0, ctrl_idx]
        lower = qlimits[:, 0].unsqueeze(0)
        upper = qlimits[:, 1].unsqueeze(0)
        next_qpos[:, ctrl_idx] = torch.clamp(next_qpos[:, ctrl_idx], min=lower, max=upper)
        qpos[env_idx] = next_qpos
        qvel[env_idx] = 0.0
        prev_reset_mask = self._set_partial_reset_mask(env_idx)
        robot.set_qpos(qpos[env_idx])
        robot.set_qvel(qvel[env_idx])
        if qf is not None and hasattr(robot, "set_qf"):
            qf[env_idx] = 0.0
            robot.set_qf(qf[env_idx])
        self._restore_reset_mask(prev_reset_mask)
        self._gpu_apply_articulation_state_only()
        self._reset_agent_controller_for_envs(self.hand_agent, env_idx)

    def _enforce_stage4_xarm_home(self, env_idx: torch.Tensor):
        if len(env_idx) == 0:
            return
        robot = self.xarm6_agent.robot
        qpos = self.xarm6_agent.robot.get_qpos().clone()
        qvel = self.xarm6_agent.robot.get_qvel().clone()
        qf = robot.get_qf().clone() if hasattr(robot, "get_qf") else None
        qpos[env_idx] = self._stage4_xarm_home_qpos(len(env_idx), qpos.dtype)
        qvel[env_idx] = 0.0
        prev_reset_mask = self._set_partial_reset_mask(env_idx)
        robot.set_qpos(qpos[env_idx])
        robot.set_qvel(qvel[env_idx])
        if qf is not None and hasattr(robot, "set_qf"):
            qf[env_idx] = 0.0
            robot.set_qf(qf[env_idx])
        self._restore_reset_mask(prev_reset_mask)
        if self.gpu_sim_enabled:
            self._gpu_apply_articulation_state_only()
        self._reset_agent_controller_for_envs(self.xarm6_agent, env_idx)

    def _place_stage4_cube(self, env_idx: torch.Tensor):
        if len(env_idx) == 0:
            return
        b = len(env_idx)
        cube_pose = self.cube.pose.raw_pose.clone()
        cube_pose[env_idx, :3] = self._sample_stage4_cube_positions(env_idx)
        cube_pose[env_idx, 3] = 1.0
        cube_pose[env_idx, 4:] = 0.0
        prev_reset_mask = self._set_partial_reset_mask(env_idx)
        self.cube.set_pose(Pose.create(cube_pose[env_idx]))
        zeros = torch.zeros((b, 3), device=self.device)
        self.cube.set_linear_velocity(zeros)
        self.cube.set_angular_velocity(zeros)
        self._restore_reset_mask(prev_reset_mask)
        if self.gpu_sim_enabled:
            self.scene.px.gpu_apply_rigid_dynamic_data()
            self.scene.px.gpu_fetch_rigid_dynamic_data()

    def _place_cube_at_table_goal(
        self, env_idx: torch.Tensor, target_xy: torch.Tensor, yaw: float = 0.0
    ):
        if len(env_idx) == 0:
            return
        b = len(env_idx)
        cube_pose = self.cube.pose.raw_pose.clone()
        cube_pose[env_idx, 0:2] = target_xy[env_idx]
        cube_pose[env_idx, 2] = self.cube_half_size
        cube_pose[env_idx, 3:7] = torch.as_tensor(
            _yaw_quat(float(yaw)), device=self.device, dtype=cube_pose.dtype
        ).unsqueeze(0).repeat(b, 1)
        prev_reset_mask = self._set_partial_reset_mask(env_idx)
        self.cube.set_pose(Pose.create(cube_pose[env_idx]))
        zeros = torch.zeros((b, 3), device=self.device)
        self.cube.set_linear_velocity(zeros)
        self.cube.set_angular_velocity(zeros)
        self._restore_reset_mask(prev_reset_mask)
        if self.gpu_sim_enabled:
            self.scene.px.gpu_apply_rigid_dynamic_data()
            self.scene.px.gpu_fetch_rigid_dynamic_data()

    def _force_stage4_setup(self, env_idx: torch.Tensor):
        if len(env_idx) == 0:
            return
        self.current_stage[env_idx] = 4
        self._enforce_stage4_xarm_home(env_idx)
        self._place_stage4_cube(env_idx)
        self._sync_waypoint_markers(env_idx)
        self._sync_tcp_markers(env_idx)
        self._sync_inactive_references(env_idx)

    def _after_control_step(self):
        self._maybe_apply_stage3_auto_release()
        self._sync_waypoint_markers()
        self._sync_tcp_markers()

    def _before_control_step(self):
        if self.manual_stage4_hand_drive:
            self._apply_manual_hand_drive_on_gpu()

    def _maybe_apply_stage3_auto_release(self):
        if not self.auto_stage3_rotate_release:
            return
        cube_p = self._cube_pose().p
        _, _, _, palm_target = self._handoff_targets()
        palm_release_target = palm_target.clone()
        palm_release_target[:, 2] += self.palm_above_z_offset
        release_dist = torch.linalg.norm(cube_p - palm_release_target, dim=1)
        xarm_grasp = self._generic_is_grasping(self.xarm6_agent)
        stage3 = self.current_stage == 3
        trigger = stage3 & xarm_grasp & (release_dist <= self.auto_release_trigger_radius)
        self._auto_release_active = torch.where(
            trigger,
            torch.ones_like(self._auto_release_active),
            self._auto_release_active & stage3,
        )
        self._auto_release_steps = torch.where(
            self._auto_release_active,
            self._auto_release_steps + 1,
            torch.zeros_like(self._auto_release_steps),
        )
        if not bool(self._auto_release_active.any()):
            return

        active_idx = torch.nonzero(self._auto_release_active, as_tuple=False).reshape(-1)
        qpos = self.xarm6_agent.robot.get_qpos().clone()
        qvel = self.xarm6_agent.robot.get_qvel().clone()
        qpos[active_idx, 4] = self.auto_release_joint5_target
        qpos[active_idx, 5] = self.auto_release_joint6_target

        release_now = active_idx[
            self._auto_release_steps[active_idx] >= self.auto_release_hold_steps
        ]
        if len(release_now) > 0:
            qpos[release_now, 6:12] = 0.0

        qvel[active_idx, :] = 0.0
        self.xarm6_agent.robot.set_qpos(qpos)
        self.xarm6_agent.robot.set_qvel(qvel)

        released_done = release_now if len(release_now) > 0 else active_idx[:0]
        if len(released_done) > 0:
            self._auto_release_active[released_done] = False
            self._auto_release_steps[released_done] = 0

    def _agent_tcp_pose(self, agent):
        if getattr(agent, "uid", "") == "fixed_inspire_hand_right":
            return self._hand_palm_pose()
        if hasattr(agent, "tcp_pose"):
            return agent.tcp_pose
        if hasattr(agent, "tcp") and hasattr(agent.tcp, "pose"):
            return agent.tcp.pose
        raise AttributeError(f"{type(agent).__name__} does not expose a TCP pose")

    def _robot_link(self, agent, link_name: str):
        robot = agent.robot
        if hasattr(robot, "links_map") and link_name in robot.links_map:
            return robot.links_map[link_name]
        for link in robot.links:
            if link.name == link_name:
                return link
        raise KeyError(f"Link {link_name} not found on {type(robot).__name__}")

    def _hand_palm_pose(self):
        return self._robot_link(self.hand_agent, "right_hand_hand_base_link").pose

    def _hand_tip_links(self):
        return [
            self._robot_link(self.hand_agent, "right_hand_thumb_tip"),
            self._robot_link(self.hand_agent, "right_hand_index_tip"),
            self._robot_link(self.hand_agent, "right_hand_middle_tip"),
            self._robot_link(self.hand_agent, "right_hand_ring_tip"),
            self._robot_link(self.hand_agent, "right_hand_pinky_tip"),
        ]

    def _cube_pose(self):
        return self.cube.pose

    def _cube_speed(self):
        lin = torch.linalg.norm(self.cube.linear_velocity, dim=1)
        ang = torch.linalg.norm(self.cube.angular_velocity, dim=1)
        return lin + 0.3 * ang

    def _handoff_targets(self):
        target01 = torch.tensor(
            self.handoff_xy_01, device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.num_envs, 1)
        target12 = torch.tensor(
            self.handoff_xy_12, device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.num_envs, 1)
        target23 = torch.tensor(
            self.handoff_xy_23, device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.num_envs, 1)
        palm_pose = self._hand_palm_pose()
        palm_rot = palm_pose.to_transformation_matrix()[..., :3, :3]
        palm_target_offset = torch.as_tensor(
            self.palm_target_local_offset, device=self.device, dtype=torch.float32
        ).view(1, 3, 1).expand(self.num_envs, -1, -1)
        palm_target = palm_pose.p + torch.bmm(
            palm_rot, palm_target_offset
        ).squeeze(-1)
        return target01, target12, target23, palm_target

    def _cube_near_table_target(self, target_xy: torch.Tensor, strict: bool = True):
        cube_pose = self._cube_pose()
        cube_p = cube_pose.p
        xy_dist = torch.linalg.norm(cube_p[:, :2] - target_xy, dim=1)
        z_ok = torch.abs(cube_p[:, 2] - self.cube_half_size) <= self.handoff_place_z_tol
        if strict:
            rot = cube_pose.to_transformation_matrix()[..., :3, :3]
            face_flat = torch.max(torch.abs(rot[:, 2, :]), dim=1).values >= self.handoff_place_face_cos
            static_ok = self._cube_speed() <= self.handoff_place_static_thresh
            placed = (xy_dist <= self.handoff_place_radius) & z_ok & face_flat & static_ok
        else:
            placed = (xy_dist <= self.handoff_place_radius) & z_ok
        return xy_dist, placed

    def _cube_table_target_progress(self, target_xy: torch.Tensor):
        cube_pose = self._cube_pose()
        cube_p = cube_pose.p
        xy_dist = torch.linalg.norm(cube_p[:, :2] - target_xy, dim=1)
        near_target = xy_dist <= self.handoff_place_radius
        z_progress = 1.0 - torch.tanh(12.0 * torch.abs(cube_p[:, 2] - self.cube_half_size))
        rot = cube_pose.to_transformation_matrix()[..., :3, :3]
        face_flat_score = torch.max(torch.abs(rot[:, 2, :]), dim=1).values
        face_flat_progress = torch.clamp(
            (face_flat_score - 0.55) / max(1e-6, 1.0 - 0.55),
            min=0.0,
            max=1.0,
        )
        speed = self._cube_speed()
        static_progress = 1.0 - torch.tanh(5.0 * speed)
        return near_target, z_progress, face_flat_progress, static_progress

    def _generic_is_grasping(self, agent):
        if hasattr(agent, "is_grasping"):
            return agent.is_grasping(self.cube)
        return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

    def _finger_pregrasp_reward(self, agent):
        if not (hasattr(agent, "finger1_link") and hasattr(agent, "finger2_link")):
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        finger1 = agent.finger1_link.pose.p
        finger2 = agent.finger2_link.pose.p
        tip_height_reward = 1.0 - torch.tanh(5.0 * torch.abs(finger1[:, 2] - finger2[:, 2]))
        target_width = 2.0 * self.cube_half_size + self.pregrasp_width_margin
        tip_width = torch.linalg.norm(finger1 - finger2, dim=1)
        tip_width_reward = 1.0 - torch.tanh(5.0 * torch.abs(tip_width - target_width))
        return 0.5 * (tip_height_reward + tip_width_reward)

    def _finger_midpoint(self, agent):
        if not (hasattr(agent, "finger1_link") and hasattr(agent, "finger2_link")):
            return self._agent_tcp_pose(agent).p
        return 0.5 * (agent.finger1_link.pose.p + agent.finger2_link.pose.p)

    def _cube_between_fingers(self, agent):
        if not (hasattr(agent, "finger1_link") and hasattr(agent, "finger2_link")):
            zero = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
            return zero, torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        cube_p = self._cube_pose().p
        finger1 = agent.finger1_link.pose.p
        finger2 = agent.finger2_link.pose.p
        finger_vec = finger2 - finger1
        finger_width = torch.linalg.norm(finger_vec, dim=1).clamp_min(1e-6)
        finger_axis = finger_vec / finger_width.unsqueeze(1)
        cube_from_finger1 = cube_p - finger1
        cube_axis_pos = torch.sum(cube_from_finger1 * finger_axis, dim=1)
        cube_axis_pos_clamped = torch.clamp(cube_axis_pos, min=0.0)
        cube_axis_pos_clamped = torch.minimum(cube_axis_pos_clamped, finger_width)
        closest_on_finger_line = finger1 + cube_axis_pos_clamped.unsqueeze(1) * finger_axis
        line_dist = torch.linalg.norm(cube_p - closest_on_finger_line, dim=1)
        axis_margin = 0.35 * self.cube_half_size
        between_axis = (cube_axis_pos > axis_margin) & (cube_axis_pos < finger_width - axis_margin)
        between_reward = (1.0 - torch.tanh(12.0 * line_dist)) * between_axis.float()
        between = between_axis & (line_dist < 0.04)
        return between_reward, between

    def _agent_action_tensor(self, action: Any, agent_index: int):
        if action is None:
            return None
        if isinstance(action, dict):
            agent_keys = list(getattr(self.agent, "agents_dict", {}).keys())
            if agent_index < len(agent_keys):
                value = action.get(agent_keys[agent_index])
                if value is not None:
                    return torch.as_tensor(value, device=self.device, dtype=torch.float32)
            return None
        if not torch.is_tensor(action):
            return None
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action_dims = []
        for agent in self.agent.agents:
            space = getattr(agent, "single_action_space", None)
            if space is None:
                return None
            action_dims.append(int(space.shape[0]))
        if action.shape[-1] == sum(action_dims):
            start = sum(action_dims[:agent_index])
            end = start + action_dims[agent_index]
            return action[..., start:end]
        if agent_index == 0 and action.shape[-1] == action_dims[0]:
            return action
        return None

    def _agent_gripper_action(self, action: Any, agent_index: int):
        agent_action = self._agent_action_tensor(action, agent_index)
        if agent_action is None or agent_action.shape[-1] == 0:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        gripper_action = agent_action[..., -1].to(self.device, dtype=torch.float32).reshape(-1)
        if gripper_action.shape[0] == 1 and self.num_envs != 1:
            gripper_action = gripper_action.expand(self.num_envs)
        return gripper_action

    def _hand_finger_close_reward(self):
        qpos = self.hand_agent.robot.get_qpos()
        active_fingers = qpos[:, 2:8]
        close_amount = torch.mean(torch.clamp(-active_fingers, min=0.0), dim=1)
        return torch.tanh(2.5 * close_amount)

    def _hand_contact_reward(self):
        palm = self._robot_link(self.hand_agent, "right_hand_hand_base_link")
        palm_force = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(palm, self.cube), dim=1
        )
        tip_forces = []
        for link in self._hand_tip_links():
            tip_forces.append(
                torch.linalg.norm(
                    self.scene.get_pairwise_contact_forces(link, self.cube), dim=1
                )
            )
        tip_forces = torch.stack(tip_forces, dim=1)
        top2_tip_force = torch.topk(
            tip_forces, k=min(2, tip_forces.shape[1]), dim=1
        ).values.sum(dim=1)
        return 1.0 - torch.exp(-(palm_force + 1.0 * top2_tip_force))

    def _hand_holding_cube(self):
        _, _, _, palm_target = self._handoff_targets()
        cube_p = self._cube_pose().p
        palm_dist = torch.linalg.norm(cube_p - palm_target, dim=1)
        close_reward = self._hand_finger_close_reward()
        contact_reward = self._hand_contact_reward()
        static_ok = self._cube_speed() < self.cube_static_thresh
        return (
            (palm_dist <= self.palm_success_radius)
            & (close_reward > 0.18)
            & (contact_reward > 0.12)
            & static_ok
        )

    def _agent_cube_contact_force(self, agent):
        total_force = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        for link in agent.robot.links:
            total_force += torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(link, self.cube), dim=1
            )
        return total_force

    def _finite_env_mask(self):
        cube_pose = self._cube_pose()
        finite = torch.isfinite(cube_pose.p).all(dim=1)
        finite = finite & torch.isfinite(cube_pose.q).all(dim=1)
        finite = finite & torch.isfinite(self.cube.linear_velocity).all(dim=1)
        finite = finite & torch.isfinite(self.cube.angular_velocity).all(dim=1)
        for agent in self.agent.agents:
            finite = finite & torch.isfinite(agent.robot.get_qpos()).all(dim=1)
            finite = finite & torch.isfinite(agent.robot.get_qvel()).all(dim=1)
            finite = finite & torch.isfinite(self._agent_tcp_pose(agent).p).all(dim=1)
        return finite

    def _update_stage_progress(self):
        old_stage = self.current_stage
        stage = old_stage.clone()
        target01, target12, target23, palm_target = self._handoff_targets()
        _, placed01 = self._cube_near_table_target(target01, strict=False)
        _, placed12 = self._cube_near_table_target(target12, strict=False)
        _, placed23 = self._cube_near_table_target(target23, strict=False)
        cube_p = self._cube_pose().p
        palm_release_target = palm_target.clone()
        palm_release_target[:, 2] += self.palm_above_z_offset
        cube_in_release_zone = (
            torch.linalg.norm(cube_p - palm_release_target, dim=1) <= self.palm_above_radius
        )
        hand_hold = self._hand_holding_cube()

        stage = torch.where((stage == 0) & placed01, 1, stage)
        stage = torch.where((stage == 1) & placed12, 2, stage)
        stage = torch.where((stage == 2) & placed23, 3, stage)
        allow_stage4_transition = self.curriculum_stage is None
        stage3_to_4 = (stage == 3) & cube_in_release_zone & allow_stage4_transition
        stage = torch.where(stage3_to_4, 4, stage)
        stage = torch.where((stage == 4) & hand_hold, 5, stage)
        self.current_stage = stage
        changed = stage != old_stage
        entered_stage1 = changed & (stage == 1)
        entered_stage2 = changed & (stage == 2)
        entered_stage3 = changed & (stage == 3)
        entered_stage4 = changed & (stage == 4)
        if bool(entered_stage1.any()):
            self._place_cube_at_table_goal(
                torch.nonzero(entered_stage1, as_tuple=False).reshape(-1), target01
            )
        if bool(entered_stage2.any()):
            self._place_cube_at_table_goal(
                torch.nonzero(entered_stage2, as_tuple=False).reshape(-1), target12
            )
        if bool(entered_stage3.any()):
            self._place_cube_at_table_goal(
                torch.nonzero(entered_stage3, as_tuple=False).reshape(-1), target23
            )
        if bool(entered_stage4.any()):
            self._force_stage4_setup(torch.nonzero(entered_stage4, as_tuple=False).reshape(-1))
        if bool(changed.any()):
            self._best_table_goal_dist[changed] = float("inf")
            self._table_goal_stall_steps[changed] = 0
            self._table_goal_stall_stage[changed] = stage[changed]
            self._push_contact_started[changed] = False
            self._push_contact_steps[changed] = 0
            self._push_contact_stage[changed] = stage[changed]
            self._push_side_reached[changed] = False
            self._push_side_stage[changed] = stage[changed]

    def _agent_touching_cube(self, agent):
        total_force = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        for link in agent.robot.links:
            total_force += torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(link, self.cube), dim=1
            )
        return total_force > self.push_contact_force_thresh

    def _current_table_goal_status(self):
        stage = self.current_stage
        cube_xy = self._cube_pose().p[:, :2]
        target01, target12, target23, _ = self._handoff_targets()
        stage_targets = [target01, target12, target23]
        stage_starts = [
            torch.as_tensor(self.cube_xy, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(self.num_envs, 1),
            target01,
            target12,
        ]

        table_stage = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        xy_dist = torch.full((self.num_envs,), float("inf"), device=self.device, dtype=torch.float32)
        placed = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        along_path = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        for stage_id, (start_xy, target_xy) in enumerate(zip(stage_starts, stage_targets)):
            mask = stage == stage_id
            target_dist, target_placed = self._cube_near_table_target(target_xy)
            direction = target_xy - start_xy
            path_len = torch.linalg.norm(direction, dim=1).clamp_min(1e-6)
            direction = direction / path_len.unsqueeze(1)
            stage_along_path = torch.sum((cube_xy - start_xy) * direction, dim=1)
            table_stage = table_stage | mask
            xy_dist = torch.where(mask, target_dist, xy_dist)
            placed = torch.where(mask, target_placed, placed)
            along_path = torch.where(mask, stage_along_path, along_path)
        return table_stage, xy_dist, placed, along_path

    def _table_goal_stalled(self):
        table_stage, xy_dist, placed, along_path = self._current_table_goal_status()
        stage_changed = self._table_goal_stall_stage != self.current_stage
        if bool(stage_changed.any()):
            self._best_table_goal_dist[stage_changed] = float("inf")
            self._table_goal_stall_steps[stage_changed] = 0
            self._table_goal_stall_stage[stage_changed] = self.current_stage[stage_changed]

        improved = xy_dist < (self._best_table_goal_dist - self.table_goal_progress_eps)
        self._best_table_goal_dist = torch.where(improved, xy_dist, self._best_table_goal_dist)

        stall_candidate = (
            table_stage
            & (~placed)
            & (xy_dist > self.handoff_place_radius)
            & (along_path > self.table_goal_stall_moved_thresh)
            & (self._cube_speed() < self.table_goal_stall_speed_thresh)
        )
        no_progress = stall_candidate & (~improved)
        self._table_goal_stall_steps = torch.where(
            no_progress,
            self._table_goal_stall_steps + 1,
            torch.zeros_like(self._table_goal_stall_steps),
        )
        return no_progress & (self._table_goal_stall_steps >= self.table_goal_stall_patience)

    def _push_contact_deadline_failed(self):
        table_stage, _, placed, _ = self._current_table_goal_status()
        stage_changed = self._push_contact_stage != self.current_stage
        if bool(stage_changed.any()):
            self._push_contact_started[stage_changed] = False
            self._push_contact_steps[stage_changed] = 0
            self._push_contact_stage[stage_changed] = self.current_stage[stage_changed]

        touch0 = self._agent_touching_cube(self.panda_agent)
        touch1 = self._agent_touching_cube(self.so100_agent)
        touch2 = self._agent_touching_cube(self.widowx_agent)
        touched = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        touched = torch.where(self.current_stage == 0, touch0, touched)
        touched = torch.where(self.current_stage == 1, touch1, touched)
        touched = torch.where(self.current_stage == 2, touch2, touched)

        active = table_stage & (~placed)
        self._push_contact_started = torch.where(
            active,
            self._push_contact_started | touched,
            torch.zeros_like(self._push_contact_started),
        )
        self._push_contact_steps = torch.where(
            active & self._push_contact_started,
            self._push_contact_steps + 1,
            torch.zeros_like(self._push_contact_steps),
        )
        return active & self._push_contact_started & (
            self._push_contact_steps >= self.push_contact_deadline_steps
        )

    def _missed_current_table_goal(self):
        stage = self.current_stage
        cube_xy = self._cube_pose().p[:, :2]
        target01, target12, target23, _ = self._handoff_targets()
        stage_targets = [target01, target12, target23]
        stage_starts = [
            torch.as_tensor(self.cube_xy, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(self.num_envs, 1),
            target01,
            target12,
        ]

        missed = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        for stage_id, (start_xy, target_xy) in enumerate(zip(stage_starts, stage_targets)):
            xy_dist, placed = self._cube_near_table_target(target_xy)
            direction = target_xy - start_xy
            path_len = torch.linalg.norm(direction, dim=1).clamp_min(1e-6)
            direction = direction / path_len.unsqueeze(1)
            start_to_cube = cube_xy - start_xy
            along_path = torch.sum(start_to_cube * direction, dim=1)
            lateral_vec = start_to_cube - along_path.unsqueeze(1) * direction
            lateral_miss = torch.linalg.norm(lateral_vec, dim=1)
            overshoot = torch.sum((cube_xy - target_xy) * direction, dim=1)
            missed_stage = (
                (stage == stage_id)
                & (~placed)
                & (xy_dist > self.handoff_place_radius)
                & (
                    (overshoot > self.handoff_goal_miss_margin)
                    | (
                        (along_path > self.handoff_goal_lateral_miss_start)
                        & (along_path < path_len + self.handoff_goal_miss_margin)
                        & (lateral_miss > self.handoff_goal_lateral_miss_margin)
                    )
                )
            )
            missed = missed | missed_stage
        return missed

    def evaluate(self):
        self._update_stage_progress()
        cube_p = self._cube_pose().p
        success = self.current_stage >= self.stage_success_threshold
        if self.curriculum_stage == 3:
            _, _, _, palm_target = self._handoff_targets()
            palm_release_target = palm_target.clone()
            palm_release_target[:, 2] += self.palm_above_z_offset
            cube_in_release_zone = (
                torch.linalg.norm(cube_p - palm_release_target, dim=1)
                <= self.palm_above_radius
            )
            success = success | ((self.current_stage == 3) & cube_in_release_zone)
        fail_drop = cube_p[:, 2] < -0.01
        table_goal_stalled = self._table_goal_stalled()
        push_contact_timeout = self._push_contact_deadline_failed()
        fail_nonfinite = ~self._finite_env_mask()
        fail = fail_drop | fail_nonfinite
        fail_reason = torch.full(
            (self.num_envs,),
            self.FAIL_REASON_NONE,
            device=self.device,
            dtype=torch.long,
        )
        fail_reason = torch.where(
            fail_drop,
            torch.full_like(fail_reason, self.FAIL_REASON_DROP),
            fail_reason,
        )
        fail_reason = torch.where(
            (~fail_drop) & (~fail_nonfinite) & push_contact_timeout,
            torch.full_like(fail_reason, self.FAIL_REASON_PUSH_TIMEOUT),
            fail_reason,
        )
        fail_reason = torch.where(
            (~fail_drop) & fail_nonfinite,
            torch.full_like(fail_reason, self.FAIL_REASON_NONFINITE),
            fail_reason,
        )
        return {
            "success": success & (~fail),
            "fail": fail,
            "fail_reason": fail_reason,
            "fail_drop": fail_drop,
            "fail_nonfinite": fail_nonfinite,
            "current_stage": self.current_stage.clone(),
            "hand_holding": self._hand_holding_cube(),
            "table_goal_stalled": table_goal_stalled,
            "push_contact_timeout": push_contact_timeout,
        }

    def _get_obs_extra(self, info: dict):
        target01, target12, target23, palm_target = self._handoff_targets()
        cube_pose = self._cube_pose()
        panda_tcp = self._agent_tcp_pose(self.panda_agent).p
        so100_tcp = self._agent_tcp_pose(self.so100_agent).p
        widowx_tcp = self._agent_tcp_pose(self.widowx_agent).p
        xarm6_tcp = self._agent_tcp_pose(self.xarm6_agent).p
        hand_palm = self._hand_palm_pose().p
        stage = self.current_stage.view(-1)
        panda_match = stage == 0
        so100_match = stage == 1
        widowx_match = stage == 2
        xarm6_match = stage == 3
        hand_match = stage >= 4

        phase_idle = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        phase_approach = torch.ones((self.num_envs,), device=self.device, dtype=torch.long)
        phase_push = torch.full((self.num_envs,), 2, device=self.device, dtype=torch.long)
        phase_return = torch.full((self.num_envs,), 3, device=self.device, dtype=torch.long)
        phase_grasp = torch.full((self.num_envs,), 4, device=self.device, dtype=torch.long)

        def _pusher_phase(stage_id: int, agent) -> torch.Tensor:
            current = torch.full((self.num_envs,), stage_id, device=self.device, dtype=torch.long)
            has_started = stage >= stage_id
            has_finished = stage > stage_id
            touching = self._agent_touching_cube(agent)
            grasping = self._generic_is_grasping(agent)
            engaged = touching | grasping
            phase = torch.where(has_started, phase_approach, phase_idle)
            phase = torch.where(has_started & engaged, phase_push, phase)
            phase = torch.where(has_finished, phase_return, phase)
            return phase

        panda_phase = _pusher_phase(0, self.panda_agent)
        so100_phase = _pusher_phase(1, self.so100_agent)
        widowx_phase = _pusher_phase(2, self.widowx_agent)
        xarm6_phase = _pusher_phase(3, self.xarm6_agent)
        hand_phase = torch.where(stage >= 4, phase_grasp, phase_idle)
        obs = {
            "cube_pos": cube_pose.p,
            "cube_q": cube_pose.q,
            "panda_tcp": panda_tcp,
            "so100_tcp": so100_tcp,
            "widowx_tcp": widowx_tcp,
            "xarm6_tcp": xarm6_tcp,
            "hand_palm": hand_palm,
            "panda_tcp_to_cube_pos": cube_pose.p - panda_tcp,
            "so100_tcp_to_cube_pos": cube_pose.p - so100_tcp,
            "widowx_tcp_to_cube_pos": cube_pose.p - widowx_tcp,
            "xarm6_tcp_to_cube_pos": cube_pose.p - xarm6_tcp,
            "hand_palm_to_cube_pos": cube_pose.p - hand_palm,
            "handoff_target_01": target01,
            "handoff_target_12": target12,
            "handoff_target_23": target23,
            "palm_target": palm_target,
            "current_stage": self.current_stage.unsqueeze(1).float(),
            "action_match": torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool),
        }
        for agent_name, agent_obj in self.agent.agents_dict.items():
            if agent_obj is self.panda_agent:
                obs[f"action_match_{agent_name}"] = panda_match
                obs[f"action_phase_{agent_name}"] = panda_phase
            elif agent_obj is self.so100_agent:
                obs[f"action_match_{agent_name}"] = so100_match
                obs[f"action_phase_{agent_name}"] = so100_phase
            elif agent_obj is self.widowx_agent:
                obs[f"action_match_{agent_name}"] = widowx_match
                obs[f"action_phase_{agent_name}"] = widowx_phase
            elif agent_obj is self.xarm6_agent:
                obs[f"action_match_{agent_name}"] = xarm6_match
                obs[f"action_phase_{agent_name}"] = xarm6_phase
            elif agent_obj is self.hand_agent:
                obs[f"action_match_{agent_name}"] = hand_match
                obs[f"action_phase_{agent_name}"] = hand_phase
        return obs

    def _push_dense_reward(
        self,
        pusher,
        stage_id: int,
        start_xy: torch.Tensor,
        target_xy: torch.Tensor,
        approach_dir_xy: torch.Tensor | None = None,
    ):
        cube_p = self._cube_pose().p
        pusher_tcp = self._agent_tcp_pose(pusher).p
        cube_xy = cube_p[:, :2]
        cube_to_target = target_xy - cube_xy
        target_dist = torch.linalg.norm(cube_to_target, dim=1)
        push_dir = cube_to_target / target_dist.unsqueeze(1).clamp_min(1e-6)
        if approach_dir_xy is None:
            # By default, approach from the side opposite to the desired push direction.
            approach_dir = -push_dir
        else:
            approach_dir = approach_dir_xy.to(device=self.device, dtype=torch.float32)
            if approach_dir.ndim == 1:
                approach_dir = approach_dir.unsqueeze(0).repeat(self.num_envs, 1)
            approach_dir = approach_dir / torch.linalg.norm(
                approach_dir, dim=1, keepdim=True
            ).clamp_min(1e-6)

        desired_tcp_xy = cube_xy + self.push_offset * approach_dir
        desired_tcp_z = torch.full(
            (self.num_envs, 1),
            self.cube_half_size + self.push_tcp_z_offset,
            device=self.device,
        )
        desired_tcp = torch.cat([desired_tcp_xy, desired_tcp_z], dim=1)
        tcp_to_push_pos = torch.linalg.norm(pusher_tcp - desired_tcp, dim=1)
        tcp_to_cube_xy = torch.linalg.norm(pusher_tcp[:, :2] - cube_xy, dim=1)
        tcp_z_err = torch.abs(pusher_tcp[:, 2] - desired_tcp_z[:, 0])
        reaching_reward = 1.0 - torch.tanh(6.0 * tcp_to_push_pos)
        stage_mask = self.current_stage == stage_id
        side_reached_now = stage_mask & (tcp_to_push_pos <= self.push_side_reach_radius)
        self._push_side_reached = torch.where(
            stage_mask,
            self._push_side_reached | side_reached_now,
            self._push_side_reached,
        )
        side_reached = stage_mask & self._push_side_reached
        cube_reach_reward = 1.0 - torch.tanh(8.0 * tcp_to_cube_xy)
        contact_band_dist = torch.abs(tcp_to_cube_xy - (self.cube_half_size + 0.004))
        contact_band_reward = 1.0 - torch.tanh(16.0 * contact_band_dist)
        z_hold_reward = 1.0 - torch.tanh(10.0 * tcp_z_err)
        touch_reward = self._agent_touching_cube(pusher).float()
        _, place_ready = self._cube_near_table_target(target_xy)
        near_target, z_progress, face_flat_progress, static_progress = self._cube_table_target_progress(
            target_xy
        )
        target_progress = torch.clamp(
            (0.40 - target_dist) / max(1e-6, 0.40 - self.handoff_place_radius),
            min=0.0,
            max=1.0,
        )

        pre_side_reward = reaching_reward
        post_side_reward = torch.full_like(reaching_reward, 0.45 * self.push_side_reach_bonus)
        post_side_reward += 0.40 * cube_reach_reward
        post_side_reward += 0.90 * contact_band_reward
        post_side_reward += 0.25 * z_hold_reward
        post_side_reward += 1.20 * touch_reward
        post_side_reward += touch_reward * (0.85 * target_progress)
        post_side_reward += touch_reward * near_target.float() * (
            0.55 * z_progress + 0.65 * face_flat_progress + 0.95 * static_progress
        )
        post_side_reward += 1.40 * place_ready.float()
        return torch.where(side_reached, post_side_reward, pre_side_reward)

    def _table_grasp_place_dense_reward(self, agent, action: Any, agent_index: int, target_xy: torch.Tensor):
        cube_p = self._cube_pose().p
        tcp_p = self._agent_tcp_pose(agent).p
        tcp_to_cube_dist = torch.linalg.norm(tcp_p - cube_p, dim=1)
        tcp_reach = 1.0 - torch.tanh(4.0 * tcp_to_cube_dist)
        loose_xy_dist, loose_placed = self._cube_near_table_target(target_xy, strict=False)
        _, _, _, static_progress = self._cube_table_target_progress(target_xy)
        xy_to_target = 1.0 - torch.tanh(7.0 * loose_xy_dist)

        # Keep stage 1/2 close to standard push shaping:
        # reach the cube, then make the cube approach the goal.
        reward = 4.0 * tcp_reach + 12.0 * xy_to_target
        reward += loose_placed.float() * 4.0 * static_progress
        return reward

    def _xarm_to_hand_dense_reward(self, action: Any):
        cube_p = self._cube_pose().p
        _, _, _, palm_target = self._handoff_targets()
        xarm_tcp = self._agent_tcp_pose(self.xarm6_agent).p
        xarm_qpos = self.xarm6_agent.robot.get_qpos()
        xarm_grasp = self._generic_is_grasping(self.xarm6_agent)
        xarm6_clear = self._agent_cube_contact_force(self.xarm6_agent) <= self.xarm6_release_contact_thresh
        xarm_reach = 1.0 - torch.tanh(4.0 * torch.linalg.norm(xarm_tcp - cube_p, dim=1))
        palm_above_target = palm_target.clone()
        palm_above_target[:, 2] += self.palm_above_z_offset
        palm_release_target = palm_above_target
        cube_to_palm_above = 1.0 - torch.tanh(
            5.0 * torch.linalg.norm(cube_p - palm_above_target, dim=1)
        )
        cube_to_release = 1.0 - torch.tanh(
            5.0 * torch.linalg.norm(cube_p - palm_release_target, dim=1)
        )
        release_zone_soft = 1.0 - torch.tanh(
            10.0 * torch.linalg.norm(cube_p - palm_release_target, dim=1)
        )
        cube_in_release_zone = (
            torch.linalg.norm(cube_p - palm_release_target, dim=1) <= self.palm_above_radius
        )
        # Reward the explicit "rotate wrist about 90 degrees, then release"
        # pose used in the scripted stage-3 video. The release target is based
        # on the recorded trigger posture plus:
        #   joint5 += 0.10
        #   joint6 += pi / 2
        # so we bias the release reward toward that more aggressive roll pose.
        joint5_release_posture = 1.0 - torch.tanh(4.5 * torch.abs(xarm_qpos[:, 4] - 1.80))
        joint6_release_posture = 1.0 - torch.tanh(6.0 * torch.abs(xarm_qpos[:, 5] - 0.56))
        wrist_release_posture = 0.3 * joint5_release_posture + 0.7 * joint6_release_posture
        # Gate most of the carry-to-palm reward until the cube is lifted
        # materially off the table, otherwise the policy can learn to drag
        # forward at low altitude and get stuck in front of the hand.
        lift_gate_start_z = self.cube_half_size + 0.014
        lift_gate_mid_z = palm_above_target[:, 2] - 0.034
        lift_gate_target_z = palm_above_target[:, 2] - 0.016
        low_height_gate = torch.clamp(
            (cube_p[:, 2] - lift_gate_start_z)
            / torch.clamp(lift_gate_mid_z - lift_gate_start_z, min=1e-6),
            min=0.0,
            max=1.0,
        )
        high_height_gate = torch.clamp(
            (cube_p[:, 2] - lift_gate_mid_z)
            / torch.clamp(lift_gate_target_z - lift_gate_mid_z, min=1e-6),
            min=0.0,
            max=1.0,
        )
        height_gate = torch.where(
            cube_p[:, 2] < lift_gate_mid_z,
            0.10 + 0.20 * low_height_gate,
            0.30 + 0.70 * (high_height_gate * high_height_gate),
        )
        height_gate = torch.clamp(height_gate, min=0.0, max=1.0)
        static_progress = 1.0 - torch.tanh(5.0 * self._cube_speed())
        released_stable = (
            (~xarm_grasp).float()
            * cube_in_release_zone.float()
            * xarm6_clear.float()
            * static_progress
        )

        # Keep stage 3 below stage4_base - stage3_base (= 22).
        # Max dense here is 21.7, still below the stage gap of 22.
        # Gate lateral carry reward by cube height in a piecewise way so
        # the policy still has some target-direction guidance before full lift,
        # but only gets the bulk of the carry reward after lifting.
        # Near the release marker, also reward a release-friendly wrist posture
        # before opening so the cube is less likely to stay hung on the gripper.
        # lift_target_z = torch.minimum(
        #     palm_above_target[:, 2],
        #     torch.full((self.num_envs,), self.handoff_lift_z, device=self.device),
        # )
        # lift_reward = 1.0 - torch.tanh(18.0 * (lift_target_z - cube_p[:, 2]).clamp_min(0.0))
        # lift_ready = cube_p[:, 2] >= (lift_target_z - 0.01)
        reward = 2.7 * xarm_reach
        reward += 3.0 * xarm_grasp.float()
        reward += 7.3 * cube_to_palm_above * xarm_grasp.float() * height_gate
        reward += 3.0 * wrist_release_posture * release_zone_soft * xarm_grasp.float()
        reward += 3.7 * cube_to_release.float() * (~xarm_grasp).float() * cube_in_release_zone.float()
        reward += 2.0 * cube_to_release.float() * released_stable
        return reward

    def _hand_hold_dense_reward(self):
        cube_p = self._cube_pose().p
        _, _, _, palm_target = self._handoff_targets()
        palm_prox = 1.0 - torch.tanh(8.0 * torch.linalg.norm(cube_p - palm_target, dim=1))
        finger_close = self._hand_finger_close_reward()
        hand_contact = self._hand_contact_reward()
        static_progress = 1.0 - torch.tanh(5.0 * self._cube_speed())
        contact_gate = torch.clamp(hand_contact, min=0.0, max=1.0)
        # Keep stage 4 below success_reward - stage4_base (= 14).
        # Max dense here is 13.2, still below that final stage gap.
        # Stage 4 should only pay well once the hand actually establishes
        # contact and stabilizes the cube near the palm target.
        grasp_dense = 3.0 * palm_prox
        grasp_dense += 3.4 * hand_contact
        grasp_dense += 1.8 * finger_close * contact_gate
        grasp_dense += 5.0 * palm_prox * contact_gate * finger_close * static_progress
        return grasp_dense

    def _agent_active_for_stage(self, stage: torch.Tensor, agent_index: int):
        if agent_index == 0:
            return stage == 0
        if agent_index == 1:
            return stage == 1
        if agent_index == 2:
            return stage == 2
        if agent_index == 3:
            return stage == 3
        if agent_index == 4:
            return stage >= 4
        return torch.zeros_like(stage, dtype=torch.bool)

    def _inactive_home_reward(self, stage: torch.Tensor):
        if self._inactive_ref_qpos is None or self._inactive_ref_tcp is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        if self.curriculum_stage is None:
            pose_coef = self.inactive_pose_reward_coef
            qpos_coef = self.inactive_qpos_reward_coef
        else:
            pose_coef = self.curriculum_inactive_pose_reward_coef
            qpos_coef = self.curriculum_inactive_qpos_reward_coef
        reward = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        for agent_index, agent in enumerate(self.agent.agents):
            active = self._agent_active_for_stage(stage, agent_index)
            inactive = (~active).float()
            tcp_dist = torch.linalg.norm(
                self._agent_tcp_pose(agent).p - self._inactive_ref_tcp[agent_index],
                dim=1,
            )
            qpos_dist = torch.linalg.norm(
                agent.robot.get_qpos() - self._inactive_ref_qpos[agent_index],
                dim=1,
            )
            tcp_home = 1.0 - torch.tanh(6.0 * tcp_dist)
            qpos_home = 1.0 - torch.tanh(0.5 * qpos_dist)
            reward += inactive * pose_coef * tcp_home
            reward += inactive * qpos_coef * qpos_home
        return reward

    def _inactive_action_still_reward(self, stage: torch.Tensor, action: Any):
        if self.curriculum_stage is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        reward = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        for agent_index, _agent in enumerate(self.agent.agents):
            active = self._agent_active_for_stage(stage, agent_index)
            inactive = (~active).float()
            agent_action = self._agent_action_tensor(action, agent_index)
            if agent_action is None:
                continue
            action_norm = torch.linalg.norm(agent_action, dim=1)
            still_reward = 1.0 - torch.tanh(1.2 * action_norm)
            reward += inactive * self.curriculum_inactive_action_still_coef * still_reward
        return reward

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        self._update_stage_progress()
        stage = self.current_stage
        target01, target12, target23, _ = self._handoff_targets()
        start01 = (
            torch.as_tensor(self.cube_xy, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # Keep stage 0 on the original push reward.
        stage0_reward = self._push_dense_reward(self.panda_agent, 0, start01, target01)
        # Legacy push reward kept for reference while stage 1/2 now use grasp-carry-place shaping.
        # stage1_reward = self._push_dense_reward(
        #     self.so100_agent,
        #     1,
        #     target01,
        #     target12,
        # )
        # stage2_reward = self._push_dense_reward(self.widowx_agent, 2, target12, target23)
        stage1_reward = self._table_grasp_place_dense_reward(
            self.so100_agent, action, 1, target12
        )
        stage2_reward = self._table_grasp_place_dense_reward(
            self.widowx_agent, action, 2, target23
        )
        stage3_reward = self._xarm_to_hand_dense_reward(action)
        stage4_reward = self._hand_hold_dense_reward()

        reward = torch.where(
            stage == 0,
            stage0_reward,
            torch.full_like(stage0_reward, self.stage1_base_reward),
        )
        reward = torch.where(
            stage == 1,
            torch.full_like(reward, self.stage1_base_reward) + stage1_reward,
            reward,
        )
        reward = torch.where(
            stage == 2,
            torch.full_like(reward, self.stage2_base_reward) + stage2_reward,
            reward,
        )
        reward = torch.where(
            stage == 3,
            torch.full_like(reward, self.stage3_base_reward) + stage3_reward,
            reward,
        )
        reward = torch.where(
            stage == 4,
            torch.full_like(reward, self.stage4_base_reward) + stage4_reward,
            reward,
        )
        reward = reward + self._inactive_home_reward(stage)
        reward = reward + self._inactive_action_still_reward(stage, action)
        reward = torch.where(
            info["success"],
            torch.full_like(reward, self._curriculum_success_reward()),
            reward,
        )
        reward = torch.where(info["fail"], torch.full_like(reward, -2.0), reward)
        reward = torch.nan_to_num(reward, nan=-2.0, posinf=self.success_reward, neginf=-2.0)
        reward = torch.clamp(reward, min=-2.0, max=self.success_reward)
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / self.success_reward
