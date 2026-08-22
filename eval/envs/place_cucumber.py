import os
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch
from sapien.pysapien import physx

os.environ.setdefault("MS_ASSET_DIR", "/home/Maniskill/.maniskill")

from mani_skill.agents.multi_agent import MultiAgent
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


def _add_wireframe_box_visual(builder, half_size, color, thickness=0.0015):
    hx, hy, hz = [float(v) for v in half_size]
    tx = min(float(thickness), hx)
    ty = min(float(thickness), hy)
    tz = min(float(thickness), hz)
    material = sapien.render.RenderMaterial(base_color=color)

    # Edges parallel to x
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            builder.add_box_visual(
                pose=sapien.Pose(p=[0.0, sy * hy, sz * hz]),
                half_size=[hx, ty, tz],
                material=material,
            )
    # Edges parallel to y
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            builder.add_box_visual(
                pose=sapien.Pose(p=[sx * hx, 0.0, sz * hz]),
                half_size=[tx, hy, tz],
                material=material,
            )
    # Edges parallel to z
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            builder.add_box_visual(
                pose=sapien.Pose(p=[sx * hx, sy * hy, 0.0]),
                half_size=[tx, ty, hz],
                material=material,
            )


COENV_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "place_cucumber_coenv"
POT_BODY_COLLISION_MESHES = (
    "original-8.obj",
    "original-13.obj",
    "original-11.obj",
    "original-10.obj",
    "original-14.obj",
)
POT_LID_COLLISION_MESHES = (
    "original-3.obj",
    "original-6.obj",
    "original-7.obj",
    "original-5.obj",
    "original-4.obj",
    "original-1.obj",
)


@register_env("PlaceCucumber-v1", max_episode_steps=200)
class PlaceCucumberEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("panda_wristcam", "widowxai", "widowxai")]
    agent: MultiAgent
    SUPPORTED_REWARD_MODES = ["none", "normalized_dense"]

    pot_asset_scale = 0.25
    pot_base_z = 0.0
    pot_radius = 0.16
    pot_height = 0.07011575
    lid_radius = 0.02125
    goal_site_radius = 0.16
    goal_site_half_length = 0.2
    lid_handle_height = 0.14
    lid_handle_marker_radius = 0.05
    lid_handle_marker_z_lift = 0.0
    lid_handle_grasp_gap = 0.05
    lid_init_qpos = 0.05
    lid_closed_qpos_thresh = 0.008
    lid_open_qpos = 0.2
    lid_open_stage_progress_thresh = 0.7
    lid_qpos_noise = 0.01
    lid_qvel_thresh = 0.12
    lid_sticky_grasp_enabled = True
    lid_sticky_grasp_dist = 0.055
    lid_sticky_release_gap = 0.075
    lid_sticky_qpos_per_meter = 2.2
    center_rest_joint1_delta = -0.1
    cucumber_asset_scale = 0.0765
    cucumber_radius = 0.0216
    cucumber_half_length = 0.0765
    cucumber_box_half_size_x = 0.018
    cucumber_box_half_size_y = 0.0675
    cucumber_box_half_size_z = 0.018
    cucumber_collision_box_rgba = [0.15, 1.0, 0.15, 1.0]
    cucumber_density = 285.7796067672614
    cucumber_static_friction = 2.0
    cucumber_dynamic_friction = 2.0
    cucumber_restitution = 0.0
    cucumber_goal_z_min = 0.0
    cucumber_goal_z_max = pot_height + 0.02
    cucumber_insert_target_z = 0.1
    cucumber_approach_z_offset = 0.08
    cucumber_approach_area_radius = 0.09
    cucumber_release_area_radius = 0.16
    cucumber_release_z_max = 0.14
    cucumber_retract_target_xy_margin = 0.06
    cucumber_retract_target_x_offset = 0.10
    cucumber_retract_target_y_outer_offset = 0.10
    cucumber_retract_target_z = 0.10
    cucumber_retract_marker_radius = 0.04
    cucumber_retract_marker_half_length = 0.035
    cucumber_top_grasp_z_offset = 0.005
    cucumber_release_grace_steps = 5
    lid_open_stage_bonus = 3.4
    cucumber_grasp_stage_bonus = 7.0
    cucumber_release_stage_bonus = 18.0
    cucumber_retract_stage_bonus = 30.0
    partner_release_prompt_bonus = 2.0
    lid_drop_penalty_coef = 4.0
    cucumber_invalid_penalty_coef = 4.0
    static_vel_thresh = 0.2
    tcp_marker_radius = 0.015
    widowx_tcp_inset = 0.035

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "widowxai", "widowxai"),
        robot_init_qpos_noise=0.02,
        center_robot_xy=(0.82, 0.0),
        left_robot_xy=(-0.02, 0.30),
        right_robot_xy=(-0.02, -0.30),
        center_robot_yaw=np.pi,
        left_robot_yaw=0.0,
        right_robot_yaw=0.0,
        pot_xy=(0.26, 0.0),
        cucumber1_xy=(0.18, 0.32),
        cucumber2_xy=(0.18, -0.28),
        cucumber_size_scale=1.0,
        enable_domain_randomization=False,
        domain_randomization_level="mild",
        enable_mixed_domain_randomization=False,
        mixed_domain_randomization_clear_ratio=0.2,
        mixed_domain_randomization_mild_ratio=0.5,
        mixed_domain_randomization_hard_ratio=0.3,
        **kwargs,
    ):
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.center_robot_xy = tuple(float(v) for v in center_robot_xy)
        self.left_robot_xy = tuple(float(v) for v in left_robot_xy)
        self.right_robot_xy = tuple(float(v) for v in right_robot_xy)
        self.center_robot_yaw = float(center_robot_yaw)
        self.left_robot_yaw = float(left_robot_yaw)
        self.right_robot_yaw = float(right_robot_yaw)
        self.pot_xy = tuple(float(v) for v in pot_xy)
        self.cucumber1_xy = tuple(float(v) for v in cucumber1_xy)
        self.cucumber2_xy = tuple(float(v) for v in cucumber2_xy)
        self.enable_domain_randomization = bool(enable_domain_randomization)
        self.domain_randomization_level = str(domain_randomization_level).lower()
        if self.domain_randomization_level not in {"mild", "hard"}:
            raise ValueError(
                f"Unsupported domain_randomization_level={domain_randomization_level}. "
                "Expected one of: mild, hard"
            )
        self.enable_mixed_domain_randomization = bool(enable_mixed_domain_randomization)
        self.mixed_domain_randomization_clear_ratio = float(mixed_domain_randomization_clear_ratio)
        self.mixed_domain_randomization_mild_ratio = float(mixed_domain_randomization_mild_ratio)
        self.mixed_domain_randomization_hard_ratio = float(mixed_domain_randomization_hard_ratio)
        self.cucumber_size_scale = float(cucumber_size_scale)
        if self.cucumber_size_scale <= 0:
            raise ValueError(f"cucumber_size_scale must be positive, got {self.cucumber_size_scale}")
        self.cucumber_asset_scale = type(self).cucumber_asset_scale * self.cucumber_size_scale
        self.cucumber_radius = type(self).cucumber_radius * self.cucumber_size_scale
        self.cucumber_half_length = type(self).cucumber_half_length * self.cucumber_size_scale
        self.cucumber_box_half_size_x = type(self).cucumber_box_half_size_x * self.cucumber_size_scale
        self.cucumber_box_half_size_y = type(self).cucumber_box_half_size_y * self.cucumber_size_scale
        self.cucumber_box_half_size_z = type(self).cucumber_box_half_size_z * self.cucumber_size_scale
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

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
        pose = sapien_utils.look_at([-0.92, 0.0, 0.88], [0.20, 0.0, 0.16])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 3, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-1.08, 0.0, 0.96], [0.22, 0.0, 0.16])
        return CameraConfig("render_camera", pose, 512, 512, np.pi / 3, 0.01, 100)

    @property
    def center_agent(self):
        return self.agent.agents[0]

    @property
    def left_agent(self):
        return self.agent.agents[1]

    @property
    def right_agent(self):
        return self.agent.agents[2]

    def _center_robot_pose(self):
        return sapien.Pose(
            p=[self.center_robot_xy[0], self.center_robot_xy[1], 0.0],
            q=_yaw_quat(self.center_robot_yaw),
        )

    def _left_robot_pose(self):
        return sapien.Pose(
            p=[self.left_robot_xy[0], self.left_robot_xy[1], 0.0],
            q=_yaw_quat(self.left_robot_yaw),
        )

    def _right_robot_pose(self):
        return sapien.Pose(
            p=[self.right_robot_xy[0], self.right_robot_xy[1], 0.0],
            q=_yaw_quat(self.right_robot_yaw),
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            [
                self._center_robot_pose(),
                self._left_robot_pose(),
                self._right_robot_pose(),
            ],
        )

    def _build_cucumber(self, name: str):
        asset_name = "cucumberA.glb" if name == "cucumber1" else "cucumberB.glb"
        mesh_path = COENV_ASSET_DIR / "cucumber" / asset_name
        builder = self.scene.create_actor_builder()
        cucumber_material = physx.PhysxMaterial(
            self.cucumber_static_friction,
            self.cucumber_dynamic_friction,
            self.cucumber_restitution,
        )
        builder.set_initial_pose(sapien.Pose(p=[0.0, 0.0, self.cucumber_radius]))
        builder.add_visual_from_file(
            filename=str(mesh_path),
            scale=[self.cucumber_asset_scale] * 3,
        )
        _add_wireframe_box_visual(
            builder,
            half_size=[
                self.cucumber_box_half_size_x,
                self.cucumber_box_half_size_y,
                self.cucumber_box_half_size_z,
            ],
            color=self.cucumber_collision_box_rgba,
        )
        builder.add_box_collision(
            pose=sapien.Pose(),
            half_size=[
                self.cucumber_box_half_size_x,
                self.cucumber_box_half_size_y,
                self.cucumber_box_half_size_z,
            ],
            material=cucumber_material,
            density=self.cucumber_density,
        )
        return builder.build(name=name)

    def _build_pot(self):
        loader = self.scene.create_urdf_loader()
        loader.scale = self.pot_asset_scale
        parsed = loader.parse(str(COENV_ASSET_DIR / "pot_annotated" / "mobility.urdf"))
        articulation_builders = parsed["articulation_builders"]
        if len(articulation_builders) != 1:
            raise RuntimeError(
                f"Expected one pot articulation, got {len(articulation_builders)}"
            )
        builder = articulation_builders[0]
        lid_builder = next(
            (lb for lb in builder.link_builders if getattr(lb, "name", None) == "link_0"),
            None,
        )
        pot_body_builder = next(
            (lb for lb in builder.link_builders if getattr(lb, "name", None) == "link_1"),
            None,
        )
        if lid_builder is None:
            raise RuntimeError("Failed to find lid link builder link_0")
        if pot_body_builder is None:
            raise RuntimeError("Failed to find pot body link builder link_1")
        lid_builder.collision_records = []
        for mesh_name in POT_LID_COLLISION_MESHES:
            lid_builder.add_multiple_convex_collisions_from_file(
                filename=str(
                    COENV_ASSET_DIR / "pot_annotated" / "textured_objs" / mesh_name
                ),
                decomposition="coacd",
                scale=[self.pot_asset_scale] * 3,
            )
        pot_body_builder.collision_records = []
        for mesh_name in POT_BODY_COLLISION_MESHES:
            pot_body_builder.add_multiple_convex_collisions_from_file(
                filename=str(
                    COENV_ASSET_DIR / "pot_annotated" / "textured_objs" / mesh_name
                ),
                decomposition="coacd",
                scale=[self.pot_asset_scale] * 3,
            )
        builder.set_initial_pose(
            sapien.Pose(
                p=[self.pot_xy[0], self.pot_xy[1], self.pot_base_z],
                q=_yaw_quat(np.pi / 2),
            )
        )
        builder.set_scene_idxs(torch.arange(self.num_envs, device=self.device))
        self.pot_articulation = builder.build(name="pot")
        links_by_name = {link.name: link for link in self.pot_articulation.links}
        self.pot = links_by_name.get("link_1", self.pot_articulation.links[0])
        self.lid = links_by_name.get("link_0", self.pot_articulation.links[-1])

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=0.0)
        self.table_scene.build()

        self._build_pot()

        self.cucumber1 = self._build_cucumber("cucumber1")
        self.cucumber2 = self._build_cucumber("cucumber2")

        # self.goal_site = actors.build_cylinder(
        #     self.scene,
        #     radius=self.goal_site_radius,
        #     half_length=self.goal_site_half_length,
        #     color=[0.2, 0.9, 0.2, 0.35],
        #     name="pot_goal",
        #     body_type="kinematic",
        #     add_collision=False,
        #     initial_pose=sapien.Pose(
        #         p=[self.pot_xy[0], self.pot_xy[1], self.pot_base_z + self.pot_height + 0.001],
        #         q=[float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0],
        #     ),
        # )
        self.goal_site = None
        self.lid_goal_site = None
        self.center_tcp_marker = None
        self.left_tcp_marker = None
        self.right_tcp_marker = None
        self.left_retract_target_marker = None
        self.right_retract_target_marker = None
        # self.lid_handle_marker = actors.build_sphere(
        #     self.scene,
        #     radius=self.lid_handle_marker_radius,
        #     color=[0.0, 1.0, 0.0, 1.0],
        #     body_type="kinematic",
        #     add_collision=False,
        #     name="lid_handle_marker",
        #     initial_pose=sapien.Pose(
        #         p=[
        #             self.pot_xy[0],
        #             self.pot_xy[1],
        #             self.pot_base_z
        #             + self.pot_height
        #             + self.lid_handle_height
        #             + self.lid_handle_marker_z_lift,
        #         ]
        #     ),
        # )
        self.lid_handle_marker = None
        self.lid_opened_with_grasp = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.lid_sticky = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.lid_sticky_ref_finger_z = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        self.lid_sticky_ref_qpos = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        self.cucumber1_grasped = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.cucumber2_grasped = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.cucumber1_released = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.cucumber2_released = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.cucumber1_retracted = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.cucumber2_retracted = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.cucumber1_release_ready_steps = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self.cucumber2_release_ready_steps = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self.left_home_qpos = None
        self.right_home_qpos = None

    def _sample_xy(self, env_count, center_xy, scale_xy):
        base = torch.tensor(center_xy, device=self.device, dtype=torch.float32).unsqueeze(0)
        noise = (torch.rand((env_count, 2), device=self.device) * 2.0 - 1.0) * torch.tensor(
            scale_xy, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        return base + noise

    def _sample_xy_from_centers(self, centers, scale_xy):
        noise = (torch.rand_like(centers) * 2.0 - 1.0) * torch.tensor(
            scale_xy, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        return centers + noise

    def _sample_cucumber_spawn_centers(self, env_count):
        c1 = torch.tensor(self.cucumber1_xy, device=self.device, dtype=torch.float32).repeat(env_count, 1)
        c2 = torch.tensor(self.cucumber2_xy, device=self.device, dtype=torch.float32).repeat(env_count, 1)
        if not (self.enable_domain_randomization or self.enable_mixed_domain_randomization):
            return c1, c2

        if self.enable_mixed_domain_randomization:
            probs = torch.tensor(
                [
                    self.mixed_domain_randomization_clear_ratio,
                    self.mixed_domain_randomization_mild_ratio,
                    self.mixed_domain_randomization_hard_ratio,
                ],
                device=self.device,
                dtype=torch.float32,
            )
            probs = torch.clamp(probs, min=0.0)
            if float(probs.sum()) <= 0.0:
                probs = torch.tensor([0.2, 0.5, 0.3], device=self.device, dtype=torch.float32)
            probs = probs / probs.sum()
            levels = torch.multinomial(probs, env_count, replacement=True)
            mild_mask = levels == 1
            hard_mask = levels == 2
        else:
            mild_mask = torch.full((env_count,), self.domain_randomization_level == "mild", device=self.device)
            hard_mask = torch.full((env_count,), self.domain_randomization_level == "hard", device=self.device)

        # Continuous variants of the online_rl shifts: mild keeps cucumbers close
        # to the original spawn band; hard samples toward the FarOut setting.
        if bool(mild_mask.any()):
            n = int(mild_mask.sum().item())
            c1[mild_mask, 0] = 0.18 + (torch.rand((n,), device=self.device) * 2.0 - 1.0) * 0.025
            c2[mild_mask, 0] = 0.18 + (torch.rand((n,), device=self.device) * 2.0 - 1.0) * 0.025
            c1[mild_mask, 1] = 0.32 + (torch.rand((n,), device=self.device) * 2.0 - 1.0) * 0.025
            c2[mild_mask, 1] = -0.28 + (torch.rand((n,), device=self.device) * 2.0 - 1.0) * 0.025
        if bool(hard_mask.any()):
            n = int(hard_mask.sum().item())
            c1[hard_mask, 0] = 0.18 + (torch.rand((n,), device=self.device) * 2.0 - 1.0) * 0.030
            c2[hard_mask, 0] = 0.18 + (torch.rand((n,), device=self.device) * 2.0 - 1.0) * 0.030
            c1[hard_mask, 1] = 0.35 + torch.rand((n,), device=self.device) * 0.040
            c2[hard_mask, 1] = -0.31 - torch.rand((n,), device=self.device) * 0.040
        return c1, c2

    def _sample_yaw_quat(self, env_count):
        yaw = (torch.rand((env_count,), device=self.device) * 2.0 - 1.0) * (0.5 * np.pi)
        q = torch.zeros((env_count, 4), device=self.device)
        q[:, 0] = torch.cos(yaw / 2.0)
        q[:, 3] = torch.sin(yaw / 2.0)
        return q

    def _retract_target_xy(self):
        left_target = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        right_target = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        left_target[:, 0] = self.left_robot_xy[0] + self.cucumber_retract_target_x_offset
        right_target[:, 0] = self.right_robot_xy[0] + self.cucumber_retract_target_x_offset
        left_target[:, 1] = self.left_robot_xy[1] + self.cucumber_retract_target_y_outer_offset
        right_target[:, 1] = self.right_robot_xy[1] - self.cucumber_retract_target_y_outer_offset
        return left_target, right_target

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            def _sample_robot_rest_qpos(robot):
                qpos = robot.robot.get_qpos().clone()
                if "rest" in robot.keyframes:
                    base_qpos = torch.as_tensor(
                        robot.keyframes["rest"].qpos,
                        device=self.device,
                        dtype=qpos.dtype,
                    )
                elif len(robot.keyframes) > 0:
                    first_keyframe = next(iter(robot.keyframes.values()))
                    base_qpos = torch.as_tensor(
                        first_keyframe.qpos,
                        device=self.device,
                        dtype=qpos.dtype,
                    )
                else:
                    base_qpos = qpos[env_idx][0]
                rest = base_qpos.unsqueeze(0).repeat(b, 1)
                rest += (torch.rand_like(rest) * 2 - 1) * self.robot_init_qpos_noise
                return qpos, rest

            for robot in [self.center_agent, self.left_agent, self.right_agent]:
                qpos, rest = _sample_robot_rest_qpos(robot)
                if robot is self.center_agent:
                    rest[:, 1] += self.center_rest_joint1_delta
                qpos[env_idx] = rest
                robot.robot.set_qpos(qpos[env_idx])

                if robot is self.left_agent:
                    if self.left_home_qpos is None:
                        self.left_home_qpos = robot.robot.get_qpos().clone()
                    self.left_home_qpos[env_idx] = rest
                elif robot is self.right_agent:
                    if self.right_home_qpos is None:
                        self.right_home_qpos = robot.robot.get_qpos().clone()
                    self.right_home_qpos[env_idx] = rest

            self.center_agent.robot.set_pose(self._center_robot_pose())
            self.left_agent.robot.set_pose(self._left_robot_pose())
            self.right_agent.robot.set_pose(self._right_robot_pose())

            pot_qpos = self.pot_articulation.get_qpos().clone()
            pot_qvel = self.pot_articulation.get_qvel().clone()
            open_qpos = torch.full((b,), self.lid_init_qpos, device=self.device)
            pot_qpos[env_idx, 0] = torch.clamp(
                open_qpos,
                min=0.03,
                max=self.lid_init_qpos,
            )
            pot_qvel[env_idx, 0] = 0.0
            self.pot_articulation.set_qpos(pot_qpos[env_idx])
            self.pot_articulation.set_qvel(pot_qvel[env_idx])
            self.lid_opened_with_grasp[env_idx] = False
            self.lid_sticky[env_idx] = False
            self.lid_sticky_ref_finger_z[env_idx] = 0.0
            self.lid_sticky_ref_qpos[env_idx] = 0.0
            self.cucumber1_grasped[env_idx] = False
            self.cucumber2_grasped[env_idx] = False
            self.cucumber1_released[env_idx] = False
            self.cucumber2_released[env_idx] = False
            self.cucumber1_retracted[env_idx] = False
            self.cucumber2_retracted[env_idx] = False
            self.cucumber1_release_ready_steps[env_idx] = 0
            self.cucumber2_release_ready_steps[env_idx] = 0

            cucumber1_center_xy, cucumber2_center_xy = self._sample_cucumber_spawn_centers(b)
            cucumber1_pose = self.cucumber1.pose.raw_pose.clone()
            cucumber1_xy = self._sample_xy_from_centers(cucumber1_center_xy, (0.035, 0.035))
            cucumber1_pose[env_idx, :2] = cucumber1_xy
            cucumber1_pose[env_idx, 2] = self.cucumber_radius
            cucumber1_pose[env_idx, 3:] = self._sample_yaw_quat(b)
            self.cucumber1.set_pose(Pose.create(cucumber1_pose[env_idx]))
            self.cucumber1.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.cucumber1.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            cucumber2_pose = self.cucumber2.pose.raw_pose.clone()
            cucumber2_xy = self._sample_xy_from_centers(cucumber2_center_xy, (0.035, 0.035))
            cucumber2_pose[env_idx, :2] = cucumber2_xy
            cucumber2_pose[env_idx, 2] = self.cucumber_radius
            cucumber2_pose[env_idx, 3:] = self._sample_yaw_quat(b)
            self.cucumber2.set_pose(Pose.create(cucumber2_pose[env_idx]))
            self.cucumber2.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.cucumber2.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            # goal_pose = self.goal_site.pose.raw_pose.clone()
            # goal_pose[env_idx, 0] = self.pot_xy[0]
            # goal_pose[env_idx, 1] = self.pot_xy[1]
            # goal_pose[env_idx, 2] = self.pot_base_z + self.pot_height + 0.001
            # self.goal_site.set_pose(Pose.create(goal_pose[env_idx]))
            self._sync_lid_handle_marker(env_idx)
            self._sync_tcp_markers(env_idx)
            self._sync_retract_target_markers(env_idx)

    def _agent_tcp_pose(self, agent):
        if hasattr(agent, "tcp_pose"):
            tcp_pose = agent.tcp_pose
        elif hasattr(agent, "tcp") and hasattr(agent.tcp, "pose"):
            tcp_pose = agent.tcp.pose
        else:
            raise AttributeError(f"{type(agent).__name__} does not expose a TCP pose")

        if getattr(agent, "uid", None) == "widowxai":
            rot = tcp_pose.to_transformation_matrix()[..., :3, :3]
            local_offset = torch.zeros_like(tcp_pose.p)
            local_offset[:, 0] = -self.widowx_tcp_inset
            world_offset = torch.einsum("bij,bj->bi", rot, local_offset)
            return Pose.create_from_pq(tcp_pose.p + world_offset, tcp_pose.q)

        return tcp_pose

    def _robot_link_pose(self, agent, link_name: str):
        return self._robot_link(agent, link_name).pose

    def _robot_link(self, agent, link_name: str):
        robot = agent.robot
        if hasattr(robot, "links_map") and link_name in robot.links_map:
            return robot.links_map[link_name]
        for link in robot.links:
            if link.name == link_name:
                return link
        raise KeyError(f"Link {link_name} not found on {type(robot).__name__}")

    def _normalize_vec(self, v: torch.Tensor, eps: float = 1e-6):
        return v / torch.clamp(torch.linalg.norm(v, dim=1, keepdim=True), min=eps)

    def _agent_arm_qpos(self, agent):
        qpos = agent.robot.get_qpos()
        arm_joint_names = getattr(agent, "arm_joint_names", None)
        if arm_joint_names is not None:
            return qpos[..., : len(arm_joint_names)]
        return qpos

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

    def _agent_gripper_gap(self, agent):
        if hasattr(agent, "finger1_link") and hasattr(agent, "finger2_link"):
            return torch.linalg.norm(agent.finger1_link.pose.p - agent.finger2_link.pose.p, dim=1)
        left_finger = self._robot_link(agent, "panda_leftfinger").pose.p
        right_finger = self._robot_link(agent, "panda_rightfinger").pose.p
        return torch.linalg.norm(left_finger - right_finger, dim=1)

    def _center_finger_midpoint(self):
        left_finger = self._robot_link(self.center_agent, "panda_leftfinger").pose.p
        right_finger = self._robot_link(self.center_agent, "panda_rightfinger").pose.p
        return 0.5 * (left_finger + right_finger)

    def _after_control_step(self):
        if not self.lid_sticky_grasp_enabled:
            return

        finger_mid = self._center_finger_midpoint()
        finger_gap = self._agent_gripper_gap(self.center_agent)
        handle_pos = self._lid_handle_pos()
        handle_dist = torch.linalg.norm(finger_mid - handle_pos, dim=1)
        lid_qpos = self._lid_qpos()

        newly_sticky = (
            (~self.lid_sticky)
            & (handle_dist <= self.lid_sticky_grasp_dist)
            & (finger_gap <= self.lid_sticky_release_gap)
        )
        if torch.any(newly_sticky):
            self.lid_sticky_ref_finger_z[newly_sticky] = finger_mid[newly_sticky, 2]
            self.lid_sticky_ref_qpos[newly_sticky] = lid_qpos[newly_sticky]

        self.lid_sticky = torch.logical_or(self.lid_sticky, newly_sticky)
        self.lid_sticky = torch.logical_and(
            self.lid_sticky,
            finger_gap <= self.lid_sticky_release_gap,
        )

        if not torch.any(self.lid_sticky):
            return

        target_qpos = lid_qpos.clone()
        dz = finger_mid[:, 2] - self.lid_sticky_ref_finger_z
        target_qpos[self.lid_sticky] = torch.clamp(
            self.lid_sticky_ref_qpos[self.lid_sticky]
            + self.lid_sticky_qpos_per_meter * dz[self.lid_sticky],
            min=0.0,
            max=max(self.lid_open_qpos * 1.8, self.lid_init_qpos),
        )

        qpos = self.pot_articulation.get_qpos().clone()
        qvel = self.pot_articulation.get_qvel().clone()
        qpos[:, 0] = target_qpos
        qvel[self.lid_sticky, 0] = 0.0
        self.pot_articulation.set_qpos(qpos)
        self.pot_articulation.set_qvel(qvel)

    def _grip_line_axis(self, agent):
        if hasattr(agent, "finger1_link") and hasattr(agent, "finger2_link"):
            left_finger = agent.finger1_link.pose.p
            right_finger = agent.finger2_link.pose.p
        else:
            left_finger = self._robot_link(agent, "left_finger_link").pose.p
            right_finger = self._robot_link(agent, "right_finger_link").pose.p
        return self._normalize_vec(left_finger - right_finger)

    def _object_long_axis(self, obj):
        rot = obj.pose.to_transformation_matrix()[..., :3, :3]
        # The elongated cucumber collision box is defined along local +y.
        return self._normalize_vec(rot[..., :, 1])

    def _lid_handle_pos(self):
        lid_handle_pos = self.lid.pose.p.clone()
        lid_handle_pos[:, 2] += self.lid_handle_height
        return lid_handle_pos

    def _sync_lid_handle_marker(self, env_idx: torch.Tensor | None = None):
        if self.lid_handle_marker is None:
            return
        marker_pose = self.lid_handle_marker.pose.raw_pose.clone()
        if env_idx is None:
            marker_pose[:, :3] = self._lid_handle_pos()
            marker_pose[:, 2] += self.lid_handle_marker_z_lift
            marker_pose[:, 3] = 1.0
            marker_pose[:, 4:] = 0.0
            self.lid_handle_marker.set_pose(Pose.create(marker_pose))
        else:
            marker_pose[env_idx, :3] = self._lid_handle_pos()[env_idx]
            marker_pose[env_idx, 2] += self.lid_handle_marker_z_lift
            marker_pose[env_idx, 3] = 1.0
            marker_pose[env_idx, 4:] = 0.0
            self.lid_handle_marker.set_pose(Pose.create(marker_pose[env_idx]))

    def _sync_tcp_markers(self, env_idx: torch.Tensor | None = None):
        marker_specs = [
            (self.center_tcp_marker, self._agent_tcp_pose(self.center_agent)),
            (self.left_tcp_marker, self._agent_tcp_pose(self.left_agent)),
            (self.right_tcp_marker, self._agent_tcp_pose(self.right_agent)),
        ]
        for marker, tcp_pose in marker_specs:
            if marker is None:
                continue
            marker_pose = marker.pose.raw_pose.clone()
            if env_idx is None:
                marker_pose[:, :3] = tcp_pose.p
                marker_pose[:, 3:] = tcp_pose.q
                marker.set_pose(Pose.create(marker_pose))
            else:
                marker_pose[env_idx, :3] = tcp_pose.p[env_idx]
                marker_pose[env_idx, 3:] = tcp_pose.q[env_idx]
                marker.set_pose(Pose.create(marker_pose[env_idx]))

    def _sync_retract_target_markers(self, env_idx: torch.Tensor | None = None):
        if self.left_retract_target_marker is None or self.right_retract_target_marker is None:
            return
        left_xy, right_xy = self._retract_target_xy()
        target_specs = [
            (self.left_retract_target_marker, left_xy),
            (self.right_retract_target_marker, right_xy),
        ]
        marker_q = torch.tensor(
            [float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        for marker, xy in target_specs:
            marker_pose = marker.pose.raw_pose.clone()
            if env_idx is None:
                marker_pose[:, 0:2] = xy
                marker_pose[:, 2] = self.cucumber_retract_target_z
                marker_pose[:, 3:] = marker_q
                marker.set_pose(Pose.create(marker_pose))
            else:
                marker_pose[env_idx, 0:2] = xy[env_idx]
                marker_pose[env_idx, 2] = self.cucumber_retract_target_z
                marker_pose[env_idx, 3:] = marker_q
                marker.set_pose(Pose.create(marker_pose[env_idx]))

    def _lid_qpos(self):
        return self.pot_articulation.qpos[:, 0]

    def _lid_qvel(self):
        return self.pot_articulation.qvel[:, 0]

    def _in_xy_area(self, pos: torch.Tensor, center: torch.Tensor, radius: float):
        xy_dist = torch.linalg.norm(pos[:, :2] - center[:, :2], dim=1)
        return xy_dist, xy_dist <= radius

    def _obj_speed(self, obj):
        lin = torch.linalg.norm(obj.linear_velocity, dim=1)
        ang = torch.linalg.norm(obj.angular_velocity, dim=1)
        return lin + 0.3 * ang

    def _in_pot(self, obj):
        pot_pos = self.pot.pose.p
        obj_pos = obj.pose.p
        xy_dist = torch.linalg.norm(obj_pos[:, :2] - pot_pos[:, :2], dim=1)
        xy_ok = xy_dist < self.pot_radius
        z_ok = (obj_pos[:, 2] > self.cucumber_goal_z_min) & (obj_pos[:, 2] < self.cucumber_goal_z_max)
        return xy_ok, z_ok

    def evaluate(self):
        self._sync_lid_handle_marker()
        self._sync_tcp_markers()
        c1_xy_ok, c1_z_ok = self._in_pot(self.cucumber1)
        c2_xy_ok, c2_z_ok = self._in_pot(self.cucumber2)
        c1_released = ~self.left_agent.is_grasping(self.cucumber1)
        c2_released = ~self.right_agent.is_grasping(self.cucumber2)
        lid_ok = torch.abs(self._lid_qpos()) < self.lid_closed_qpos_thresh
        static_ok = (
            (self._obj_speed(self.cucumber1) < self.static_vel_thresh)
            & (self._obj_speed(self.cucumber2) < self.static_vel_thresh)
        )
        released_ok = c1_released & c2_released
        cucumber1_fell = self.cucumber1.pose.p[:, 2] < -0.03
        cucumber2_fell = self.cucumber2.pose.p[:, 2] < -0.03
        fail = cucumber1_fell | cucumber2_fell
        success = (
            c1_xy_ok
            & c1_z_ok
            & c2_xy_ok
            & c2_z_ok
            & released_ok
            & static_ok
            & (~fail)
        )
        return {
            "success": success,
            "fail": fail,
            "cucumber1_fell": cucumber1_fell,
            "cucumber2_fell": cucumber2_fell,
            "cucumber1_in_pot": c1_xy_ok & c1_z_ok,
            "cucumber2_in_pot": c2_xy_ok & c2_z_ok,
            "released_ok": released_ok,
            "lid_closed": lid_ok,
            "static": static_ok,
        }

    def _get_obs_extra(self, info: dict):
        self._sync_lid_handle_marker()
        self._sync_tcp_markers()
        center_tcp = self._agent_tcp_pose(self.center_agent).p
        left_tcp = self._agent_tcp_pose(self.left_agent).p
        right_tcp = self._agent_tcp_pose(self.right_agent).p
        center_tcp_q = self._agent_tcp_pose(self.center_agent).q
        left_tcp_q = self._agent_tcp_pose(self.left_agent).q
        right_tcp_q = self._agent_tcp_pose(self.right_agent).q
        pot_pos = self.pot.pose.p
        lid_pos = self.lid.pose.p
        lid_handle_pos = self._lid_handle_pos()
        c1_pos = self.cucumber1.pose.p
        c2_pos = self.cucumber2.pose.p
        c1_q = self.cucumber1.pose.q
        c2_q = self.cucumber2.pose.q
        c1_long_axis = self._object_long_axis(self.cucumber1)
        c2_long_axis = self._object_long_axis(self.cucumber2)
        left_grip_line_axis = self._grip_line_axis(self.left_agent)
        right_grip_line_axis = self._grip_line_axis(self.right_agent)
        left_phase = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        left_phase = torch.where(self.cucumber1_grasped, torch.ones_like(left_phase), left_phase)
        left_phase = torch.where(self.cucumber1_released, torch.full_like(left_phase, 2), left_phase)
        left_phase = torch.where(self.cucumber1_retracted, torch.full_like(left_phase, 3), left_phase)

        right_phase = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        right_phase = torch.where(self.cucumber2_grasped, torch.ones_like(right_phase), right_phase)
        right_phase = torch.where(self.cucumber2_released, torch.full_like(right_phase, 2), right_phase)
        right_phase = torch.where(self.cucumber2_retracted, torch.full_like(right_phase, 3), right_phase)

        left_right_match = (left_phase == right_phase) & (left_phase < 3)
        center_match = (~self.lid_opened_with_grasp) | (
            self.lid_opened_with_grasp & self.cucumber1_retracted & self.cucumber2_retracted
        )

        # Semantic phase ids for effective-communication accounting.
        # Left/right agents share the same phase vocabulary because they are
        # mirror-symmetric workers on the same subtask family.
        # Center uses distinct ids while opening / monitoring the lid, then
        # joins the shared "done/hold" phase after both side agents retract.
        center_phase = torch.full((self.num_envs,), 10, device=self.device, dtype=torch.long)
        center_phase = torch.where(self.lid_opened_with_grasp, torch.full_like(center_phase, 11), center_phase)
        center_phase = torch.where(
            self.lid_opened_with_grasp & self.cucumber1_retracted & self.cucumber2_retracted,
            torch.full_like(center_phase, 3),
            center_phase,
        )

        obs = {
            "center_tcp": center_tcp,
            "left_tcp": left_tcp,
            "right_tcp": right_tcp,
            "center_tcp_q": center_tcp_q,
            "left_tcp_q": left_tcp_q,
            "right_tcp_q": right_tcp_q,
            "pot_pos": pot_pos,
            "lid_pos": lid_pos,
            "lid_handle_pos": lid_handle_pos,
            "lid_qpos": self.pot_articulation.qpos,
            "lid_qvel": self.pot_articulation.qvel,
            "cucumber1_pos": c1_pos,
            "cucumber2_pos": c2_pos,
            "cucumber1_q": c1_q,
            "cucumber2_q": c2_q,
            "cucumber1_long_axis": c1_long_axis,
            "cucumber2_long_axis": c2_long_axis,
            "left_grip_line_axis": left_grip_line_axis,
            "right_grip_line_axis": right_grip_line_axis,
            "left_tcp_to_cucumber1": c1_pos - left_tcp,
            "right_tcp_to_cucumber2": c2_pos - right_tcp,
            "center_tcp_to_lid_handle": lid_handle_pos - center_tcp,
            "cucumber1_to_pot": pot_pos - c1_pos,
            "cucumber2_to_pot": pot_pos - c2_pos,
            "action_match": left_right_match,
        }
        for agent_name, agent_obj in self.agent.agents_dict.items():
            if agent_obj is self.center_agent:
                obs[f"action_match_{agent_name}"] = center_match
                obs[f"action_phase_{agent_name}"] = center_phase
            elif agent_obj is self.left_agent:
                obs[f"action_match_{agent_name}"] = left_right_match
                obs[f"action_phase_{agent_name}"] = left_phase
            elif agent_obj is self.right_agent:
                obs[f"action_match_{agent_name}"] = left_right_match
                obs[f"action_phase_{agent_name}"] = right_phase
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        self._sync_lid_handle_marker()
        self._sync_tcp_markers()
        c1_xy_ok, c1_z_ok = self._in_pot(self.cucumber1)
        c2_xy_ok, c2_z_ok = self._in_pot(self.cucumber2)
        c1_in = c1_xy_ok & c1_z_ok
        c2_in = c2_xy_ok & c2_z_ok
        cucumbers_in = c1_in & c2_in

        left_tcp = self._agent_tcp_pose(self.left_agent).p
        right_tcp = self._agent_tcp_pose(self.right_agent).p
        center_tcp = self._agent_tcp_pose(self.center_agent).p
        center_left_finger_link = self._robot_link(self.center_agent, "panda_leftfinger")
        center_right_finger_link = self._robot_link(self.center_agent, "panda_rightfinger")
        center_left_finger = center_left_finger_link.pose.p
        center_right_finger = center_right_finger_link.pose.p

        c1_pos = self.cucumber1.pose.p
        c2_pos = self.cucumber2.pose.p
        c1_long_axis = self._object_long_axis(self.cucumber1)
        c2_long_axis = self._object_long_axis(self.cucumber2)
        left_grip_line_axis = self._grip_line_axis(self.left_agent)
        right_grip_line_axis = self._grip_line_axis(self.right_agent)
        lid_pos = self.lid.pose.p
        lid_handle_pos = self._lid_handle_pos()
        pot_pos = self.pot.pose.p

        left_reach = 1.0 - torch.tanh(4.0 * torch.linalg.norm(left_tcp - c1_pos, dim=1))
        right_reach = 1.0 - torch.tanh(4.0 * torch.linalg.norm(right_tcp - c2_pos, dim=1))
        lid_handle_reach = 1.0 - torch.tanh(5.0 * torch.linalg.norm(center_tcp - lid_handle_pos, dim=1))
        left_true_grasp = self.left_agent.is_grasping(self.cucumber1)
        right_true_grasp = self.right_agent.is_grasping(self.cucumber2)
        self.cucumber1_grasped = torch.logical_or(self.cucumber1_grasped, left_true_grasp)
        self.cucumber2_grasped = torch.logical_or(self.cucumber2_grasped, right_true_grasp)
        left_xy_align = 1.0 - torch.tanh(
            8.0 * torch.linalg.norm(left_tcp[:, :2] - c1_pos[:, :2], dim=1)
        )
        right_xy_align = 1.0 - torch.tanh(
            8.0 * torch.linalg.norm(right_tcp[:, :2] - c2_pos[:, :2], dim=1)
        )
        left_top_clear = 1.0 - torch.tanh(
            12.0 * torch.clamp(c1_pos[:, 2] + self.cucumber_top_grasp_z_offset - left_tcp[:, 2], min=0.0)
        )
        right_top_clear = 1.0 - torch.tanh(
            12.0 * torch.clamp(c2_pos[:, 2] + self.cucumber_top_grasp_z_offset - right_tcp[:, 2], min=0.0)
        )
        left_top_approach = left_xy_align * left_top_clear
        right_top_approach = right_xy_align * right_top_clear
        left_grip_perp = 1.0 - torch.abs(torch.sum(left_grip_line_axis * c1_long_axis, dim=1))
        right_grip_perp = 1.0 - torch.abs(torch.sum(right_grip_line_axis * c2_long_axis, dim=1))
        left_pregrasp = left_top_approach * left_grip_perp
        right_pregrasp = right_top_approach * right_grip_perp

        finger_mid = 0.5 * (center_left_finger + center_right_finger)
        finger_mid_align = 1.0 - torch.tanh(10.0 * torch.linalg.norm(finger_mid - lid_handle_pos, dim=1))
        left_handle_dist = torch.linalg.norm(center_left_finger - lid_handle_pos, dim=1)
        right_handle_dist = torch.linalg.norm(center_right_finger - lid_handle_pos, dim=1)
        finger_balance = torch.exp(-10.0 * torch.abs(left_handle_dist - right_handle_dist))
        finger_gap = torch.linalg.norm(center_left_finger - center_right_finger, dim=1)
        finger_gap_match = torch.exp(
            -80.0 * torch.square(finger_gap - self.lid_handle_grasp_gap)
        )
        physical_lid_grasp = self.center_agent.is_grasping(self.lid)
        true_grasp = physical_lid_grasp | self.lid_sticky
        lid_handle_grasp = (
            finger_mid_align
            * finger_balance
            * finger_gap_match
            * true_grasp.float()
        )
        lid_open_progress = torch.clamp(self._lid_qpos() / self.lid_open_qpos, min=0.0, max=1.0)
        lid_close_progress = 1.0 - torch.tanh(20.0 * torch.abs(self._lid_qpos()))
        stable_lid_grasp = true_grasp & (lid_handle_grasp > 0.12)
        self.lid_opened_with_grasp = torch.logical_or(
            self.lid_opened_with_grasp,
            stable_lid_grasp
            & (lid_open_progress > self.lid_open_stage_progress_thresh),
        )

        c1_transport = 1.0 - torch.tanh(6.0 * torch.linalg.norm(c1_pos[:, :2] - pot_pos[:, :2], dim=1))
        c2_transport = 1.0 - torch.tanh(6.0 * torch.linalg.norm(c2_pos[:, :2] - pot_pos[:, :2], dim=1))
        c1_ready = left_true_grasp | c1_in
        c2_ready = right_true_grasp | c2_in
        c1_ready_score = torch.where(
            c1_in,
            torch.ones_like(left_pregrasp),
            left_true_grasp.float() * (0.2 + 0.8 * left_pregrasp),
        )
        c2_ready_score = torch.where(
            c2_in,
            torch.ones_like(right_pregrasp),
            right_true_grasp.float() * (0.2 + 0.8 * right_pregrasp),
        )

        approach_z = self.pot_base_z + self.pot_height + self.cucumber_approach_z_offset
        c1_lift = 1.0 - torch.tanh(
            12.0 * torch.clamp(approach_z - torch.minimum(left_tcp[:, 2], c1_pos[:, 2]), min=0.0)
        )
        c2_lift = 1.0 - torch.tanh(
            12.0 * torch.clamp(approach_z - torch.minimum(right_tcp[:, 2], c2_pos[:, 2]), min=0.0)
        )
        _, c1_xy_in_area = self._in_xy_area(c1_pos, pot_pos, self.cucumber_approach_area_radius)
        _, c2_xy_in_area = self._in_xy_area(c2_pos, pot_pos, self.cucumber_approach_area_radius)
        c1_xy_to_area = 1.0 - torch.tanh(
            8.0
            * torch.clamp(
                torch.linalg.norm(c1_pos[:, :2] - pot_pos[:, :2], dim=1)
                - self.cucumber_approach_area_radius,
                min=0.0,
            )
        )
        c2_xy_to_area = 1.0 - torch.tanh(
            8.0
            * torch.clamp(
                torch.linalg.norm(c2_pos[:, :2] - pot_pos[:, :2], dim=1)
                - self.cucumber_approach_area_radius,
                min=0.0,
            )
        )
        c1_approach = c1_lift * torch.where(c1_xy_in_area, torch.ones_like(c1_xy_to_area), c1_xy_to_area)
        c2_approach = c2_lift * torch.where(c2_xy_in_area, torch.ones_like(c2_xy_to_area), c2_xy_to_area)
        insert_pos = pot_pos.clone()
        insert_pos[:, 2] = self.cucumber_insert_target_z
        c1_insert_progress = 1.0 - torch.tanh(10.0 * torch.linalg.norm(c1_pos - insert_pos, dim=1))
        c2_insert_progress = 1.0 - torch.tanh(10.0 * torch.linalg.norm(c2_pos - insert_pos, dim=1))
        pot_rim_z = self.pot_base_z + self.pot_height
        c1_release_xy_dist = torch.linalg.norm(c1_pos[:, :2] - pot_pos[:, :2], dim=1)
        c2_release_xy_dist = torch.linalg.norm(c2_pos[:, :2] - pot_pos[:, :2], dim=1)
        c1_release_xy_ok = c1_release_xy_dist <= self.cucumber_release_area_radius
        c2_release_xy_ok = c2_release_xy_dist <= self.cucumber_release_area_radius
        c1_release_z_ok = c1_pos[:, 2] <= self.cucumber_release_z_max
        c2_release_z_ok = c2_pos[:, 2] <= self.cucumber_release_z_max
        c1_release_ready = c1_release_xy_ok & c1_release_z_ok
        c2_release_ready = c2_release_xy_ok & c2_release_z_ok
        self.cucumber1_release_ready_steps = torch.where(
            c1_release_ready & left_true_grasp & (~self.cucumber1_released),
            self.cucumber1_release_ready_steps + 1,
            torch.zeros_like(self.cucumber1_release_ready_steps),
        )
        self.cucumber2_release_ready_steps = torch.where(
            c2_release_ready & right_true_grasp & (~self.cucumber2_released),
            self.cucumber2_release_ready_steps + 1,
            torch.zeros_like(self.cucumber2_release_ready_steps),
        )
        c1_late_release = (
            self.cucumber1_release_ready_steps > self.cucumber_release_grace_steps
        )
        c2_late_release = (
            self.cucumber2_release_ready_steps > self.cucumber_release_grace_steps
        )
        left_gripper_action = self._agent_gripper_action(action, 1)
        right_gripper_action = self._agent_gripper_action(action, 2)
        left_open_cmd = torch.clamp(left_gripper_action, min=0.0, max=1.0)
        right_open_cmd = torch.clamp(right_gripper_action, min=0.0, max=1.0)
        left_close_cmd = torch.clamp(-left_gripper_action, min=0.0, max=1.0)
        right_close_cmd = torch.clamp(-right_gripper_action, min=0.0, max=1.0)
        c1_release_action_gate = c1_release_ready & left_true_grasp & (~self.cucumber1_released)
        c2_release_action_gate = c2_release_ready & right_true_grasp & (~self.cucumber2_released)
        c1_transport = torch.where(c1_in, torch.ones_like(c1_transport), c1_transport)
        c2_transport = torch.where(c2_in, torch.ones_like(c2_transport), c2_transport)
        c1_approach = torch.where(c1_in, torch.ones_like(c1_approach), c1_approach)
        c2_approach = torch.where(c2_in, torch.ones_like(c2_approach), c2_approach)
        c1_insert_progress = torch.where(
            c1_in, torch.ones_like(c1_insert_progress), c1_insert_progress
        )
        c2_insert_progress = torch.where(
            c2_in, torch.ones_like(c2_insert_progress), c2_insert_progress
        )
        c1_insert_progress = torch.where(
            c1_release_ready, torch.ones_like(c1_insert_progress), c1_insert_progress
        )
        c2_insert_progress = torch.where(
            c2_release_ready, torch.ones_like(c2_insert_progress), c2_insert_progress
        )
        c1_release_now = c1_in & (~left_true_grasp)
        c2_release_now = c2_in & (~right_true_grasp)
        self.cucumber1_released = torch.logical_or(self.cucumber1_released, c1_release_now)
        self.cucumber2_released = torch.logical_or(self.cucumber2_released, c2_release_now)
        c1_release = self.cucumber1_released
        c2_release = self.cucumber2_released
        c1_transport = torch.where(c1_release, torch.ones_like(c1_transport), c1_transport)
        c2_transport = torch.where(c2_release, torch.ones_like(c2_transport), c2_transport)
        c1_approach = torch.where(c1_release, torch.ones_like(c1_approach), c1_approach)
        c2_approach = torch.where(c2_release, torch.ones_like(c2_approach), c2_approach)
        c1_insert_progress = torch.where(
            c1_release, torch.ones_like(c1_insert_progress), c1_insert_progress
        )
        c2_insert_progress = torch.where(
            c2_release, torch.ones_like(c2_insert_progress), c2_insert_progress
        )
        c1_retract_target, c2_retract_target = self._retract_target_xy()
        c1_tcp_xy_dist = torch.linalg.norm(left_tcp[:, :2] - c1_retract_target, dim=1)
        c2_tcp_xy_dist = torch.linalg.norm(right_tcp[:, :2] - c2_retract_target, dim=1)
        c1_retract_xy = 1.0 - torch.tanh(8.0 * c1_tcp_xy_dist)
        c2_retract_xy = 1.0 - torch.tanh(8.0 * c2_tcp_xy_dist)
        # safe_retract_z = pot_rim_z + 0.05
        # c1_retract_z = 1.0 - torch.tanh(
        #     12.0 * torch.clamp(safe_retract_z - left_tcp[:, 2], min=0.0)
        # )
        # c2_retract_z = 1.0 - torch.tanh(
        #     12.0 * torch.clamp(safe_retract_z - right_tcp[:, 2], min=0.0)
        # )
        # left_home_arm_qpos = self.left_home_qpos[..., : self._agent_arm_qpos(self.left_agent).shape[-1]]
        # right_home_arm_qpos = self.right_home_qpos[..., : self._agent_arm_qpos(self.right_agent).shape[-1]]
        # c1_home_reward = 1.0 - torch.tanh(
        #     4.0 * torch.linalg.norm(self._agent_arm_qpos(self.left_agent) - left_home_arm_qpos, dim=1)
        # )
        # c2_home_reward = 1.0 - torch.tanh(
        #     4.0 * torch.linalg.norm(self._agent_arm_qpos(self.right_agent) - right_home_arm_qpos, dim=1)
        # )
        c1_target_reached = c1_retract_xy > 0.90
        c2_target_reached = c2_retract_xy > 0.90
        c1_retract = c1_retract_xy
        c2_retract = c2_retract_xy
        c1_retract_done = c1_release & c1_target_reached
        c2_retract_done = c2_release & c2_target_reached
        self.cucumber1_retracted = torch.logical_or(self.cucumber1_retracted, c1_retract_done)
        self.cucumber2_retracted = torch.logical_or(self.cucumber2_retracted, c2_retract_done)
        cucumbers_placed = c1_release & c2_release
        cucumbers_retracted = self.cucumber1_retracted & self.cucumber2_retracted

        place_gate = self.lid_opened_with_grasp.float()
        insert_gate = self.lid_opened_with_grasp.float()
        lid_ready = self.lid_opened_with_grasp
        close_stage = lid_ready & cucumbers_placed & cucumbers_retracted
        lid_drop_penalty = (
            self.lid_opened_with_grasp.float()
            * (~close_stage).float()
            * (
                0.65 * (~true_grasp).float()
                + torch.clamp(self.lid_open_stage_progress_thresh - lid_open_progress, min=0.0)
            )
        )
        c1_invalid = self.cucumber1_released & (~c1_in)
        c2_invalid = self.cucumber2_released & (~c2_in)
        c1_dropped_before_pot = (
            self.cucumber1_grasped
            & (~left_true_grasp)
            & (~c1_in)
            & (~self.cucumber1_released)
        )
        c2_dropped_before_pot = (
            self.cucumber2_grasped
            & (~right_true_grasp)
            & (~c2_in)
            & (~self.cucumber2_released)
        )
        c1_fallen_or_far = (c1_pos[:, 2] < -0.03) | (
            torch.linalg.norm(c1_pos[:, :2] - pot_pos[:, :2], dim=1) > 0.75
        )
        c2_fallen_or_far = (c2_pos[:, 2] < -0.03) | (
            torch.linalg.norm(c2_pos[:, :2] - pot_pos[:, :2], dim=1) > 0.75
        )
        cucumber_invalid_penalty = (
            c1_invalid.float()
            + c2_invalid.float()
            + c1_dropped_before_pot.float()
            + c2_dropped_before_pot.float()
            + c1_fallen_or_far.float()
            + c2_fallen_or_far.float()
        )

        lid_stage_reward = 0.25 * lid_handle_reach
        lid_stage_reward += 0.35 * true_grasp.float() + 0.8 * lid_handle_grasp
        lid_stage_reward += 1.2 * stable_lid_grasp.float() * lid_open_progress
        lid_stage_reward -= (~self.lid_opened_with_grasp).float() * 0.6 * lid_open_progress
        lid_stage_reward = torch.where(
            self.lid_opened_with_grasp,
            torch.full_like(lid_stage_reward, self.lid_open_stage_bonus),
            lid_stage_reward,
        )
        lid_stage_reward -= self.lid_drop_penalty_coef * lid_drop_penalty

        c1_grasp_stage_reward = 0.15 * left_reach
        c1_grasp_stage_reward += 0.65 * left_top_approach
        c1_grasp_stage_reward += 0.75 * left_grip_perp * left_top_approach
        c1_grasp_stage_reward += 1.5 * left_true_grasp.float()
        c1_grasp_stage_reward += 0.9 * left_true_grasp.float() * (0.2 + 0.8 * left_pregrasp)
        c1_grasp_stage_reward -= 0.2 * left_true_grasp.float() * (1.0 - left_pregrasp)
        c2_grasp_stage_reward = 0.15 * right_reach
        c2_grasp_stage_reward += 0.65 * right_top_approach
        c2_grasp_stage_reward += 0.75 * right_grip_perp * right_top_approach
        c2_grasp_stage_reward += 1.5 * right_true_grasp.float()
        c2_grasp_stage_reward += 0.9 * right_true_grasp.float() * (0.2 + 0.8 * right_pregrasp)
        c2_grasp_stage_reward -= 0.2 * right_true_grasp.float() * (1.0 - right_pregrasp)

        c1_release_stage_reward = place_gate * (
            0.45 * c1_transport * c1_ready_score
            + 0.75 * c1_approach * c1_ready_score
        )
        c1_release_stage_reward += insert_gate * (
            0.8 * c1_insert_progress * c1_ready_score
            + 1.2 * c1_in.float() * c1_ready_score
            + 1.0 * c1_release_ready.float()
            + 2.2 * ((c1_in | c1_release_ready) & (~left_true_grasp)).float()
            + 1.5 * c1_release_action_gate.float() * left_open_cmd
        )
        c1_release_stage_reward -= insert_gate * (
            0.9 * c1_in.float() * left_true_grasp.float()
            + 0.25 * c1_release_ready.float() * left_true_grasp.float()
            + 0.5 * c1_release_action_gate.float() * left_close_cmd
            + 2.8 * c1_late_release.float() * left_true_grasp.float()
            + 1.5 * c1_late_release.float() * left_true_grasp.float() * left_close_cmd
        )
        c2_release_stage_reward = place_gate * (
            0.45 * c2_transport * c2_ready_score
            + 0.75 * c2_approach * c2_ready_score
        )
        c2_release_stage_reward += insert_gate * (
            0.8 * c2_insert_progress * c2_ready_score
            + 1.2 * c2_in.float() * c2_ready_score
            + 1.0 * c2_release_ready.float()
            + 2.2 * ((c2_in | c2_release_ready) & (~right_true_grasp)).float()
            + 1.5 * c2_release_action_gate.float() * right_open_cmd
        )
        c2_release_stage_reward -= insert_gate * (
            0.9 * c2_in.float() * right_true_grasp.float()
            + 0.25 * c2_release_ready.float() * right_true_grasp.float()
            + 0.5 * c2_release_action_gate.float() * right_close_cmd
            + 2.8 * c2_late_release.float() * right_true_grasp.float()
            + 1.5 * c2_late_release.float() * right_true_grasp.float() * right_close_cmd
        )
        c1_retract_stage_reward = 7.0 * c1_release.float() * c1_retract
        c1_retract_stage_reward -= (
            1.2
            * c1_release.float()
            * (~self.cucumber1_retracted).float()
            * torch.clamp(self.cucumber_release_area_radius - c1_tcp_xy_dist, min=0.0)
            / self.cucumber_release_area_radius
        )
        c2_retract_stage_reward = 7.0 * c2_release.float() * c2_retract
        c2_retract_stage_reward -= (
            1.2
            * c2_release.float()
            * (~self.cucumber2_retracted).float()
            * torch.clamp(self.cucumber_release_area_radius - c2_tcp_xy_dist, min=0.0)
            / self.cucumber_release_area_radius
        )
        c1_partner_progress = (
            0.35 * c1_transport * c1_ready_score
            + 0.55 * c1_approach * c1_ready_score
            + 0.75 * c1_insert_progress * c1_ready_score
            + 0.90 * c1_release_ready.float()
        )
        c2_partner_progress = (
            0.35 * c2_transport * c2_ready_score
            + 0.55 * c2_approach * c2_ready_score
            + 0.75 * c2_insert_progress * c2_ready_score
            + 0.90 * c2_release_ready.float()
        )
        partner_prompt_reward = (
            (c2_release & (~c1_release)).float() * c1_partner_progress
            + (c1_release & (~c2_release)).float() * c2_partner_progress
        ) * self.partner_release_prompt_bonus

        zero = torch.zeros_like(lid_stage_reward)
        c1_pipeline_reward = c1_grasp_stage_reward
        c1_pipeline_reward = torch.where(
            self.cucumber1_grasped,
            torch.full_like(c1_pipeline_reward, self.cucumber_grasp_stage_bonus) + c1_release_stage_reward,
            c1_pipeline_reward,
        )
        c1_pipeline_reward = torch.where(
            c1_release,
            torch.full_like(c1_pipeline_reward, self.cucumber_release_stage_bonus) + c1_retract_stage_reward,
            c1_pipeline_reward,
        )
        c1_pipeline_reward = torch.where(
            self.cucumber1_retracted,
            torch.full_like(c1_pipeline_reward, self.cucumber_retract_stage_bonus),
            c1_pipeline_reward,
        )

        c2_pipeline_reward = c2_grasp_stage_reward
        c2_pipeline_reward = torch.where(
            self.cucumber2_grasped,
            torch.full_like(c2_pipeline_reward, self.cucumber_grasp_stage_bonus) + c2_release_stage_reward,
            c2_pipeline_reward,
        )
        c2_pipeline_reward = torch.where(
            c2_release,
            torch.full_like(c2_pipeline_reward, self.cucumber_release_stage_bonus) + c2_retract_stage_reward,
            c2_pipeline_reward,
        )
        c2_pipeline_reward = torch.where(
            self.cucumber2_retracted,
            torch.full_like(c2_pipeline_reward, self.cucumber_retract_stage_bonus),
            c2_pipeline_reward,
        )

        reward = lid_stage_reward
        reward += c1_pipeline_reward + c2_pipeline_reward
        reward += lid_ready.float() * partner_prompt_reward
        reward += close_stage.float() * (0.8 * lid_handle_reach + 1.2 * lid_close_progress)
        reward += close_stage.float() * info["lid_closed"].float()
        reward -= (~close_stage).float() * 0.4 * info["lid_closed"].float()
        reward -= self.cucumber_invalid_penalty_coef * cucumber_invalid_penalty
        reward += 5.0 * info["success"].float()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 80.0
