from typing import Any, Tuple

import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG


@register_env("TwoRobotPickCube-v2", max_episode_steps=100)
class TwoRobotPickCube(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up a red cube and lift it to a goal location. There are two robots in this task and the
    goal location is out of reach of the left robot while the cube is out of reach of the right robot, thus the two robots must work together
    to move the cube to the goal.

    **Randomizations:**
    - cube has its z-axis rotation randomized
    - cube has its xy positions on top of the table scene randomized such that it is in within reach of the left robot but not the right.
    - the target goal position (marked by a green sphere) of the cube is randomized such that it is within reach of the right robot but not the left.


    **Success Conditions:**
    - red cube is at the goal location
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[Panda, Panda]]
    cube_half_size = 0.02
    goal_thresh = 0.03
    RUNTIME_RANDOMIZATION_KEYS = (
        "robot_init_qpos_noise",
        "ambient_light_temperature",
        "ambient_light_intensity",
        "directional_light_temperature",
        "directional_light_intensity",
        "camera_y_rotate",
        "camera_eye_jitter",
        "camera_target_jitter",
        "object_spawn_x_abs",
        "object_spawn_y_min",
        "object_spawn_y_max",
        "goal_spawn_x_abs",
        "goal_spawn_y_min",
        "goal_spawn_y_max",
        "goal_spawn_z_max_delta",
        "shadow_scale",
    )

    @staticmethod
    def _allocate_env_counts(num_envs: int, ratios):
        ratios = np.asarray(ratios, dtype=np.float64)
        if np.any(ratios < 0):
            raise ValueError(f"Mixed env ratios must be non-negative, got {ratios.tolist()}")
        if ratios.sum() <= 0:
            raise ValueError("At least one mixed env ratio must be positive")

        normalized = ratios / ratios.sum()
        raw_counts = normalized * num_envs
        counts = np.floor(raw_counts).astype(int)
        remainder = int(num_envs - counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw_counts - counts))
            for idx in order[:remainder]:
                counts[idx] += 1

        positive_ratio_indices = [idx for idx, ratio in enumerate(normalized) if ratio > 0]
        if num_envs >= len(positive_ratio_indices):
            zero_positive_indices = [idx for idx in positive_ratio_indices if counts[idx] == 0]
            for idx in zero_positive_indices:
                donor = int(np.argmax(counts))
                if counts[donor] <= 1:
                    break
                counts[donor] -= 1
                counts[idx] += 1

        return counts.tolist()

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        object_scale=1.,
        ambient_light_temperature=-1,
        light_intensity=-1,
        mass_scale=1.,
        object_color="red",
        object_type="cube",
        camera_y_rotate=0,
        goal_thresh=None,
        enable_domain_randomization=False,
        domain_randomization_level="mild",
        enable_mixed_domain_randomization=False,
        mixed_domain_randomization_clear_ratio=0.5,
        mixed_domain_randomization_mild_ratio=0.3,
        mixed_domain_randomization_hard_ratio=0.2,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.object_size_scale = object_scale
        self.ambient_light_temperature = ambient_light_temperature
        self.light_intensity = light_intensity
        self.mass_scale = mass_scale
        self.object_color = object_color
        self.object_type = object_type
        self.camera_y_rotate = camera_y_rotate
        if goal_thresh is not None:
            self.goal_thresh = float(goal_thresh)
        self.enable_domain_randomization = bool(enable_domain_randomization)
        self.domain_randomization_level = domain_randomization_level
        self.enable_mixed_domain_randomization = bool(enable_mixed_domain_randomization)
        self.mixed_domain_randomization_clear_ratio = float(mixed_domain_randomization_clear_ratio)
        self.mixed_domain_randomization_mild_ratio = float(mixed_domain_randomization_mild_ratio)
        self.mixed_domain_randomization_hard_ratio = float(mixed_domain_randomization_hard_ratio)
        self.num_envs = kwargs.get("num_envs", 1)
        self.per_env_settings = [{} for _ in range(self.num_envs)]
        self.sampled_domain_randomization = None
        self._direct_lights = [None for _ in range(self.num_envs)]
        self.objects = [None] * self.num_envs
        self.goal_sites = [None] * self.num_envs
        self.base_object_masses = [None] * self.num_envs
        self.cube = None
        self.goal_site = None
        self.cam_mount = None
        self.rngs = None
        self.left_init_qpos = None
        self.env_levels = ["clear"] * self.num_envs
        self.left_action_phase_state = None
        self.right_action_phase_state = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _initialize_own_rngs(self):
        derived_seeds = self._batched_episode_rng.randint(0, 2**31 - 1)
        self.rngs = BatchedRNG.from_seeds(
            derived_seeds.tolist(),
            backend=self._batched_rng_backend,
        )

    def _sample_one_domain_randomization_profile(self, rng, level_override=None):
        level = str(self.domain_randomization_level if level_override is None else level_override).lower()
        if level not in {"mild", "hard"}:
            raise ValueError(
                f"Unsupported domain_randomization_level={self.domain_randomization_level}. "
                "Expected one of: mild, hard"
            )

        if level == "mild":
            object_scale = float(rng.uniform(0.7, 0.95, size=1)[0])
            profile = {
                "level": level,
                "robot_init_qpos_noise": float(rng.uniform(0.02, 0.05, size=1)[0]),
                "object_scale": object_scale,
                "ambient_light_temperature": float(rng.uniform(0.75, 1.2, size=1)[0]),
                "ambient_light_intensity": float(rng.uniform(0.7, 1.15, size=1)[0]),
                "directional_light_temperature": float(rng.uniform(0.8, 1.2, size=1)[0]),
                "directional_light_intensity": float(rng.uniform(0.7, 1.2, size=1)[0]),
                "mass_scale": float(rng.uniform(0.7, 1.35, size=1)[0]),
                "object_color": str(rng.choice(["red", "blue", "yellow"])),
                "object_type": str(rng.choice(["cube", "sphere"])),
                "camera_y_rotate": float(rng.uniform(-14.0, 14.0, size=1)[0]),
                "camera_eye_jitter": [
                    float(rng.uniform(-0.08, 0.08, size=1)[0]),
                    float(rng.uniform(-0.08, 0.08, size=1)[0]),
                    float(rng.uniform(-0.05, 0.05, size=1)[0]),
                ],
                "camera_target_jitter": [
                    float(rng.uniform(-0.05, 0.05, size=1)[0]),
                    float(rng.uniform(-0.05, 0.05, size=1)[0]),
                    float(rng.uniform(-0.04, 0.04, size=1)[0]),
                ],
                "object_spawn_x_abs": float(rng.uniform(0.05, 0.08, size=1)[0]),
                "object_spawn_y_min": float(rng.uniform(-0.22, -0.16, size=1)[0]),
                "object_spawn_y_max": float(rng.uniform(-0.12, -0.08, size=1)[0]),
                "goal_spawn_x_abs": float(rng.uniform(0.05, 0.08, size=1)[0]),
                "goal_spawn_y_min": float(rng.uniform(0.08, 0.12, size=1)[0]),
                "goal_spawn_y_max": float(rng.uniform(0.18, 0.24, size=1)[0]),
                "goal_spawn_z_max_delta": float(rng.uniform(0.25, 0.38, size=1)[0]),
                "shadow_scale": float(rng.uniform(4.0, 6.5, size=1)[0]),
            }
        else:
            object_scale = float(rng.uniform(0.5, 1.0, size=1)[0])
            profile = {
                "level": level,
                "robot_init_qpos_noise": float(rng.uniform(0.03, 0.07, size=1)[0]),
                "object_scale": object_scale,
                "ambient_light_temperature": float(rng.uniform(0.55, 1.5, size=1)[0]),
                "ambient_light_intensity": float(rng.uniform(0.45, 1.35, size=1)[0]),
                "directional_light_temperature": float(rng.uniform(0.6, 1.5, size=1)[0]),
                "directional_light_intensity": float(rng.uniform(0.45, 1.4, size=1)[0]),
                "mass_scale": float(rng.uniform(0.5, 1.8, size=1)[0]),
                "object_color": str(rng.choice(["red", "blue", "green", "yellow", "white"])),
                "object_type": str(rng.choice(["cube", "sphere", "cylinder"])),
                "camera_y_rotate": float(rng.uniform(-24.0, 24.0, size=1)[0]),
                "camera_eye_jitter": [
                    float(rng.uniform(-0.14, 0.14, size=1)[0]),
                    float(rng.uniform(-0.12, 0.12, size=1)[0]),
                    float(rng.uniform(-0.10, 0.10, size=1)[0]),
                ],
                "camera_target_jitter": [
                    float(rng.uniform(-0.08, 0.08, size=1)[0]),
                    float(rng.uniform(-0.08, 0.08, size=1)[0]),
                    float(rng.uniform(-0.08, 0.08, size=1)[0]),
                ],
                "object_spawn_x_abs": float(rng.uniform(0.06, 0.10, size=1)[0]),
                "object_spawn_y_min": float(rng.uniform(-0.24, -0.18, size=1)[0]),
                "object_spawn_y_max": float(rng.uniform(-0.12, -0.06, size=1)[0]),
                "goal_spawn_x_abs": float(rng.uniform(0.06, 0.10, size=1)[0]),
                "goal_spawn_y_min": float(rng.uniform(0.06, 0.12, size=1)[0]),
                "goal_spawn_y_max": float(rng.uniform(0.20, 0.26, size=1)[0]),
                "goal_spawn_z_max_delta": float(rng.uniform(0.30, 0.45, size=1)[0]),
                "shadow_scale": float(rng.uniform(3.0, 7.5, size=1)[0]),
            }
        profile["half_height"] = self.cube_half_size * object_scale
        return profile

    def _build_fixed_domain_randomization_profile(self):
        return {
            "level": "fixed",
            "robot_init_qpos_noise": float(self.robot_init_qpos_noise),
            "object_scale": float(self.object_size_scale),
            "ambient_light_temperature": float(self.ambient_light_temperature),
            "ambient_light_intensity": float(1.0 if self.light_intensity == -1 else self.light_intensity),
            "directional_light_temperature": float(self.ambient_light_temperature),
            "directional_light_intensity": float(1.0 if self.light_intensity == -1 else self.light_intensity),
            "mass_scale": float(self.mass_scale),
            "object_color": str(self.object_color),
            "object_type": str(self.object_type),
            "camera_y_rotate": float(self.camera_y_rotate),
            "camera_eye_jitter": [0.0, 0.0, 0.0],
            "camera_target_jitter": [0.0, 0.0, 0.0],
            "object_spawn_x_abs": 0.05,
            "object_spawn_y_min": -0.20,
            "object_spawn_y_max": -0.10,
            "goal_spawn_x_abs": 0.05,
            "goal_spawn_y_min": 0.10,
            "goal_spawn_y_max": 0.20,
            "goal_spawn_z_max_delta": 0.30,
            "shadow_scale": 5.0,
            "half_height": self.cube_half_size * float(self.object_size_scale),
        }

    def _build_clear_domain_randomization_profile(self):
        profile = self._build_fixed_domain_randomization_profile()
        profile["level"] = "clear"
        return profile

    @staticmethod
    def _get_rgb_by_temperature(temperature_kelvin: float):
        r = min(1.0, max(0.0, 1.0 - (6500 - temperature_kelvin) / 10000))
        g = min(1.0, max(0.0, 1.0 - abs(5000 - temperature_kelvin) / 20000))
        b = min(1.0, max(0.0, 1.0 - (temperature_kelvin - 3000) / 10000))
        return [r, g, b]

    def _compute_light_colors(self, cfg: dict):
        ambient_rgb = [0.3, 0.3, 0.3]
        directional_rgb = [1.0, 1.0, 1.0]
        if cfg["ambient_light_temperature"] != -1:
            ambient_temp_rgb = self._get_rgb_by_temperature(6500 * cfg["ambient_light_temperature"])
            ambient_rgb = [c * ambient_temp_rgb[i] for i, c in enumerate(ambient_rgb)]
        ambient_rgb = [c * cfg["ambient_light_intensity"] for c in ambient_rgb]

        if cfg["directional_light_temperature"] != -1:
            directional_rgb = self._get_rgb_by_temperature(6500 * cfg["directional_light_temperature"])
        directional_rgb = [c * cfg["directional_light_intensity"] for c in directional_rgb]
        return ambient_rgb, directional_rgb

    def _refresh_runtime_randomization(self, env_ids):
        if not (self.enable_mixed_domain_randomization or self.enable_domain_randomization):
            return
        for env_id in env_ids:
            level = self.env_levels[env_id]
            if level == "clear":
                continue
            fresh_profile = self._sample_one_domain_randomization_profile(
                self.rngs[env_id],
                level_override=level,
            )
            for key in self.RUNTIME_RANDOMIZATION_KEYS:
                self.per_env_settings[env_id][key] = fresh_profile[key]

    def _apply_runtime_randomization(self, env_idx: torch.Tensor):
        for env_id in env_idx.tolist():
            cfg = self.per_env_settings[env_id]
            ambient_rgb, directional_rgb = self._compute_light_colors(cfg)
            self.scene.sub_scenes[env_id].ambient_light = ambient_rgb
            light = self._direct_lights[env_id]
            if light is not None:
                light.set_color(directional_rgb)
                light.set_shadow_half_size(cfg["shadow_scale"])

    def _initialize_mixed_per_env_settings(self):
        self._initialize_own_rngs()
        profile_names = ["clear", "mild", "hard"]
        counts = self._allocate_env_counts(
            self.num_envs,
            [
                self.mixed_domain_randomization_clear_ratio,
                self.mixed_domain_randomization_mild_ratio,
                self.mixed_domain_randomization_hard_ratio,
            ],
        )

        perm_seed = int(np.asarray(self._batched_episode_rng.randint(0, 2**31 - 1)).reshape(-1)[0])
        env_order = np.random.default_rng(perm_seed).permutation(self.num_envs).tolist()

        env_levels = [None] * self.num_envs
        start = 0
        for profile_name, count in zip(profile_names, counts):
            for env_id in env_order[start:start + count]:
                env_levels[env_id] = profile_name
            start += count

        sampled_profiles = []
        for env_id in range(self.num_envs):
            level = env_levels[env_id]
            self.env_levels[env_id] = level
            if level == "clear":
                profile = self._build_clear_domain_randomization_profile()
            else:
                profile = self._sample_one_domain_randomization_profile(self.rngs[env_id], level_override=level)
            self.per_env_settings[env_id] = profile
            sampled_profiles.append(profile)

        self.sampled_domain_randomization = sampled_profiles
        counts_summary = ", ".join(
            f"{profile_name}={count}" for profile_name, count in zip(profile_names, counts)
        )
        preview_count = min(4, len(sampled_profiles))
        preview = " | ".join(
            f"env{idx}={sampled_profiles[idx]}" for idx in range(preview_count)
        )
        print(
            f"[TwoRobotPickCube] mixed per-env domain randomization "
            f"({self.num_envs} envs, {counts_summary}): {preview}"
        )

    def _initialize_per_env_settings(self):
        if self.enable_mixed_domain_randomization:
            self._initialize_mixed_per_env_settings()
        elif self.enable_domain_randomization:
            self._initialize_own_rngs()
            sampled_profiles = []
            for env_id in range(self.num_envs):
                self.env_levels[env_id] = self.domain_randomization_level
                profile = self._sample_one_domain_randomization_profile(self.rngs[env_id])
                self.per_env_settings[env_id] = profile
                sampled_profiles.append(profile)
            self.sampled_domain_randomization = sampled_profiles
            preview_count = min(4, len(sampled_profiles))
            preview = " | ".join(
                f"env{idx}={sampled_profiles[idx]}" for idx in range(preview_count)
            )
            print(
                f"[TwoRobotPickCube] sampled per-env domain randomization "
                f"({self.num_envs} envs): {preview}"
            )
        else:
            fixed_profile = self._build_fixed_domain_randomization_profile()
            self.per_env_settings = [dict(fixed_profile) for _ in range(self.num_envs)]
            self.sampled_domain_randomization = self.per_env_settings

    def _color_name_to_rgba(self, color_name: str):
        if color_name == "red":
            return [1, 0, 0, 1]
        if color_name == "blue":
            return [0, 0, 1, 1]
        if color_name == "green":
            return [0, 1, 0, 1]
        if color_name == "yellow":
            return [1, 1, 0, 1]
        if color_name == "white":
            return [1, 1, 1, 1]
        raise ValueError(f"Unsupported object_color={color_name}")

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
        return [CameraConfig("base_camera", sapien.Pose(), 128, 128, np.pi / 2, 0.01, 100, mount=self.cam_mount)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.4, 0.8, 0.75], [0.0, 0.1, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )

    def _load_scene(self, options: dict):
        self._initialize_per_env_settings()
        builder_mount = self.scene.create_actor_builder()
        builder_mount.set_initial_pose(sapien.Pose())
        self.cam_mount = builder_mount.build_kinematic("camera_mount")
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=0.0
        )
        self.table_scene.build()
        for env_id in range(self.num_envs):
            cfg = self.per_env_settings[env_id]
            color = self._color_name_to_rgba(cfg["object_color"])
            object_scale = cfg["object_scale"]
            half_height = cfg["half_height"]
            object_type = cfg["object_type"]

            if object_type == "cube":
                obj = actors.build_cube(
                    self.scene,
                    half_size=self.cube_half_size * object_scale,
                    color=color,
                    scene_idxs=[env_id],
                    name=f"cube_{env_id}",
                    initial_pose=sapien.Pose(p=[0, 0, half_height]),
                )
            elif object_type == "sphere":
                obj = actors.build_sphere(
                    self.scene,
                    radius=self.cube_half_size * object_scale,
                    color=color,
                    scene_idxs=[env_id],
                    name=f"sphere_{env_id}",
                    initial_pose=sapien.Pose(p=[0, 0, half_height]),
                )
            elif object_type == "cylinder":
                obj = actors.build_cylinder(
                    self.scene,
                    radius=self.cube_half_size * object_scale,
                    half_length=self.cube_half_size * object_scale,
                    color=color,
                    scene_idxs=[env_id],
                    name=f"cylinder_{env_id}",
                    initial_pose=sapien.Pose(p=[0, 0, half_height]),
                )
            else:
                raise ValueError(f"Unsupported object_type={object_type}")

            self.base_object_masses[env_id] = float(obj.mass[0])
            obj.mass = self.base_object_masses[env_id] * cfg["mass_scale"]
            self.objects[env_id] = obj

        self.cube = Actor.merge(self.objects, name="cube")

        # ---- build goal ----
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            # color=[0, 1, 0, 0.01],
            body_type="kinematic",
            name="goal_site",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)
        

    def _load_lighting(self, options: dict):
        shadow = self.enable_shadow
        for env_id in range(self.num_envs):
            cfg = self.per_env_settings[env_id]
            sub_scene = self.scene.sub_scenes[env_id]
            ambient_light_rgb, directional_light_rgb = self._compute_light_colors(cfg)
            sub_scene.ambient_light = ambient_light_rgb
            self._direct_lights[env_id] = sub_scene.add_directional_light(
                [1, 1, -1],
                directional_light_rgb,
                shadow=shadow,
                shadow_scale=cfg["shadow_scale"],
                shadow_map_size=2048,
            )

    def _get_camera_pose(self, env_idx: torch.Tensor):
        def add_pitch_to_eye(eye, target, delta_pitch):
            delta_pitch = np.deg2rad(delta_pitch)
            eye = np.asarray(eye)
            target = np.asarray(target)
            v = eye - target
            c = np.cos(delta_pitch)
            s = np.sin(delta_pitch)
            rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            return (target + rot @ v).tolist()

        eye = np.asarray([1.0, 0, 0.75], dtype=np.float32)
        target = np.asarray([0.0, 0.0, 0.25], dtype=np.float32)
        raw_poses = []
        for env_id in env_idx.tolist():
            cfg = self.per_env_settings[env_id]
            eye_offset = np.asarray(cfg["camera_eye_jitter"], dtype=np.float32)
            target_offset = np.asarray(cfg["camera_target_jitter"], dtype=np.float32)
            rotated_eye = add_pitch_to_eye(
                eye + eye_offset,
                target + target_offset,
                cfg["camera_y_rotate"],
            )
            target_pose = (target + target_offset).tolist()
            pose = sapien_utils.look_at(rotated_eye, target_pose)
            pose_p = torch.as_tensor(pose.p, dtype=torch.float32, device=self.device).reshape(-1)
            pose_q = torch.as_tensor(pose.q, dtype=torch.float32, device=self.device).reshape(-1)
            raw_poses.append(
                torch.cat([pose_p, pose_q], dim=0)
            )
        return Pose.create(torch.stack(raw_poses, dim=0))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            if (
                self.left_action_phase_state is None
                or self.left_action_phase_state.shape[0] != self.num_envs
            ):
                self.left_action_phase_state = torch.zeros(
                    (self.num_envs,), device=self.device, dtype=torch.long
                )
            if (
                self.right_action_phase_state is None
                or self.right_action_phase_state.shape[0] != self.num_envs
            ):
                self.right_action_phase_state = torch.zeros(
                    (self.num_envs,), device=self.device, dtype=torch.long
                )
            self.left_action_phase_state[env_idx] = 0
            self.right_action_phase_state[env_idx] = 0
            self.table_scene.initialize(env_idx)
            self._refresh_runtime_randomization(env_idx.tolist())
            self._apply_runtime_randomization(env_idx)
            left_qpos = self.left_agent.robot.get_qpos().clone()
            right_qpos = self.right_agent.robot.get_qpos().clone()
            noise_scales = torch.tensor(
                [self.per_env_settings[i]["robot_init_qpos_noise"] for i in env_idx.tolist()],
                device=self.device,
                dtype=left_qpos.dtype,
            ).unsqueeze(-1)
            left_noise = (torch.rand((b, left_qpos.shape[1]), device=self.device) * 2 - 1) * noise_scales
            right_noise = (torch.rand((b, right_qpos.shape[1]), device=self.device) * 2 - 1) * noise_scales
            left_qpos[env_idx] = left_qpos[env_idx] + left_noise
            right_qpos[env_idx] = right_qpos[env_idx] + right_noise
            self.left_agent.robot.set_qpos(left_qpos[env_idx])
            self.right_agent.robot.set_qpos(right_qpos[env_idx])
            self.left_init_qpos = self.left_agent.robot.get_qpos().clone()

            half_heights = torch.tensor(
                [self.per_env_settings[i]["half_height"] for i in env_idx.tolist()],
                device=self.device,
                dtype=torch.float32,
            )
            xyz = torch.zeros((b, 3), device=self.device)
            object_x_abs = torch.tensor(
                [self.per_env_settings[i]["object_spawn_x_abs"] for i in env_idx.tolist()],
                device=self.device,
                dtype=torch.float32,
            )
            object_y_min = torch.tensor(
                [self.per_env_settings[i]["object_spawn_y_min"] for i in env_idx.tolist()],
                device=self.device,
                dtype=torch.float32,
            )
            object_y_max = torch.tensor(
                [self.per_env_settings[i]["object_spawn_y_max"] for i in env_idx.tolist()],
                device=self.device,
                dtype=torch.float32,
            )
            xyz[:, 0] = (torch.rand((b,), device=self.device) * 2 - 1) * object_x_abs
            xyz[:, 1] = object_y_min + torch.rand((b,), device=self.device) * (object_y_max - object_y_min)
            xyz[:, 2] = half_heights
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, device=self.device)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3), device=self.device)
            goal_x_abs = torch.tensor(
                [self.per_env_settings[i]["goal_spawn_x_abs"] for i in env_idx.tolist()],
                device=self.device,
                dtype=torch.float32,
            )
            goal_y_min = torch.tensor(
                [self.per_env_settings[i]["goal_spawn_y_min"] for i in env_idx.tolist()],
                device=self.device,
                dtype=torch.float32,
            )
            goal_y_max = torch.tensor(
                [self.per_env_settings[i]["goal_spawn_y_max"] for i in env_idx.tolist()],
                device=self.device,
                dtype=torch.float32,
            )
            goal_z_max_delta = torch.tensor(
                [self.per_env_settings[i]["goal_spawn_z_max_delta"] for i in env_idx.tolist()],
                device=self.device,
                dtype=torch.float32,
            )
            goal_xyz[:, 0] = (torch.rand((b,), device=self.device) * 2 - 1) * goal_x_abs
            goal_xyz[:, 1] = goal_y_min + torch.rand((b,), device=self.device) * (goal_y_max - goal_y_min)
            goal_xyz[:, 2] = torch.rand((b,), device=self.device) * goal_z_max_delta + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            self.cam_mount.set_pose(self._get_camera_pose(env_idx))

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_right_arm_static = self.right_agent.is_static(0.6)
        is_fall_off = self.cube.pose.p[:, 2] < 0
        return {
            "success": torch.logical_and(is_obj_placed, is_right_arm_static),
            "fail": is_fall_off,
            "is_obj_placed": is_obj_placed,
            "is_right_arm_static": is_right_arm_static,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        # <= 50 stage 0, > 50 stage 1
        stage = self._elapsed_steps // (100 // 2)
        if "state" in self.obs_mode:
            left_tcp_to_cube = self.cube.pose.p - self.left_agent.tcp.pose.p
            right_tcp_to_cube = self.cube.pose.p - self.right_agent.tcp.pose.p
            left_tcp_to_cube_dist = torch.linalg.norm(left_tcp_to_cube, axis=1)
            right_tcp_to_cube_dist = torch.linalg.norm(right_tcp_to_cube, axis=1)
            cube_at_other_side = self.cube.pose.p[:, 1] >= 0.0
            right_is_grasping = self.right_agent.is_grasping(self.cube)

            if (
                self.left_action_phase_state is None
                or self.left_action_phase_state.shape[0] != self.num_envs
            ):
                self.left_action_phase_state = torch.zeros(
                    (self.num_envs,), device=self.device, dtype=torch.long
                )
            if (
                self.right_action_phase_state is None
                or self.right_action_phase_state.shape[0] != self.num_envs
            ):
                self.right_action_phase_state = torch.zeros(
                    (self.num_envs,), device=self.device, dtype=torch.long
                )

            left_action_phase = self.left_action_phase_state.clone()
            left_action_phase[
                torch.logical_and(
                    left_action_phase == 0,
                    torch.logical_and(~cube_at_other_side, left_tcp_to_cube_dist <= 0.08),
                )
            ] = 1
            left_action_phase[
                torch.logical_and(left_action_phase <= 1, cube_at_other_side)
            ] = 2

            right_action_phase = self.right_action_phase_state.clone()
            right_action_phase[
                torch.logical_and(
                    right_action_phase == 0,
                    torch.logical_and(~right_is_grasping, right_tcp_to_cube_dist <= 0.08),
                )
            ] = 1
            right_action_phase[
                torch.logical_and(right_action_phase <= 1, right_is_grasping)
            ] = 2

            self.left_action_phase_state = left_action_phase
            self.right_action_phase_state = right_action_phase

            action_match = torch.logical_and(left_action_phase == 0, right_action_phase == 0)
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                left_arm_tcp_to_cube_pos=left_tcp_to_cube,
                right_arm_tcp_to_cube_pos=right_tcp_to_cube,
                cube_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
                stage=stage,
                left_action_phase=left_action_phase,
                right_action_phase=right_action_phase,
                action_match=action_match,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Stage 1: Reach and push cube to be near other robot
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.left_agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)

        # set a sub_goal here where we want the cube to first be pushed to close to the right arm robot
        # by moving cube past y = 0.05
        cube_to_other_side_reward = 1 - torch.tanh(
            5
            * (
                torch.max(
                    0.05 - self.cube.pose.p[:, 1], torch.zeros_like(reaching_reward)
                )
            )
        )
        reward = (reaching_reward + cube_to_other_side_reward) / 2

        # stage 1 passes if cube is near a sub-goal
        cube_at_other_side = self.cube.pose.p[:, 1] >= 0.0

        # Stage 2: reach and grasp cube with right robot and make left robot leave space
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.right_agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        stage_2_reward = reaching_reward

        # condition for good grasp: both fingers are at the same height and open
        self.right_agent: Panda
        right_tip_1_height = self.right_agent.finger1_link.pose.p[:, 2]
        right_tip_2_height = self.right_agent.finger2_link.pose.p[:, 2]
        tip_height_reward = 1 - torch.tanh(
            5 * torch.abs(right_tip_1_height - right_tip_2_height)
        )
        tip_width_reward = 1 - torch.tanh(
            5
            * torch.abs(
                torch.linalg.norm(
                    self.right_agent.finger1_link.pose.p
                    - self.right_agent.finger2_link.pose.p,
                    axis=1,
                )
                - 0.07
            )
        )
        tip_reward = (tip_height_reward + tip_width_reward) / 2
        stage_2_reward += tip_reward

        # make left arm move as close as possible to the y=-0.2 line
        left_arm_leave_reward = 1 - torch.tanh(
            5 * (self.left_agent.tcp.pose.p[:, 1] + 0.2).abs()
        )
        stage_2_reward += left_arm_leave_reward

        # stage 2 passes if cube is grasped
        is_grasped = self.right_agent.is_grasping(self.cube)
        stage_2_reward += 2 * is_grasped

        reward[cube_at_other_side] = 2 + stage_2_reward[cube_at_other_side]

        # Stage 3: bring cube towards goal
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        stage_3_reward = 2 * place_reward

        # return left arm to original position
        left_qpos_reward = 1 - torch.tanh(
            torch.linalg.norm(
                self.left_agent.robot.get_qpos() - self.left_init_qpos, axis=1
            )
        )
        stage_3_reward += left_qpos_reward

        reward[is_grasped] = 8 + stage_3_reward[is_grasped]

        # Stage 4 bonus only kicks in once the grasped object-carrier is within
        # the same radius used by the goal-success sphere.
        is_obj_near = torch.logical_and(obj_to_goal_dist < self.goal_thresh, is_grasped)
        # Stage 4: reuse same reward as stage 3 but stronger incentive
        reward[is_obj_near] = 12 + 2 * stage_3_reward[is_obj_near]

        # stage 4 passes if object is placed
        is_obj_placed = info["is_obj_placed"]

        # Stage 5: keep robot static at the goal
        right_static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.right_agent.robot.get_qvel()[..., :-2], axis=1)
        )
        left_static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.left_agent.robot.get_qvel()[..., :-2], axis=1)
        )
        static_reward = (right_static_reward + left_static_reward) / 2

        reward[is_obj_placed] = 19 + static_reward[is_obj_placed]

        reward[info["success"]] = 21
        reward[info['fail']] -= 2

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 21
