from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from mani_skill.agents.robots.unitree_g1.g1_upper_body import (
    UnitreeG1UpperBodyWithHeadCamera,
)
from mani_skill.envs.tasks.humanoid import (
    HumanoidPlaceAppleInBowl as _BaseHumanoidPlaceAppleInBowl,
)
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig

HUMAN_ENV_NAME = "UnitreeG1LiftApple-v1"
HUMAN_ENV_IDS: list[str] = [HUMAN_ENV_NAME]
HUMAN_VARIANT_ENV_IDS: list[str] = []
HUMAN_VARIANT_CLASSES: dict[str, type] = {}

_BASE_LIGHT_COLOR_TEMPERATURE_K = 6500.0
_DEFAULT_AMBIENT_LIGHT = np.array([0.3, 0.3, 0.3], dtype=np.float32)
_DEFAULT_DIRECTIONAL_LIGHT = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def _clamp_rgb(color: np.ndarray) -> np.ndarray:
    return np.clip(color.astype(np.float32), 0.0, 1.0)


def _kelvin_to_rgb(kelvin: float) -> np.ndarray:
    temperature = np.clip(kelvin, 1000.0, 40000.0) / 100.0

    if temperature <= 66.0:
        red = 255.0
        green = 99.4708025861 * np.log(temperature) - 161.1195681661
        if temperature <= 19.0:
            blue = 0.0
        else:
            blue = 138.5177312231 * np.log(temperature - 10.0) - 305.0447927307
    else:
        red = 329.698727446 * np.power(temperature - 60.0, -0.1332047592)
        green = 288.1221695283 * np.power(temperature - 60.0, -0.0755148492)
        blue = 255.0

    rgb = np.array([red, green, blue], dtype=np.float32) / 255.0
    rgb = _clamp_rgb(rgb)
    return rgb / np.max(rgb)


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    else:
        image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def _tile_images(images: np.ndarray, columns: int = 3, background: int = 0) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(f"Expected batched images with shape (N, H, W, C), got {images.shape}.")

    batch, height, width, channels = images.shape
    columns = max(1, min(columns, batch))
    rows = (batch + columns - 1) // columns

    canvas = np.full(
        (rows * height, columns * width, channels),
        fill_value=background,
        dtype=images.dtype,
    )
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    return canvas


@register_env(
    HUMAN_ENV_NAME,
    max_episode_steps=100,
    override=True,
)
class UnitreeG1LiftAppleEnv(_BaseHumanoidPlaceAppleInBowl):
    """Humanoid task simplified to holding two fingertips on a near-hand cube."""

    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]
    agent: UnitreeG1UpperBodyWithHeadCamera
    kitchen_scene_scale = 0.82
    cube_half_size = 0.05
    cube_body_type = "kinematic"
    cube_start_xy = (-0.005, -0.13)
    cube_start_z = 0.74
    cube_xy_noise = 0.005
    pregrasp_z_offset = 0.08
    contact_success_seconds = 0.1
    required_contact_count = 2
    finger_contact_distance_threshold = 0.042
    finger_approach_radius = 0.12
    max_dense_reward = 10.0
    object_shape = "cube"
    object_color = (0.1, 0.45, 0.95, 1.0)
    object_scale = 1.0
    light_intensity_scale = 1.0
    light_color_temperature_k: float | None = None
    right_contact_finger_link_names = (
        "right_two_link",
        "right_four_link",
        "right_six_link",
    )
    right_success_finger_indices = (0, 2)

    def __init__(self, *args, **kwargs):
        self.init_robot_pose = copy.deepcopy(
            UnitreeG1UpperBodyWithHeadCamera.keyframes["standing"].pose
        )
        self.init_robot_pose.p = [-0.3, 0, 0.755]
        super().__init__(
            *args,
            robot_uids="unitree_g1_simplified_upper_body_with_head_camera",
            **kwargs,
        )

    def _load_scene(self, options: dict):
        super(_BaseHumanoidPlaceAppleInBowl, self)._load_scene(options)
        initial_pose = Pose.create_from_pq(p=[*self.cube_start_xy, self.cube_start_z])
        object_size = self.cube_half_size * float(self.object_scale)
        if self.object_shape == "cube":
            self.cube = actors.build_cube(
                self.scene,
                half_size=object_size,
                color=list(self.object_color),
                name="cube",
                body_type=self.cube_body_type,
                initial_pose=initial_pose,
            )
        elif self.object_shape == "sphere":
            self.cube = actors.build_sphere(
                self.scene,
                radius=object_size,
                color=list(self.object_color),
                name="sphere",
                body_type=self.cube_body_type,
                initial_pose=initial_pose,
            )
        else:
            raise ValueError(f"Unsupported object shape `{self.object_shape}`.")
        self.apple = self.cube

    def _load_lighting(self, options: dict):
        use_default_lighting = (
            self.light_color_temperature_k is None
            and abs(float(self.light_intensity_scale) - 1.0) < 1e-6
        )
        if use_default_lighting:
            return super()._load_lighting(options)

        tint = _kelvin_to_rgb(
            self.light_color_temperature_k or _BASE_LIGHT_COLOR_TEMPERATURE_K
        )
        ambient = (
            _DEFAULT_AMBIENT_LIGHT * float(self.light_intensity_scale) * tint
        ).tolist()
        directional = (
            _DEFAULT_DIRECTIONAL_LIGHT * float(self.light_intensity_scale) * tint
        ).tolist()

        self.scene.set_ambient_light(ambient)
        self.scene.add_directional_light(
            [1, 1, -1],
            directional,
            shadow=self.enable_shadow,
            shadow_scale=5,
            shadow_map_size=2048,
        )
        self.scene.add_directional_light([0, 0, -1], directional)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22,
                max_rigid_patch_count=2**21,
            ),
            scene_config=SceneConfig(contact_offset=0.01),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)

            self.agent.robot.set_qpos(self.agent.keyframes["standing"].qpos)
            self.agent.robot.set_pose(self.init_robot_pose)

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, :2] = torch.tensor(self.cube_start_xy, device=self.device)
            xyz[:, :2] += randomization.uniform(
                low=-self.cube_xy_noise,
                high=self.cube_xy_noise,
                size=(b, 2),
            )
            xyz[:, 2] = self.cube_start_z
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            if not hasattr(self, "contact_hold_steps"):
                self.contact_hold_steps = torch.zeros(
                    self.num_envs,
                    dtype=torch.int32,
                    device=self.device,
                )
            self.contact_hold_steps[env_idx] = 0
            if not hasattr(self, "last_contact_update_step"):
                self.last_contact_update_step = torch.full(
                    (self.num_envs,),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
            self.last_contact_update_step[env_idx] = -1

    @property
    def required_contact_steps(self) -> int:
        return max(1, int(np.ceil(self.contact_success_seconds * self.control_freq)))

    def _compute_right_finger_contacts(self):
        finger_positions = []
        for link_name in self.right_contact_finger_link_names:
            finger_positions.append(self.agent.robot.links_map[link_name].pose.p)
        finger_positions = torch.stack(finger_positions, dim=1)
        cube_to_finger = finger_positions - self.cube.pose.p[:, None, :]
        object_size = self.cube_half_size * float(self.object_scale)
        if self.object_shape == "sphere":
            finger_distances = torch.clamp(
                torch.linalg.norm(cube_to_finger, axis=2) - object_size,
                min=0.0,
            )
        else:
            outside_box = torch.clamp(
                torch.abs(cube_to_finger) - object_size,
                min=0.0,
            )
            finger_distances = torch.linalg.norm(outside_box, axis=2)
        finger_contacts = finger_distances <= self.finger_contact_distance_threshold
        contact_count = torch.sum(finger_contacts, dim=1)
        success_finger_contacts = finger_contacts[
            :, list(self.right_success_finger_indices)
        ]
        has_two_finger_contact = torch.all(success_finger_contacts, dim=1)
        return finger_contacts, finger_distances, contact_count, has_two_finger_contact

    def evaluate(self):
        (
            finger_contacts,
            finger_distances,
            contact_count,
            has_two_finger_contact,
        ) = self._compute_right_finger_contacts()
        if not hasattr(self, "contact_hold_steps"):
            self.contact_hold_steps = torch.zeros(
                self.num_envs,
                dtype=torch.int32,
                device=self.device,
            )
        if not hasattr(self, "last_contact_update_step"):
            self.last_contact_update_step = torch.full(
                (self.num_envs,),
                -1,
                dtype=torch.int32,
                device=self.device,
            )

        current_step = self.elapsed_steps.to(torch.int32)
        should_update = current_step != self.last_contact_update_step
        updated_hold_steps = torch.where(
            has_two_finger_contact,
            self.contact_hold_steps + 1,
            torch.zeros_like(self.contact_hold_steps),
        )
        self.contact_hold_steps = torch.where(
            should_update,
            updated_hold_steps,
            self.contact_hold_steps,
        )
        self.last_contact_update_step = torch.where(
            should_update,
            current_step,
            self.last_contact_update_step,
        )

        contact_hold_time = self.contact_hold_steps.float() / float(self.control_freq)
        success = self.contact_hold_steps >= self.required_contact_steps
        return {
            "success": success,
            "is_grasped": has_two_finger_contact,
            "has_grasped": self.contact_hold_steps > 0,
            "is_lifted": success,
            "right_finger_contacts": finger_contacts,
            "right_finger_distances": finger_distances,
            "right_finger_contact_forces": torch.zeros_like(finger_distances),
            "right_finger_contact_count": contact_count,
            "right_success_finger_contacts": finger_contacts[
                :, list(self.right_success_finger_indices)
            ],
            "has_two_finger_contact": has_two_finger_contact,
            "contact_hold_steps": self.contact_hold_steps,
            "contact_hold_time": contact_hold_time,
            "required_contact_steps": torch.full_like(
                self.contact_hold_steps,
                self.required_contact_steps,
            ),
            "cube_lift": torch.zeros(self.num_envs, device=self.device),
            "apple_lift": torch.zeros(self.num_envs, device=self.device),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            has_grasped=info["has_grasped"],
            is_lifted=info["is_lifted"],
            has_two_finger_contact=info["has_two_finger_contact"],
            tcp_pose=self.agent.right_tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            pregrasp_pos = self.cube.pose.p.clone()
            pregrasp_pos[:, 2] += self.pregrasp_z_offset
            contact_goal_pos = self.cube.pose.p.clone()
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.right_tcp.pose.p,
                tcp_to_pregrasp_pos=pregrasp_pos - self.agent.right_tcp.pose.p,
                obj_to_lift_goal_pos=contact_goal_pos - self.cube.pose.p,
                right_finger_contacts=info["right_finger_contacts"],
                right_finger_distances=info["right_finger_distances"],
                right_finger_contact_count=info["right_finger_contact_count"],
                right_success_finger_contacts=info["right_success_finger_contacts"],
                contact_hold_time=info["contact_hold_time"],
                cube_lift=info["cube_lift"],
                apple_lift=info["apple_lift"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_pos = self.agent.right_tcp.pose.p
        obj_pos = self.cube.pose.p
        tcp_to_obj = obj_pos - tcp_pos
        tcp_to_obj_dist = torch.linalg.norm(tcp_to_obj, axis=1)

        pregrasp_pos = obj_pos.clone()
        pregrasp_pos[:, 2] += self.pregrasp_z_offset
        tcp_to_pregrasp_dist = torch.linalg.norm(pregrasp_pos - tcp_pos, axis=1)
        reaching_reward = 1.0 - torch.tanh(8.0 * tcp_to_pregrasp_dist)

        tcp_to_obj_xy_dist = torch.linalg.norm(tcp_to_obj[:, :2], axis=1)
        centering_reward = 0.5 * (1.0 - torch.tanh(10.0 * tcp_to_obj_xy_dist))

        tcp_obj_z_gap = tcp_pos[:, 2] - obj_pos[:, 2]
        z_align_reward = 0.5 * (
            1.0 - torch.tanh(10.0 * torch.abs(tcp_obj_z_gap - self.pregrasp_z_offset))
        )

        target_contact_progress = torch.clamp(
            torch.sum(info["right_success_finger_contacts"].float(), dim=1)
            / float(self.required_contact_count),
            min=0.0,
            max=1.0,
        )
        hold_progress = torch.clamp(
            info["contact_hold_steps"].float() / float(self.required_contact_steps),
            min=0.0,
            max=1.0,
        )
        target_finger_distances = info["right_finger_distances"][
            :, list(self.right_success_finger_indices)
        ]
        target_finger_progress = torch.clamp(
            (self.finger_approach_radius - target_finger_distances)
            / (self.finger_approach_radius - self.finger_contact_distance_threshold),
            min=0.0,
            max=1.0,
        )
        finger_reach_reward = torch.min(target_finger_progress, dim=1).values

        reward = 0.05 * reaching_reward + 0.05 * centering_reward + 0.05 * z_align_reward
        reward += finger_reach_reward
        reward += 0.5 * target_contact_progress
        reward += 4.0 * info["has_two_finger_contact"].float()
        reward += 4.0 * hold_progress
        reward = torch.clamp(reward, max=self.max_dense_reward)
        reward[info["success"]] = self.max_dense_reward
        return reward

    def compute_normalized_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.max_dense_reward


_TASK_SPECS = [
    dict(
        task_name="UnitreeG1LiftCube",
        class_prefix="UnitreeG1LiftCube",
        object_shape="cube",
        description="cube object",
    ),
    dict(
        task_name="UnitreeG1LiftSphere",
        class_prefix="UnitreeG1LiftSphere",
        object_shape="sphere",
        description="sphere object",
    ),
]

_VARIANT_SPECS = [
    dict(
        class_suffix="LightStronger50",
        env_suffix="LightStronger50",
        light_intensity_scale=1.5,
        description="lighting 50% stronger than the baseline scene",
    ),
    dict(
        class_suffix="LightWeaker50",
        env_suffix="LightWeaker50",
        light_intensity_scale=0.5,
        description="lighting 50% weaker than the baseline scene",
    ),
    dict(
        class_suffix="ObjectBlack",
        env_suffix="ObjectBlack",
        object_color=(0.0, 0.0, 0.0, 1.0),
        description="target object recolored to black",
    ),
    dict(
        class_suffix="ObjectPurple",
        env_suffix="ObjectPurple",
        object_color=(0.5, 0.0, 0.5, 1.0),
        description="target object recolored to purple",
    ),
    dict(
        class_suffix="ColorTempLower50",
        env_suffix="ColorTempLower50",
        light_color_temperature_k=_BASE_LIGHT_COLOR_TEMPERATURE_K * 0.5,
        description="lighting color temperature 50% lower than the baseline scene",
    ),
    dict(
        class_suffix="ColorTempHigher50",
        env_suffix="ColorTempHigher50",
        light_color_temperature_k=_BASE_LIGHT_COLOR_TEMPERATURE_K * 1.5,
        description="lighting color temperature 50% higher than the baseline scene",
    ),
    dict(
        class_suffix="ObjectScaleDown1p1",
        env_suffix="ObjectScaleDown1p1",
        object_scale=1 / 1.1,
        description="target object size scaled down to 1/1.1x of the baseline size",
    ),
    dict(
        class_suffix="ObjectScaleDown1p3",
        env_suffix="ObjectScaleDown1p3",
        object_scale=1 / 1.3,
        description="target object size scaled down to 1/1.3x of the baseline size",
    ),
    dict(
        class_suffix="ObjectScaleDown1p5",
        env_suffix="ObjectScaleDown1p5",
        object_scale=1 / 1.5,
        description="target object size scaled down to 1/1.5x of the baseline size",
    ),
]


def _register_human_variants() -> None:
    for task_spec in _TASK_SPECS:
        for variant_spec in _VARIANT_SPECS:
            class_name = f"{task_spec['class_prefix']}{variant_spec['class_suffix']}Env"
            env_id = f"{task_spec['task_name']}{variant_spec['env_suffix']}-v1"
            attrs = {
                "__doc__": (
                    f"{HUMAN_ENV_NAME} variant with {task_spec['description']} and "
                    f"{variant_spec['description']}."
                ),
                "__module__": __name__,
                "ENV_ID": env_id,
                "object_shape": task_spec["object_shape"],
                "object_color": variant_spec.get(
                    "object_color",
                    UnitreeG1LiftAppleEnv.object_color,
                ),
                "object_scale": variant_spec.get("object_scale", 1.0),
                "light_intensity_scale": variant_spec.get("light_intensity_scale", 1.0),
                "light_color_temperature_k": variant_spec.get(
                    "light_color_temperature_k"
                ),
            }
            variant_cls = type(class_name, (UnitreeG1LiftAppleEnv,), attrs)
            variant_cls = register_env(
                env_id,
                max_episode_steps=100,
                override=True,
            )(variant_cls)
            globals()[class_name] = variant_cls
            HUMAN_VARIANT_ENV_IDS.append(env_id)
            HUMAN_VARIANT_CLASSES[env_id] = variant_cls
            HUMAN_ENV_IDS.append(env_id)


def render_env_preview(env, seed: int = 0) -> np.ndarray:
    env.reset(seed=seed)
    image = env.unwrapped.render_rgb_array()
    if image is None:
        raise RuntimeError(f"Failed to render preview for `{type(env.unwrapped).__name__}`.")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    else:
        image = np.asarray(image)
    if image.ndim == 4 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim == 4:
        image = _tile_images(_to_uint8_image(image), columns=3)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return _to_uint8_image(image)


def save_env_preview(env, output_path: str | Path, seed: int = 0) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(render_env_preview(env, seed=seed)).convert("RGB")
    image.save(output_path, format="JPEG")
    return output_path


def generate_all_previews(
    output_dir: str | Path | None = None,
    seed: int = 0,
    num_envs: int = 6,
    env_ids: list[str] | None = None,
) -> list[Path]:
    output_dir = Path(output_dir or Path(__file__).with_name("previews"))
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for env_id in (env_ids or HUMAN_VARIANT_ENV_IDS):
        env = gym.make(
            env_id,
            obs_mode="state",
            render_mode="rgb_array",
            num_envs=num_envs,
        )
        try:
            saved_paths.append(save_env_preview(env, output_dir / f"{env_id}.jpg", seed=seed))
        finally:
            env.close()
    return saved_paths


HumanoidPlaceAppleInBowl = UnitreeG1LiftAppleEnv
_register_human_variants()


__all__ = [
    "HUMAN_ENV_NAME",
    "HUMAN_ENV_IDS",
    "HUMAN_VARIANT_ENV_IDS",
    "HUMAN_VARIANT_CLASSES",
    "HumanoidPlaceAppleInBowl",
    "UnitreeG1LiftAppleEnv",
    "render_env_preview",
    "save_env_preview",
    "generate_all_previews",
] + [cls.__name__ for cls in HUMAN_VARIANT_CLASSES.values()]


if __name__ == "__main__":
    preview_paths = generate_all_previews(num_envs=6)
    for preview_path in preview_paths:
        print(f"saved preview: {preview_path}")

    """
    UnitreeG1LiftCubeColorTempLower50-v1
    UnitreeG1LiftCubeLightWeaker50-v1
    UnitreeG1LiftCubeObjectPurple-v1
    UnitreeG1LiftCubeObjectScaleDown1p1-v1
    UnitreeG1LiftCubeObjectScaleDown1p3-v1
    UnitreeG1LiftSphereLightStronger50-v1
    UnitreeG1LiftSphereObjectScaleDown1p3-v1

    "['UnitreeG1LiftCubeObjectScaleDown1p3-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeObjectPurple-v1','UnitreeG1LiftSphereLightStronger50-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectScaleDown1p1-v1','UnitreeG1LiftSphereObjectScaleDown1p3-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectPurple-v1']"
    """