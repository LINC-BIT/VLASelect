from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import sapien
import torch
from PIL import Image

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.place_sphere import PlaceSphereEnv
from mani_skill.envs.tasks.tabletop.poke_cube import PokeCubeEnv
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.envs.tasks.tabletop.roll_ball import RollBallEnv
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose

try:
    _RESAMPLING = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLING = Image.LANCZOS

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


def _scale_value(value, factor: float):
    if torch.is_tensor(value):
        return value * factor
    if isinstance(value, np.ndarray):
        return value * factor
    if isinstance(value, tuple):
        return tuple(_scale_value(item, factor) for item in value)
    if isinstance(value, list):
        return [_scale_value(item, factor) for item in value]
    return value * factor


_OBJECT_SCALE_ATTRS_BY_BASE_ENV_ID = {
    "PickCube-v1": ("cube_half_size",),
    "PushCube-v1": ("cube_half_size",),
    "PokeCube-v1": ("cube_half_size",),
    "RollBall-v1": ("ball_radius",),
    "PlaceSphere-v1": (
        "radius",
        "inner_side_half_len",
        "short_side_half_size",
        "block_half_size",
        "edge_block_half_size",
    ),
}


class _TableTopVariantMixin:
    TARGET_OBJECT_ATTR: Optional[str] = None
    OBJECT_COLOR: Optional[tuple[float, float, float, float]] = None
    OBJECT_SCALE: float = 1.0
    LIGHT_INTENSITY_SCALE: float = 1.0
    LIGHT_COLOR_TEMPERATURE_K: Optional[float] = None

    def _load_scene(self, options: dict):
        self._apply_object_scale()
        if self.BASE_ENV_ID == "StackCube-v1":
            self._load_stack_cube_scene()
        else:
            super()._load_scene(options)
        if self.OBJECT_COLOR is not None:
            self._apply_target_object_color(np.asarray(self.OBJECT_COLOR, dtype=np.float32))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.BASE_ENV_ID == "StackCube-v1":
            self._initialize_stack_cube_episode(env_idx)
            return
        super()._initialize_episode(env_idx, options)

    def _apply_object_scale(self) -> None:
        scale = float(getattr(self, "OBJECT_SCALE", 1.0))
        base_env_id = getattr(self, "BASE_ENV_ID", None)
        attr_names = _OBJECT_SCALE_ATTRS_BY_BASE_ENV_ID.get(base_env_id, ())

        if not hasattr(self, "_base_scaled_attrs"):
            self._base_scaled_attrs = {
                name: copy.deepcopy(getattr(self, name)) for name in attr_names
            }
            self._base_stack_cube_half_size = 0.02

        for name, base_value in self._base_scaled_attrs.items():
            setattr(self, name, _scale_value(base_value, scale))

        self._stack_cube_half_size = self._base_stack_cube_half_size * scale

    def _load_stack_cube_scene(self) -> None:
        cube_half_size = self._stack_cube_half_size
        self.cube_half_size = common.to_tensor([cube_half_size] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=cube_half_size,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, cube_half_size]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=cube_half_size,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, cube_half_size]),
        )

    def _initialize_stack_cube_episode(self, env_idx: torch.Tensor) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = self._stack_cube_half_size
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(self.cube_half_size[:2]) + 0.001
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def _load_lighting(self, options: dict):
        use_default_lighting = (
            self.LIGHT_COLOR_TEMPERATURE_K is None
            and abs(self.LIGHT_INTENSITY_SCALE - 1.0) < 1e-6
        )
        if use_default_lighting:
            return super()._load_lighting(options)

        tint = _kelvin_to_rgb(
            self.LIGHT_COLOR_TEMPERATURE_K or _BASE_LIGHT_COLOR_TEMPERATURE_K
        )
        ambient = (_DEFAULT_AMBIENT_LIGHT * self.LIGHT_INTENSITY_SCALE * tint).tolist()
        directional = (
            _DEFAULT_DIRECTIONAL_LIGHT * self.LIGHT_INTENSITY_SCALE * tint
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

    def _get_target_actor(self):
        if self.TARGET_OBJECT_ATTR is None:
            raise AttributeError(
                f"{type(self).__name__} does not declare TARGET_OBJECT_ATTR."
            )
        actor = getattr(self, self.TARGET_OBJECT_ATTR, None)
        if actor is None:
            raise AttributeError(
                f"{type(self).__name__} has no target actor `{self.TARGET_OBJECT_ATTR}`."
            )
        return actor

    def _apply_target_object_color(self, rgba: np.ndarray) -> None:
        actor = self._get_target_actor()
        for entity in getattr(actor, "_objs", []):
            for component in getattr(entity, "components", []):
                render_shapes = getattr(component, "render_shapes", None)
                if render_shapes is None:
                    continue
                for render_shape in render_shapes:
                    material = getattr(render_shape, "material", None)
                    if material is None and hasattr(render_shape, "get_material"):
                        material = render_shape.get_material()
                    if material is not None:
                        material.set_base_color(rgba.tolist())


_TASK_SPECS = [
    dict(
        task_name="PickCube",
        base_env_id="PickCube-v1",
        base_cls=PickCubeEnv,
        max_episode_steps=50,
        target_object_attr="cube",
    ),
    dict(
        task_name="PushCube",
        base_env_id="PushCube-v1",
        base_cls=PushCubeEnv,
        max_episode_steps=50,
        target_object_attr="obj",
    ),
    dict(
        task_name="PokeCube",
        base_env_id="PokeCube-v1",
        base_cls=PokeCubeEnv,
        max_episode_steps=50,
        target_object_attr="cube",
    ),
    dict(
        task_name="RollBall",
        base_env_id="RollBall-v1",
        base_cls=RollBallEnv,
        max_episode_steps=80,
        target_object_attr="ball",
    ),
    dict(
        task_name="StackCube",
        base_env_id="StackCube-v1",
        base_cls=StackCubeEnv,
        max_episode_steps=50,
        target_object_attr="cubeA",
    ),
    dict(
        task_name="PlaceSphere",
        base_env_id="PlaceSphere-v1",
        base_cls=PlaceSphereEnv,
        max_episode_steps=50,
        target_object_attr="obj",
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
        class_suffix="ObjectScaleUp1p2",
        env_suffix="ObjectScaleUp1p2",
        object_scale=1.2,
        description="task-relevant object size scaled up by 1.2x",
    ),
    dict(
        class_suffix="ObjectScaleDown1p2",
        env_suffix="ObjectScaleDown1p2",
        object_scale=1 / 1.2,
        description="task-relevant object size scaled down to 1/1.2x of the baseline scene",
    ),
    dict(
        class_suffix="ObjectScaleUp1p4",
        env_suffix="ObjectScaleUp1p4",
        object_scale=1.4,
        description="task-relevant object size scaled up by 1.4x",
    ),
    dict(
        class_suffix="ObjectScaleDown1p4",
        env_suffix="ObjectScaleDown1p4",
        object_scale=1 / 1.4,
        description="task-relevant object size scaled down to 1/1.4x of the baseline scene",
    ),
]

TABLE_TOP_VARIANT_ENV_IDS: list[str] = []
TABLE_TOP_VARIANT_CLASSES: dict[str, type] = {}
TABLE_TOP_OBJECT_SCALE_ENV_IDS: list[str] = []
EXP_WORKLOAD_ENVS: list[str] = [
    "PickCubeObjectScaleUp1p2-v1",
    "PickCubeLightStronger50-v1",
    "PickCubeObjectScaleUp1p4-v1",
    "PickCubeLightWeaker50-v1",


    "PushCubeLightWeaker50-v1",
    "PushCubeLightStronger50-v1",
    "PushCubeColorTempHigher50-v1",
    "PushCubeColorTempLower50-v1",
    
    "PickCubeColorTempHigher50-v1",
    "PickCubeObjectScaleDown1p2-v1",
]


def _register_table_top_variants() -> None:
    for task_spec in _TASK_SPECS:
        for variant_spec in _VARIANT_SPECS:
            class_name = f"{task_spec['task_name']}{variant_spec['class_suffix']}Env"
            env_id = f"{task_spec['task_name']}{variant_spec['env_suffix']}-v1"
            attrs = {
                "__doc__": (
                    f"{task_spec['base_env_id']} variant with "
                    f"{variant_spec['description']}."
                ),
                "__module__": __name__,
                "BASE_ENV_ID": task_spec["base_env_id"],
                "ENV_ID": env_id,
                "TARGET_OBJECT_ATTR": task_spec["target_object_attr"],
                "OBJECT_COLOR": variant_spec.get("object_color"),
                "OBJECT_SCALE": variant_spec.get("object_scale", 1.0),
                "LIGHT_INTENSITY_SCALE": variant_spec.get("light_intensity_scale", 1.0),
                "LIGHT_COLOR_TEMPERATURE_K": variant_spec.get(
                    "light_color_temperature_k"
                ),
            }
            variant_cls = type(
                class_name,
                (_TableTopVariantMixin, task_spec["base_cls"]),
                attrs,
            )
            variant_cls = register_env(
                env_id,
                max_episode_steps=task_spec["max_episode_steps"],
                override=True,
            )(variant_cls)
            globals()[class_name] = variant_cls
            TABLE_TOP_VARIANT_ENV_IDS.append(env_id)
            TABLE_TOP_VARIANT_CLASSES[env_id] = variant_cls
            if "object_scale" in variant_spec:
                TABLE_TOP_OBJECT_SCALE_ENV_IDS.append(env_id)


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
    env_ids: Optional[list[str]] = None,
) -> list[Path]:
    output_dir = Path(output_dir or Path(__file__).with_name("previews"))
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for env_id in (env_ids or TABLE_TOP_VARIANT_ENV_IDS):
        env = gym.make(env_id, obs_mode="state", render_mode="rgb_array", num_envs=num_envs)
        try:
            saved_paths.append(save_env_preview(env, output_dir / f"{env_id}.jpg", seed=seed))
        finally:
            env.close()
    return saved_paths


_register_table_top_variants()

__all__ = [
    "TABLE_TOP_VARIANT_ENV_IDS",
    "TABLE_TOP_VARIANT_CLASSES",
    "TABLE_TOP_OBJECT_SCALE_ENV_IDS",
    "EXP_WORKLOAD_ENVS",
    "render_env_preview",
    "save_env_preview",
    "generate_all_previews",
] + [cls.__name__ for cls in TABLE_TOP_VARIANT_CLASSES.values()]


if __name__ == "__main__":
    print(f"registered variants: {len(TABLE_TOP_VARIANT_ENV_IDS)}")
    print(f"object-scale variants: {len(TABLE_TOP_OBJECT_SCALE_ENV_IDS)}")
    print(f"experiment workload variants: {len(EXP_WORKLOAD_ENVS)}")
    assert all(env_id in TABLE_TOP_VARIANT_ENV_IDS for env_id in EXP_WORKLOAD_ENVS), "All experiment workload envs must be in the registered variant env ids."
