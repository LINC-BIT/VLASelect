from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import gymnasium as gym
import numpy as np
import sapien
import torch
from PIL import Image

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from mani_skill import ASSET_DIR
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from train.vla_adapter_new.model_impl.env import HoldCubeInHand

_BASE_LIGHT_COLOR_TEMPERATURE_K = 6500.0
_DEFAULT_AMBIENT_LIGHT = np.array([0.3, 0.3, 0.3], dtype=np.float32)
_DEFAULT_DIRECTIONAL_LIGHT = np.array([1.0, 1.0, 1.0], dtype=np.float32)
_YCB_MODEL_DB = load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")


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


def _build_scaled_ycb_builder(scene, model_id: str, object_scale: float):
    metadata = _YCB_MODEL_DB[model_id]
    density = metadata.get("density", 1000)
    dataset_scale = metadata.get("scales", [1.0])[0]
    scale = dataset_scale * float(object_scale)
    model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / model_id

    builder = scene.create_actor_builder()
    builder.add_multiple_convex_collisions_from_file(
        filename=str(model_dir / "collision.ply"),
        scale=[scale] * 3,
        material=None,
        density=density,
    )
    builder.add_visual_from_file(
        filename=str(model_dir / "textured.obj"),
        scale=[scale] * 3,
    )
    return builder


class _HoldInHandVariantMixin:
    TASK_NAME: str = "HoldCubeInHand"
    OBJECT_KIND: str = "cube"
    YCB_MODEL_ID: Optional[str] = None
    OBJECT_SCALE: float = 1.0
    LIGHT_INTENSITY_SCALE: float = 1.0
    LIGHT_COLOR_TEMPERATURE_K: Optional[float] = None

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        if self.OBJECT_KIND == "cube":
            half_size = 0.04 * float(self.OBJECT_SCALE)
            self.obj = actors.build_cube(
                self.scene,
                half_size=half_size,
                color=np.array([255, 255, 255, 255], dtype=np.float32) / 255.0,
                name="cube",
                body_type="dynamic",
            )
            self.obj_heights = common.to_tensor(
                [0.03 * float(self.OBJECT_SCALE)],
                device=self.device,
            )
            return

        if self.OBJECT_KIND != "ycb" or self.YCB_MODEL_ID is None:
            raise ValueError(
                f"{type(self).__name__} requires either cube object or a valid YCB model id."
            )

        self._objs = []
        for scene_idx in range(self.num_envs):
            builder = _build_scaled_ycb_builder(
                self.scene,
                model_id=self.YCB_MODEL_ID,
                object_scale=self.OBJECT_SCALE,
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([scene_idx])
            obj = builder.build(name=f"{self.YCB_MODEL_ID}-{scene_idx}")
            self._objs.append(obj)
            self.remove_from_state_dict_registry(obj)
        self.obj = Actor.merge(self._objs, name="ycb_object")
        self.add_to_state_dict_registry(self.obj)

    def _after_reconfigure(self, options: dict):
        if self.OBJECT_KIND != "ycb":
            return

        obj_heights = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            obj_heights.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.obj_heights = common.to_tensor(obj_heights, device=self.device)

    def _load_lighting(self, options: dict):
        use_default_lighting = (
            self.LIGHT_COLOR_TEMPERATURE_K is None
            and abs(float(self.LIGHT_INTENSITY_SCALE) - 1.0) < 1e-6
        )
        if use_default_lighting:
            return super()._load_lighting(options)

        tint = _kelvin_to_rgb(
            self.LIGHT_COLOR_TEMPERATURE_K or _BASE_LIGHT_COLOR_TEMPERATURE_K
        )
        ambient = (_DEFAULT_AMBIENT_LIGHT * float(self.LIGHT_INTENSITY_SCALE) * tint).tolist()
        directional = (
            _DEFAULT_DIRECTIONAL_LIGHT * float(self.LIGHT_INTENSITY_SCALE) * tint
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


_TASK_SPECS = [
    dict(
        task_name="HoldCubeInHand",
        object_kind="cube",
        ycb_model_id=None,
        asset_download_ids=[],
    ),
    dict(
        task_name="HoldHammerInHand",
        object_kind="ycb",
        ycb_model_id="048_hammer",
        asset_download_ids=["ycb"],
    ),
    dict(
        task_name="HoldWrenchInHand",
        object_kind="ycb",
        ycb_model_id="042_adjustable_wrench",
        asset_download_ids=["ycb"],
    ),
    dict(
        task_name="HoldWoodBlockInHand",
        object_kind="ycb",
        ycb_model_id="036_wood_block",
        asset_download_ids=["ycb"],
    ),
]

_VARIANT_SPECS = [
    # dict(
    #     class_suffix="LightStronger50",
    #     env_suffix="LightStronger50",
    #     light_intensity_scale=1.5,
    #     description="lighting 50% stronger than the baseline scene",
    # ),
    # dict(
    #     class_suffix="LightWeaker50",
    #     env_suffix="LightWeaker50",
    #     light_intensity_scale=0.5,
    #     description="lighting 50% weaker than the baseline scene",
    # ),
    # dict(
    #     class_suffix="ColorTempLower50",
    #     env_suffix="ColorTempLower50",
    #     light_color_temperature_k=_BASE_LIGHT_COLOR_TEMPERATURE_K * 0.5,
    #     description="lighting color temperature 50% lower than the baseline scene",
    # ),
    # dict(
    #     class_suffix="ColorTempHigher50",
    #     env_suffix="ColorTempHigher50",
    #     light_color_temperature_k=_BASE_LIGHT_COLOR_TEMPERATURE_K * 1.5,
    #     description="lighting color temperature 50% higher than the baseline scene",
    # ),
    dict(
        class_suffix="ObjectScaleDown1p2",
        env_suffix="ObjectScaleDown1p2",
        object_scale=1 / 1.2,
        description="held object scaled down to 1/1.2x of the baseline size",
    ),
    dict(
        class_suffix="ObjectScaleDown1p4",
        env_suffix="ObjectScaleDown1p4",
        object_scale=1 / 1.4,
        description="held object scaled down to 1/1.4x of the baseline size",
    ),
    dict(
        class_suffix="ObjectScaleDown1p6",
        env_suffix="ObjectScaleDown1p6",
        object_scale=1 / 1.6,
        description="held object scaled down to 1/1.6x of the baseline size",
    ),
    dict(
        class_suffix="ObjectScaleUp1p2",
        env_suffix="ObjectScaleUp1p2",
        object_scale=1.2,
        description="held object scaled up by 1.2x",
    ),
    dict(
        class_suffix="ObjectScaleUp1p4",
        env_suffix="ObjectScaleUp1p4",
        object_scale=1.4,
        description="held object scaled up by 1.4x",
    ),
    dict(
        class_suffix="ObjectScaleUp1p6",
        env_suffix="ObjectScaleUp1p6",
        object_scale=1.6,
        description="held object scaled up by 1.6x",
    ),
]

HOLD_IN_HAND_VARIANT_ENV_IDS: list[str] = []
HOLD_IN_HAND_VARIANT_CLASSES: dict[str, type] = {}


def _register_hold_in_hand_variants() -> None:
    for task_spec in _TASK_SPECS:
        for variant_spec in _VARIANT_SPECS:
            class_name = f"{task_spec['task_name']}{variant_spec['class_suffix']}Env"
            env_id = f"{task_spec['task_name']}{variant_spec['env_suffix']}-v1"
            variant_cls = type(
                class_name,
                (_HoldInHandVariantMixin, HoldCubeInHand),
                {
                    "__doc__": (
                        f"{task_spec['task_name']} variant with "
                        f"{variant_spec['description']}."
                    ),
                    "__module__": __name__,
                    "ENV_ID": env_id,
                    "TASK_NAME": task_spec["task_name"],
                    "OBJECT_KIND": task_spec["object_kind"],
                    "YCB_MODEL_ID": task_spec["ycb_model_id"],
                    "OBJECT_SCALE": variant_spec.get("object_scale", 1.0),
                    "LIGHT_INTENSITY_SCALE": variant_spec.get("light_intensity_scale", 1.0),
                    "LIGHT_COLOR_TEMPERATURE_K": variant_spec.get(
                        "light_color_temperature_k"
                    ),
                },
            )
            variant_cls = register_env(
                env_id,
                max_episode_steps=100,
                override=True,
                asset_download_ids=task_spec["asset_download_ids"],
            )(variant_cls)
            globals()[class_name] = variant_cls
            HOLD_IN_HAND_VARIANT_ENV_IDS.append(env_id)
            HOLD_IN_HAND_VARIANT_CLASSES[env_id] = variant_cls


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
    for env_id in (env_ids or HOLD_IN_HAND_VARIANT_ENV_IDS):
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


_register_hold_in_hand_variants()

__all__ = [
    "HOLD_IN_HAND_VARIANT_ENV_IDS",
    "HOLD_IN_HAND_VARIANT_CLASSES",
    "render_env_preview",
    "save_env_preview",
    "generate_all_previews",
] + [cls.__name__ for cls in HOLD_IN_HAND_VARIANT_CLASSES.values()]


if __name__ == "__main__":
    # print(f"registered hold-in-hand variants: {len(HOLD_IN_HAND_VARIANT_ENV_IDS)}")
    # assert len(HOLD_IN_HAND_VARIANT_ENV_IDS) == 40, "Expected exactly 40 hold-in-hand variants."
    # preview_paths = generate_all_previews()
    # for preview_path in preview_paths:
    #     print(f"saved preview: {preview_path}")
    print(HOLD_IN_HAND_VARIANT_ENV_IDS)

    selected_envs = [
        'HoldHammerInHandObjectScaleDown1p2-v1',
        'HoldHammerInHandObjectScaleDown1p4-v1',
        'HoldHammerInHandObjectScaleDown1p6-v1',
        'HoldHammerInHandObjectScaleUp1p4-v1',
        'HoldHammerInHandObjectScaleUp1p6-v1',
        'HoldWoodBlockInHandObjectScaleDown1p6-v1',
        'HoldWrenchInHandObjectScaleDown1p6-v1',
        'HoldWrenchInHandObjectScaleUp1p2-v1',
        'HoldWrenchInHandObjectScaleUp1p4-v1',
        'HoldWrenchInHandObjectScaleUp1p6-v1',
    ]