from __future__ import annotations

import sys
from pathlib import Path
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

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.envs.tasks.mobile_manipulation import OpenCabinetDrawerEnv
from mani_skill.envs.tasks.mobile_manipulation.open_cabinet_drawer import (
    CABINET_COLLISION_BIT,
)
from mani_skill.envs.utils import randomization
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.articulations import partnet_mobility
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link

_OFFICIAL_TRAIN_JSON = (
    PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
)
_CABINET_MODEL_IDS = (
    "1000",
    "1004",
    "1005",
    "1013",
    "1016",
    "1021",
    "1024",
    "1027",
    "1032",
    "1033",
)
_VARIANT_SPECS = [
    dict(class_suffix="Default", env_suffix="Default", cabinet_scale=1.0),
    dict(class_suffix="ScaleDown1p3", env_suffix="ScaleDown1p3", cabinet_scale=1 / 1.3),
    dict(class_suffix="ScaleDown1p6", env_suffix="ScaleDown1p6", cabinet_scale=1 / 1.6),
    dict(class_suffix="ScaleUp1p3", env_suffix="ScaleUp1p3", cabinet_scale=1.3),
    dict(class_suffix="ScaleUp1p6", env_suffix="ScaleUp1p6", cabinet_scale=1.6),
]


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


def _get_scaled_partnet_mobility_builder(scene, model_id: str, cabinet_scale: float):
    if partnet_mobility.PARTNET_MOBILITY is None:
        partnet_mobility._load_partnet_mobility_dataset()

    metadata = partnet_mobility.PARTNET_MOBILITY["model_data"][model_id]
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.scale = float(metadata["scale"]) * float(cabinet_scale)
    loader.load_multiple_collisions_from_file = True

    urdf_path = partnet_mobility.PARTNET_MOBILITY["model_urdf_paths"][model_id]
    applied_urdf_config = sapien_utils.parse_urdf_config(
        dict(material=dict(static_friction=1, dynamic_friction=1, restitution=0))
    )
    sapien_utils.apply_urdf_config(loader, applied_urdf_config)
    articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
    return articulation_builders[0]


class _MobileArmVariantMixin:
    TRAIN_JSON = _OFFICIAL_TRAIN_JSON
    CABINET_MODEL_ID: Optional[str] = None
    CABINET_SCALE: float = 1.0

    def _select_drawer_indices(
        self, handle_links: list[list[Link]], handle_links_meshes: list[list[object]]
    ) -> list[int]:
        link_ids = self._batched_episode_rng.randint(0, 2**31)
        return [int(link_ids[i] % len(links)) for i, links in enumerate(handle_links)]

    def _load_cabinets(self, joint_types: list[str]):
        if self.CABINET_MODEL_ID is None:
            raise ValueError(f"{type(self).__name__} requires CABINET_MODEL_ID.")

        model_ids = np.array([self.CABINET_MODEL_ID] * self.num_envs)

        self._cabinets: list[Articulation] = []
        handle_links: list[list[Link]] = []
        handle_links_meshes: list[list[object]] = []
        for i, model_id in enumerate(model_ids):
            cabinet_builder = _get_scaled_partnet_mobility_builder(
                self.scene,
                model_id=model_id,
                cabinet_scale=self.CABINET_SCALE,
            )
            cabinet_builder.set_scene_idxs(scene_idxs=[i])
            cabinet_builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
            cabinet = cabinet_builder.build(name=f"{model_id}-{i}")
            self.remove_from_state_dict_registry(cabinet)
            for link in cabinet.links:
                link.set_collision_group_bit(group=2, bit_idx=CABINET_COLLISION_BIT, bit=1)
            self._cabinets.append(cabinet)
            handle_links.append([])
            handle_links_meshes.append([])

            for link, joint in zip(cabinet.links, cabinet.joints):
                if joint.type[0] in joint_types:
                    handle_links[-1].append(link)
                    handle_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, render_shape: "handle" in render_shape.name,
                            mesh_name="handle",
                        )[0]
                    )

        self.cabinet = Articulation.merge(self._cabinets, name="cabinet")
        self.add_to_state_dict_registry(self.cabinet)
        selected_drawer_indices = self._select_drawer_indices(
            handle_links, handle_links_meshes
        )
        self.handle_link = Link.merge(
            [links[selected_drawer_indices[i]] for i, links in enumerate(handle_links)],
            name="handle_link",
        )
        self.handle_link_pos = common.to_tensor(
            np.array(
                [
                    meshes[selected_drawer_indices[i]].bounding_box.center_mass
                    for i, meshes in enumerate(handle_links_meshes)
                ]
            ),
            device=self.device,
        )

        self.handle_link_goal = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0, 1, 0, 1],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )


class _OpenCabinetDrawerEasyBase(_MobileArmVariantMixin, OpenCabinetDrawerEnv):
    CABINET_MODEL_ID = "1000"
    CABINET_SCALE = 1.0
    fetch_init_qpos = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.09994358569383621,
        0.0,
        -0.720679759979248,
        -0.06573399901390076,
        0.928787887096405,
        -0.2604657709598541,
        1.0377516746520996,
        0.8531534075737,
        0.015,
        0.015,
    )
    preferred_handle_height = 0.75
    preferred_handle_lateral = 0.0
    min_open_frac = 0.25
    spawn_relative_to_handle = True
    spawn_handle_offset_x = -0.84
    spawn_handle_offset_y = 0.03
    spawn_handle_offset_x_noise = 0.02
    spawn_handle_offset_y_noise = 0.02
    spawn_yaw_center = 0.0
    spawn_dist_min = 1.05
    spawn_dist_max = 1.2
    spawn_theta_min = 0.97 * np.pi
    spawn_theta_max = 1.03 * np.pi
    spawn_yaw_noise = 0.01 * np.pi
    base_approach_reward_weight = 0.0
    max_dense_reward = 5.0
    open_reward_deadzone_frac = 0.15
    reach_reward_open_gate_frac = 0.25

    def _select_drawer_indices(
        self, handle_links: list[list[Link]], handle_links_meshes: list[list[object]]
    ) -> list[int]:
        selected_indices: list[int] = []
        for meshes in handle_links_meshes:
            centers = np.array([mesh.bounding_box.center_mass for mesh in meshes])
            scores = np.abs(centers[:, 2] - self.preferred_handle_height)
            scores += 0.25 * np.abs(centers[:, 1] - self.preferred_handle_lateral)
            selected_indices.append(int(np.argmin(scores)))
        return selected_indices

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xy = torch.zeros((b, 3))
            xy[:, 2] = self.cabinet_zs[env_idx]
            self.cabinet.set_pose(Pose.create_from_pq(p=xy))

            if self.robot_uids == "fetch":
                qpos = torch.tensor(self.fetch_init_qpos)
                qpos = qpos.repeat(b).reshape(b, -1)
                if self.spawn_relative_to_handle:
                    handle_pos = self.handle_link_positions(env_idx)
                    offset_x = randomization.uniform(
                        self.spawn_handle_offset_x - self.spawn_handle_offset_x_noise,
                        self.spawn_handle_offset_x + self.spawn_handle_offset_x_noise,
                        size=(b,),
                    )
                    offset_y = randomization.uniform(
                        self.spawn_handle_offset_y - self.spawn_handle_offset_y_noise,
                        self.spawn_handle_offset_y + self.spawn_handle_offset_y_noise,
                        size=(b,),
                    )
                    qpos[:, 0] = handle_pos[:, 0] + offset_x
                    qpos[:, 1] = handle_pos[:, 1] + offset_y
                    noise_ori = randomization.uniform(
                        -self.spawn_yaw_noise, self.spawn_yaw_noise, size=(b,)
                    )
                    qpos[:, 2] = self.spawn_yaw_center + noise_ori
                else:
                    dist = randomization.uniform(
                        self.spawn_dist_min, self.spawn_dist_max, size=(b,)
                    )
                    theta = randomization.uniform(
                        self.spawn_theta_min, self.spawn_theta_max, size=(b,)
                    )
                    xy = torch.zeros((b, 2))
                    xy[:, 0] += torch.cos(theta) * dist
                    xy[:, 1] += torch.sin(theta) * dist
                    qpos[:, :2] = xy
                    noise_ori = randomization.uniform(
                        -self.spawn_yaw_noise, self.spawn_yaw_noise, size=(b,)
                    )
                    ori = (theta - torch.pi) + noise_ori
                    qpos[:, 2] = ori
                self.agent.robot.set_qpos(qpos)
                self.agent.robot.set_pose(sapien.Pose())

            qlimits = self.cabinet.get_qlimits()
            self.cabinet.set_qpos(qlimits[env_idx, :, 0])
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            self.handle_link_goal.set_pose(
                Pose.create_from_pq(p=self.handle_link_positions(env_idx))
            )

    def compute_dense_reward(self, obs, action, info: dict):
        tcp_to_handle_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["handle_link_pos"], axis=1
        )
        opened_frac = torch.clamp(
            self.handle_link.joint.qpos / self.target_qpos, min=0.0
        )

        reaching_reward = 0.03 * (1 - torch.tanh(5 * tcp_to_handle_dist))
        open_progress = torch.clamp(
            (opened_frac - self.open_reward_deadzone_frac)
            / (self.min_open_frac - self.open_reward_deadzone_frac),
            min=0.0,
            max=1.0,
        )
        open_reward = 0.05 * torch.square(open_progress)
        near_success_reward = 0.04 * torch.pow(open_progress, 6)
        reward = reaching_reward + open_reward + near_success_reward - 0.01

        reward = torch.clamp(reward, min=-0.01, max=0.1)
        reward[info["success"]] = 10.0 * self.max_dense_reward
        return reward

    def compute_normalized_dense_reward(self, obs, action, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.max_dense_reward


@register_env(
    "OpenCabinetDrawerEasyLevel0-v1",
    max_episode_steps=100,
    override=True,
    asset_download_ids=["partnet_mobility/1000"],
)
class OpenCabinetDrawerEasyLevel0Env(_OpenCabinetDrawerEasyBase):
    """Most forgiving curriculum stage for quick PPO bootstrapping."""


@register_env(
    "OpenCabinetDrawerEasyLevel1-v1",
    max_episode_steps=100,
    override=True,
    asset_download_ids=["partnet_mobility/1000"],
)
class OpenCabinetDrawerEasyLevel1Env(_OpenCabinetDrawerEasyBase):
    """Alias of the Level0 difficulty with the original Level1 environment id."""


@register_env(
    "OpenCabinetDrawerEasy-v1",
    max_episode_steps=100,
    override=True,
    asset_download_ids=["partnet_mobility/1000"],
)
class OpenCabinetDrawerEasyEnv(OpenCabinetDrawerEasyLevel1Env):
    """Alias of `OpenCabinetDrawerEasyLevel1-v1` for convenience."""


@register_env(
    "OpenCabinetDrawerEasyLevel2-v1",
    max_episode_steps=100,
    override=True,
    asset_download_ids=["partnet_mobility/1000"],
)
class OpenCabinetDrawerEasyLevel2Env(_OpenCabinetDrawerEasyBase):
    """Alias of the Level0 difficulty with the original Level2 environment id."""


MOBILE_ARM_VARIANT_ENV_IDS: list[str] = []
MOBILE_ARM_VARIANT_CLASSES: dict[str, type] = {}


def _register_mobile_arm_variants() -> None:
    for model_id in _CABINET_MODEL_IDS:
        for variant_spec in _VARIANT_SPECS:
            class_name = (
                f"OpenCabinetDrawerCabinet{model_id}{variant_spec['class_suffix']}Env"
            )
            env_id = (
                f"OpenCabinetDrawerCabinet{model_id}{variant_spec['env_suffix']}-v1"
            )
            variant_cls = type(
                class_name,
                (_OpenCabinetDrawerEasyBase,),
                {
                    "__doc__": (
                        f"OpenCabinetDrawer-v1 variant with cabinet {model_id} and "
                        f"scale {variant_spec['cabinet_scale']}."
                    ),
                    "__module__": __name__,
                    "ENV_ID": env_id,
                    "CABINET_MODEL_ID": model_id,
                    "CABINET_SCALE": variant_spec["cabinet_scale"],
                },
            )
            variant_cls = register_env(
                env_id,
                max_episode_steps=100,
                override=True,
                asset_download_ids=[f"partnet_mobility/{model_id}"],
            )(variant_cls)
            globals()[class_name] = variant_cls
            MOBILE_ARM_VARIANT_ENV_IDS.append(env_id)
            MOBILE_ARM_VARIANT_CLASSES[env_id] = variant_cls


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
    for env_id in (env_ids or MOBILE_ARM_VARIANT_ENV_IDS):
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


_register_mobile_arm_variants()
assert len(MOBILE_ARM_VARIANT_ENV_IDS) == len(_CABINET_MODEL_IDS) * len(_VARIANT_SPECS)

__all__ = [
    "MOBILE_ARM_VARIANT_ENV_IDS",
    "MOBILE_ARM_VARIANT_CLASSES",
    "render_env_preview",
    "save_env_preview",
    "generate_all_previews",
] + [cls.__name__ for cls in MOBILE_ARM_VARIANT_CLASSES.values()]


if __name__ == "__main__":
    preview_paths = generate_all_previews()
    print(f"registered variants: {len(MOBILE_ARM_VARIANT_ENV_IDS)}")
    for preview_path in preview_paths:
        print(f"saved preview: {preview_path}")

    """
    OpenCabinetDrawerCabinet1033ScaleUp1p3-v1
    OpenCabinetDrawerCabinet1032Default-v1
    OpenCabinetDrawerCabinet1027Default-v1
    OpenCabinetDrawerCabinet1021Default-v1
    OpenCabinetDrawerCabinet1016ScaleUp1p3-v1
    OpenCabinetDrawerCabinet1033ScaleUp1p3-v1
    OpenCabinetDrawerCabinet1032Default-v1
    OpenCabinetDrawerCabinet1027Default-v1
    OpenCabinetDrawerCabinet1021Default-v1
    OpenCabinetDrawerCabinet1016ScaleUp1p3-v1
    """