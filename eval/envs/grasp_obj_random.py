from typing import Any, Union
import time
import numpy as np
import sapien
import torch
import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig, Camera
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from .grasp_obj_cfgs import GRASP_OBJ_CONFIGS
from .agents.amazinghand import AH_RIGHT
from mani_skill.utils.structs import Actor, Link
from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    matrix_to_euler_angles,
    quaternion_to_matrix,
    quaternion_apply,
    axis_angle_to_quaternion
)
from mani_skill.utils.structs.types import Device
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig, SceneConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_raw_multiply
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actors.ycb import ASSET_DIR
from mani_skill.utils.io_utils import load_json
from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent
from sapien.render import RenderTexture2D 
from PIL import Image
from noise import pnoise2
# pnoise2 = lambda x:x
import torch.nn.functional as F
import cv2
from pathlib import Path
import os

GRASP_OBJECT_DOC_STRING = """
**Task Description:** A task where the objective is to grasp a object with the {robot_id} robot. 
Randomizations include object position, orientation, size, mass, color, 
and camera position to enable domain adaptation.
"""
# CENTER = [0.01013767, 0.06346342, 0.00019646]
AXIS = [-0.14349499, 0.98964592, 0.00318564]
# -----------------------------
# 工具函数
# -----------------------------
def normalize(img):
    img = img - img.min()
    img = img / (img.max() + 1e-6)
    return img


def to_uint8(img):
    return (img * 255).astype(np.uint8)

def get_ycb_builder(
    scene: ManiSkillScene, id: str, scale: int = 1., scene_idxs: list[int] = [], add_collision: bool = True, add_visual: bool = True
):
    YCB_DATASET = {
        "model_data": load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"),
    }
    model_db = YCB_DATASET["model_data"]

    builder = scene.create_actor_builder()
    builder.set_scene_idxs(scene_idxs)

    metadata = model_db[id]
    density = metadata.get("density", 1000)
    physical_material = None
    (metadata["bbox"]["max"][2] - metadata["bbox"]["min"][2]) * scale
    model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / id
    if add_collision:
        collision_file = str(model_dir / "collision.ply")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale, 1, scale],
            material=physical_material,
            density=density,
        )
    if add_visual:
        visual_file = str(model_dir / "textured.obj")
        builder.add_visual_from_file(filename=visual_file, scale=[scale, 1, scale])

    return builder

def get_custom_obj_builder(
    scene: ManiSkillScene, obj_type: str, scale: int, scene_idxs: list[int] = [], add_collision: bool = True, add_visual: bool = True
):
    builder = scene.create_actor_builder()
    builder.set_scene_idxs(scene_idxs)
    
    load_path = str(Path(__file__).parent.resolve() / f"assets/grasp_items/{obj_type}.obj")
    if not os.path.exists(load_path):
        load_path = str(Path(__file__).parent.resolve() / f"assets/grasp_items/{obj_type}.glb")

    if add_visual:
        builder.add_visual_from_file(filename=load_path, scale=[scale, scale, scale])
    if add_collision:
        c_load_path = str(Path(__file__).parent.resolve() / f"assets/grasp_items/{obj_type}.obj")
        if not os.path.exists(c_load_path):
            c_load_path = str(Path(__file__).parent.resolve() / f"assets/grasp_items/{obj_type}.glb")

        builder.add_multiple_convex_collisions_from_file(
            filename=c_load_path,
            scale=[scale, 2 * scale, scale],
            material=None,
            density=1000,
            decomposition="coacd",
        )

    return builder

# -----------------------------
# 基础纹理
# -----------------------------
def perlin_texture(rng, size=128, scale_range=(5, 20)):
    scale = rng.uniform(*scale_range)
    img = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            img[i, j] = pnoise2(i / scale, j / scale)
    return normalize(img)


def checker_texture(rng, size=128):
    num_checks = rng.randint(4, 12)
    s = size // num_checks

    img = np.zeros((size, size))
    for i in range(num_checks):
        for j in range(num_checks):
            img[i*s:(i+1)*s, j*s:(j+1)*s] = rng.random()
    return img


def gradient_texture(rng, size=128):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xv, yv = np.meshgrid(x, y)

    a, b = rng.random(), rng.random()
    img = a * xv + b * yv
    return normalize(img)


def blob_texture(rng, size=128, num_blobs=8):
    img = np.zeros((size, size))
    for _ in range(num_blobs):
        cx, cy = rng.randint(0, size, 2)
        r = rng.randint(size//12, size//5)

        for i in range(size):
            for j in range(size):
                if (i - cx)**2 + (j - cy)**2 < r**2:
                    img[i, j] = rng.random()
    return normalize(img)


# -----------------------------
# 🎯 核心：统一生成函数
# -----------------------------
def generate_texture(rng, color=None, size=128, mode="object"):
    """
    mode:
        "object" -> 强 randomization（操作物体）
        "table"  -> 弱 randomization（桌面/背景）
    """

    # =========================
    # 🟦 OBJECT：强随机（论文级）
    # =========================
    if mode == "object":
        textures = [
            perlin_texture(rng, size, (5, 20)),
            checker_texture(rng, size),
            gradient_texture(rng, size),
            blob_texture(rng, size)
        ]

        weights = rng.dirichlet(np.ones(len(textures)))
        gray = sum(w * t for w, t in zip(weights, textures))
        # ✅ normalize
        gray = gray - gray.mean() + 0.5
        gray = np.clip(gray, 0, 1)


        # 颜色丰富
        if color is None:
            color = rng.uniform(0.4, 1.0, (3,))
        else:
            color = np.array(color)

        if rng.random() < 0.2:
            img = np.stack([gray * c for c in color], axis=-1)
        else:
            # 纯色
            img = np.stack([np.ones((size, size)) * c for c in color], axis=-1)

        # illumination shift（增强鲁棒性）
        if rng.random() < 0.3:
            img *= rng.uniform(0.7, 1.3)

        # ✅ gamma
        img = np.power(img, rng.uniform(0.8, 1.2))

    # =========================
    # 🟩 TABLE：弱随机（稳定背景）
    # =========================
    elif mode == "table":
        # 低频为主
        gray = perlin_texture(rng, size, scale_range=(0, 50))
        # ✅ normalize
        gray = gray - gray.mean() + 0.5
        gray = np.clip(gray, 0, 1)
        
        # 限制颜色空间（更真实）
        base_colors = np.array([
            [0.6, 0.5, 0.4],  # 木
            [0.8, 0.8, 0.8],  # 灰
            [0.9, 0.9, 0.9],  # 白
        ])

        if color is None:
            color = base_colors[rng.randint(0, len(base_colors))]
        else:
            color = np.array(color)


        # if rng.random() < 0.8:
        #     img = np.stack([gray * c for c in color], axis=-1)
        # else:
        #     # 纯色
        img = np.stack([np.ones((size, size)) * c for c in color], axis=-1)

        # 小扰动（不要太强）
        img *= rng.normal(1.0, 0.05, img.shape)

        # ✅ gamma
        img = np.power(img, rng.uniform(0.8, 1.2))
    else:
        raise ValueError("mode must be 'object' or 'table'")

    # -------------------------
    # clamp & 输出
    # -------------------------
    img = np.clip(img, 0, 1)
    return to_uint8(img), color


# -----------------------------
# 🎯 材质整体（推荐）
# -----------------------------
def to_uint8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)

def to_texture(np_img):
    if np_img.dtype != np.uint8:
        np_img = to_uint8(np_img)

    # 保证 RGBA
    if np_img.shape[-1] == 3:
        alpha = np.ones((*np_img.shape[:2], 1), dtype=np.uint8) * 255
        np_img = np.concatenate([np_img, alpha], axis=-1)

    return RenderTexture2D(
        np_img,
        format="R8G8B8A8Unorm",
        mipmap_levels=1,
        filter_mode="linear",
        address_mode="repeat",
        srgb=True
    )


def generate_material(rng, color=None, size=64, mode="object"):
    # ===== base color =====
    base, color = generate_texture(rng, color, size, mode)   # 你已有函数 (float32, 0~1)

    # ===== roughness =====
    rough = perlin_texture(
        rng,
        size,
        scale_range=(10, 40) if mode == "table" else (5, 20)
    )

    if mode == "table":
        rough = rough * 0.5 + 0.5
    else:
        if rng.random() < 0.7:
            rough = rough * 0.5 + 0.5

    rough = np.stack([rough]*3, axis=-1)

    # ===== metallic =====
    if mode == "table":
        val = rng.uniform(0, 0.1)
    else:
        val = rng.uniform(0, 0.2) if rng.random() < 0.7 else rng.uniform(0.8, 1.0)

    metallic = np.ones((size, size, 3), dtype=np.float32) * val

    # ===== 转 RenderTexture2D =====
    base_tex = to_texture(base)
    rough_tex = to_texture(rough)
    metallic_tex = to_texture(metallic)

    return {
        "base_color": base_tex,
        "roughness": rough_tex,
        "metallic": metallic_tex,
        "color_value": color
    }

def random_quaternions(
    rng: BatchedRNG,
    device: Device = None,
    lock_x: bool = False,
    lock_y: bool = False,
    lock_z: bool = False,
    bounds=(0, np.pi * 2),
):
    """
    Generates random quaternions by generating random euler angles uniformly, with each of
    the X, Y, Z angles ranging from bounds[0] to bounds[1] radians. Can optionally
    choose to fix X, Y, and/or Z euler angles to 0 via lock_x, lock_y, lock_z arguments
    """
    dist = bounds[1] - bounds[0]
    xyz_angles = torch.from_numpy(rng.rand(3)).to(device) * (dist) + bounds[0]
    if lock_x:
        xyz_angles[:, 0] *= 0
    if lock_y:
        xyz_angles[:, 1] *= 0
    if lock_z:
        xyz_angles[:, 2] *= 0
    return matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))


# PickObjectRandom-v1 设置
# 若某项设置为随机，但是只想在开始时随机一次，可以添加 reconfiguration_freq=0 参数（或不添加）
# 若想每次重置时都随机，可以添加 reconfiguration_freq=1 参数，仅限eval阶段 (Maniskill不支持在train阶段更换环境)
# 若不添加reconfiguration_freq=1，reset时仅随机位置参数，不随机尺寸、质量、颜色等参数
# 因此若想在训练中改变尺寸、质量、颜色等参数，只能close当前环境，重新创建该环境
# 色温是越高越偏红，越低越偏蓝
# env_kwargs.update(
#     object_type="cube",                                     # "cube", "sphere", "cylinder", "box", 或者 None 表示随机选择
#     object_size_info={                                      # 一个字典，预设物体尺寸参数 (m)，设置为 {} 表示随机选择
#         'cube': {'half_size': 0.02},                        # cube 的边长为 0.06m
#         'sphere': {'radius': 0.03},                         # sphere 的半径为 0.03m
#         'cylinder': {'radius': 0.03, 'half_length': 0.03},  # cylinder 的半径为 0.03m，长度为 0.06m
#         'box': {'half_sizes': [0.03, 0.04, 0.05]},          # box 的长宽高分别为 0.06m, 0.08m, 0.1m
#     },                                      
#     object_mass=0.1,                                        # 物体质量 (kg)，None 表示随机选择
#     object_color=[1, 0, 0, 1],                              # 物体颜色，输入一个 RGBA 列表，None 表示随机选择，各维度 0 ~ 1 之间
#     randomize_camera=False,                                 # 是否随机摄像头位置, None 表示部分随机
#     ambient_light_temperature=2000,                         # 环境光色温，None 表示随机选择，建议范围 2000K~6500K
#     ambient_light_intensity=1.0,                            # 环境光强度，None 表示随机选择，建议范围 0.1 ~ 1.0
#     directional_light_temperature=2000,                     # 方向光色温，None 表示随机选择，建议范围 2000K~6500K
#     directional_light_direction=[1, 1, -1],                 # 方向光方向，输入一个 3 维列表，None 表示随机选择, 建议各维度 0 ~ 0.5 之间
#     shadow_scale=5,                                         # 阴影模糊程度，None 表示随机选择, 建议 5 ~ 10
#     global_random=False,                                    # 是否在随机时全局统一
# )

@register_env("GraspObjectRandom-v1", max_episode_steps=100)
class GraspObjectRandomEnv(BaseEnv):
    SUPPORTED_ROBOTS = [
        "amazinghand_right"
    ]
    agent: Union[AH_RIGHT]

    shadow_scale_range = (5, 10)
    ambient_light_intensity_range = (0.1, 1.0)
    directional_light_range = (-1, 1)
    friction_range = (0.1, 0.3)
    resitution_range = (0, 0.3)
    disruptor_count_range = (0, 5)
    disruptor_mass_range = (0.01, 5)
    disruptor_half_size_range = (0.02, 0.3)

    object_type_options = ["cube", "sphere", "cylinder", "box"]  # 可扩展到 mesh
    temperature_range = (2000, 6500)  # 色温范围
    large_object_threshold = 0.05

    init_settings = dict()

    camera_pos_range = {
        "x": (-0.05, 0.05),
        "y": (-0.5, 0.5),
        "z": (-0.05, 0.05)
    }

    def __init__(
        self,
        *args,
        robot_uids="amazinghand_right",
        robot_init_qpos_noise=0.02,
        object_type=None,                       # "cube", "sphere", "cylinder", "box", 或者 None 表示随机选择
        object_size_info={},                    # 一个字典，预设物体尺寸参数 (m)，cube 的 half_size, cylinder 的 radius 和 half_length,
                                                # box 的 half_sizes, sphere 的 radius，如果为空则随机选择 
        object_mass=None,                       # 物体质量 (kg)，None 表示随机选择
        object_color=None,                      # 物体颜色，输入一个 RGBA 列表，None 表示随机选择，各维度 0 ~ 1 之间
        randomize_camera=None,                  # 是否随机摄像头位置（True/False）, None 表示随机选择随机或不随机 （可以在部分平行环境中随机，部分环境中不随机）
        randomize_table=True,                   # 随机桌面纹理
        randomize_skybox=True,                  # 随机天空盒
        randomize_disruptors=True,              # 随机干扰物（桌面上随机生成一些小物体，增加复杂度）
        randomize_blur=True,                    # 随机模糊图像
        ambient_light_temperature=None,         # 环境光色温，None 表示随机选择，建议范围 2000K~6500K
        ambient_light_intensity=None,           # 环境光强度，None 表示随机选择，建议范围 0.1 ~ 1.0
        directional_light_temperature=None,     # 方向光色温，None 表示随机选择，建议范围 2000K~6500K
        directional_light_direction=None,       # 方向光方向，输入一个 3 维列表，None 表示随机选择, 建议各维度 0 ~ 0.5 之间
        shadow_scale=None,                      # 阴影模糊程度，None 表示随机选择, 建议 5 ~ 10
        default_friction=True,                   # 是否使用默认摩擦参数，True/False
        dynamic_friction=None,              # 动态摩擦系数范围，None 表示随机选择
        static_friction=None,               # 静态摩擦系数范围，None 表示随机选择
        restitution=None,                   # 恢复系数范围，None 表示随机选择
        reconfiguration_freq=None,
        global_random=False,                    # 是否在随机时全局统一
        default_lighting=False,                   # 是否使用默认光照设置，True/False
        default_grasp_items=True,                   # 是否使用默认抓取物体（YCB），True/False
        restart_id=0,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.object_type = object_type
        self.randomize_camera = randomize_camera
        self.randomize_table = randomize_table
        self.randomize_skybox = randomize_skybox
        self.randomize_disruptors = randomize_disruptors
        self.object_size_info = object_size_info
        self.object_mass = object_mass
        self.object_color = object_color
        self.ambient_light_temperature = ambient_light_temperature
        self.ambient_light_intensity = ambient_light_intensity
        self.directional_light_temperature = directional_light_temperature
        self.directional_light_direction = directional_light_direction
        self.shadow_scale = shadow_scale
        self.dynamic_friction = dynamic_friction
        self.default_friction = default_friction
        self.static_friction = static_friction
        self.restitution = restitution
        self.randomize_blur = randomize_blur
        self.default_lighting = default_lighting
        self.default_grasp_items = default_grasp_items

        self.init_settings.update(
            object_type=object_type,
            object_size_info=object_size_info,
            object_mass=object_mass,
            object_color=object_color,
            randomize_camera=randomize_camera,
            randomize_table=randomize_table,
            randomize_disruptors=randomize_disruptors,
            ambient_light_temperature=ambient_light_temperature,
            ambient_light_intensity=ambient_light_intensity,
            directional_light_temperature=directional_light_temperature,
            directional_light_direction=directional_light_direction,
            shadow_scale=shadow_scale,
            dynamic_friction=dynamic_friction,
            static_friction=static_friction,
            restitution=restitution,
            disruptor_size_info={}
        )

        if robot_uids in GRASP_OBJ_CONFIGS:
            cfg = GRASP_OBJ_CONFIGS[robot_uids]
        else:
            cfg = GRASP_OBJ_CONFIGS["panda"]
        
        self.robot_uids = robot_uids
        self.sensor_cam_eye_pos = np.array(cfg["sensor_cam_eye_pos"])
        self.sensor_cam_target_pos = np.array(cfg["sensor_cam_target_pos"])
        self.human_cam_eye_pos = np.array(cfg["human_cam_eye_pos"])
        self.human_cam_target_pos = np.array(cfg["human_cam_target_pos"])
        self.object_spawn_half_size_range = cfg["object_spawn_half_size_range"]
        self.object_mass_range = cfg["object_mass_range"]
        self.object_spawn_radius = cfg["object_spawn_radius"]
        self.object_spawn_center = cfg["object_spawn_center"]
        # self.reconfig_freq = reconfiguration_freq
        self.num_envs = kwargs.get("num_envs", 1)
        # self.reconfig_counters = [reconfiguration_freq if reconfiguration_freq is not None else 0] * self.num_envs
        self.per_env_settings = [{} for _ in range(self.num_envs)]
        self._direct_lights = [None for _ in range(self.num_envs)]
        self.objects = [None] * self.num_envs
        self.object = None
        self.cam_mount = None
        self.cam_mount_human = None
        self.half_heights = [0.0] * self.num_envs
        self.has_inited = False
        self.global_random = global_random
        self.global_camera_pose = None
        self.restart_id = restart_id
        self.rngs = None
        self.agent_root_pos = None
        self._disruptors = None
        super().__init__(*args, robot_uids=robot_uids, reconfiguration_freq=reconfiguration_freq, **kwargs)
        

    @property
    def _default_sensor_configs(self):
        # pose = self._get_random_camera_pose(self.sensor_cam_eye_pos, self.sensor_cam_target_pos, self.num_envs)
        return [CameraConfig("base_camera", sapien.Pose(), 128, 128, np.pi / 2, 0.01, 100, mount=self.cam_mount)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _initialize_own_rngs(self):
        episode_rng = self._batched_episode_rng
        # 1️⃣ 批量从官方 RNG 派生 seeds（向量化）
        derived_seeds = episode_rng.randint(0, 2**31 - 1)

        # 2️⃣ 引入 restart 偏移，避免重启重复
        derived_seeds = derived_seeds + self.restart_id * 99991

        # 3️⃣ 构造内部 RNG 列表（每 env 一个）
        self.rngs = BatchedRNG.from_seeds(
            derived_seeds.tolist(),
            backend=self._batched_rng_backend
        )

    def _randomize_objects(self, rng, object_types, b):
        size_infos = []
        half_heights = []

        def sample_size_val(rng_local):
            return rng_local.uniform(*self.object_spawn_half_size_range, size=1)[0]
        
        def sample_ycb_size_val(rng_local):
            return rng_local.uniform(1, 2, size=1)[0]

        if self.global_random:
            base_rng = rng[0]
            obj_type = object_types[0]
            size_info = {}
            half_height = 0.0

            if obj_type == "cube":
                size_info["half_size"] = float(
                    self.init_settings["object_size_info"].get(obj_type, {}).get("half_size", sample_size_val(base_rng))
                )
                half_height = size_info["half_size"]

            elif obj_type == "sphere":
                size_info["radius"] = float(
                    self.init_settings["object_size_info"].get(obj_type, {}).get("radius", sample_size_val(base_rng))
                )
                half_height = size_info["radius"]

            elif obj_type == "cylinder":
                size_info["radius"] = float(
                    self.init_settings["object_size_info"].get(obj_type, {}).get("radius", sample_size_val(base_rng))
                )
                size_info["half_length"] = float(
                    self.init_settings["object_size_info"].get(obj_type, {}).get("half_length", sample_size_val(base_rng))
                )
                half_height = size_info["radius"]

            elif obj_type == "box":  # box
                size_info["half_sizes"] = self.init_settings["object_size_info"].get(obj_type, {}).get(
                    "half_sizes", base_rng.uniform(*self.object_spawn_half_size_range, size=(3,)).tolist()
                )
                half_height = size_info["half_sizes"][-1]

            else:
                size_info["scale"] = float(
                    self.init_settings["object_size_info"].get(obj_type, {}).get("scale", sample_ycb_size_val(base_rng))
                )

            for _ in range(b):
                size_infos.append({obj_type: size_info.copy()})
                half_heights.append(half_height)

        else:
            for j in range(b):
                obj_type = object_types[j]
                local_rng = rng[j]
                size_info = {}
                half_height = 0.0

                if obj_type == "cube":
                    size_info["half_size"] = float(
                        self.init_settings["object_size_info"].get(obj_type, {}).get("half_size", sample_size_val(local_rng))
                    )
                    half_height = size_info["half_size"]

                elif obj_type == "sphere":
                    size_info["radius"] = float(
                        self.init_settings["object_size_info"].get(obj_type, {}).get("radius", sample_size_val(local_rng))
                    )
                    half_height = size_info["radius"]

                elif obj_type == "cylinder":
                    size_info["radius"] = float(
                        self.init_settings["object_size_info"].get(obj_type, {}).get("radius", sample_size_val(local_rng))
                    )
                    size_info["half_length"] = float(
                        self.init_settings["object_size_info"].get(obj_type, {}).get("half_length", sample_size_val(local_rng))
                    )
                    half_height = size_info["radius"]

                elif obj_type == "box":  # box
                    size_info["half_sizes"] = self.init_settings["object_size_info"].get(obj_type, {}).get(
                        "half_sizes", local_rng.uniform(*self.object_spawn_half_size_range, size=(3,)).tolist()
                    )
                    half_height = size_info["half_sizes"][-1]

                else:
                    size_info["scale"] = float(
                        self.init_settings["object_size_info"].get(obj_type, {}).get("scale", sample_ycb_size_val(local_rng))
                    )

                size_infos.append({obj_type: size_info})
                half_heights.append(half_height)

        return size_infos, half_heights

    def _randomize_disruptors(self, rng, disruptor_types, b):
        size_infos = []
        half_heights = []
        colors = []
        counts = []
        types = []

        min_n, max_n = self.disruptor_count_range

        # ⭐ ===== 总开关 =====
        if not self.randomize_disruptors:
            for _ in range(b):
                counts.append(0)
                size_infos.append([None] * max_n)
                types.append([None] * max_n)
                half_heights.append([0.0] * max_n)
                colors.append([None] * max_n)
            return counts, types, size_infos, half_heights, colors

        # =========================
        # 正常 randomization
        # =========================

        def sample_size_val(local_rng):
            return float(local_rng.uniform(*self.disruptor_half_size_range))

        def sample_color(local_rng):
            return local_rng.uniform(0.2, 0.9, size=(3,)).tolist()

        # ===== 数量 =====
        if self.global_random:
            base_rng = rng[0]
            n = int(base_rng.randint(min_n, max_n + 1))
            counts = [n] * b
        else:
            for j in range(b):
                local_rng = rng[j]
                n = int(local_rng.randint(min_n, max_n + 1))
                counts.append(n)

        # =========================================================
        # 🔥 global_random：只生成一次
        # =========================================================
        if self.global_random:

            shared_rng = rng[0]
            shared_n = counts[0]

            shared_types = []          # ⭐ 新增
            shared_size_infos = []
            shared_half_heights = []
            shared_colors = []

            for i in range(max_n):

                if i >= shared_n:
                    shared_types.append(None)
                    shared_size_infos.append(None)
                    shared_half_heights.append(0.0)
                    shared_colors.append(None)
                    continue

                # ✅ 类型随机（关键）
                obj_type = shared_rng.choice(disruptor_types)

                size_info = {}
                half_height = 0.0

                if obj_type == "cube":
                    val = float(
                        self.init_settings["disruptor_size_info"]
                        .get(obj_type, {})
                        .get("half_size", sample_size_val(shared_rng))
                    )
                    size_info["half_size"] = val
                    half_height = val

                elif obj_type == "sphere":
                    val = float(
                        self.init_settings["disruptor_size_info"]
                        .get(obj_type, {})
                        .get("radius", sample_size_val(shared_rng))
                    )
                    size_info["radius"] = val
                    half_height = val

                elif obj_type == "cylinder":
                    r = float(
                        self.init_settings["disruptor_size_info"]
                        .get(obj_type, {})
                        .get("radius", sample_size_val(shared_rng))
                    )
                    h = float(
                        self.init_settings["disruptor_size_info"]
                        .get(obj_type, {})
                        .get("half_length", sample_size_val(shared_rng))
                    )
                    size_info["radius"] = r
                    size_info["half_length"] = h
                    half_height = r  # XY 用 radius

                else:  # box
                    sizes = self.init_settings["disruptor_size_info"].get(
                        obj_type, {}
                    ).get(
                        "half_sizes",
                        shared_rng.uniform(
                            *self.object_spawn_half_size_range, size=(3,)
                        ).tolist(),
                    )
                    size_info["half_sizes"] = sizes
                    half_height = sizes[-1]

                color = sample_color(shared_rng) + [1]

                # ⭐ 全部记录
                shared_types.append(obj_type)
                shared_size_infos.append({obj_type: size_info})
                shared_half_heights.append(half_height)
                shared_colors.append(color)

            # ✅ 广播
            for j in range(b):
                env_types = []
                env_size_infos = []
                env_half_heights = []
                env_colors = []

                for i in range(max_n):
                    if i >= shared_n:
                        env_types.append(None)
                        env_size_infos.append(None)
                        env_half_heights.append(0.0)
                        env_colors.append(None)
                    else:
                        env_types.append(shared_types[i])
                        env_size_infos.append(shared_size_infos[i])
                        env_half_heights.append(shared_half_heights[i])
                        env_colors.append(shared_colors[i])

                types.append(env_types)
                size_infos.append(env_size_infos)
                half_heights.append(env_half_heights)
                colors.append(env_colors)

        # =========================================================
        # 🔥 per-env random
        # =========================================================
        else:

            for j in range(b):
                local_rng = rng[j]
                n = counts[j]

                env_types = []          # ⭐ 新增
                env_size_infos = []
                env_half_heights = []
                env_colors = []

                for i in range(max_n):

                    if i >= n:
                        env_types.append(None)
                        env_size_infos.append(None)
                        env_half_heights.append(0.0)
                        env_colors.append(None)
                        continue

                    obj_type = local_rng.choice(disruptor_types)

                    size_info = {}
                    half_height = 0.0

                    if obj_type == "cube":
                        val = float(
                            self.init_settings["disruptor_size_info"]
                            .get(obj_type, {})
                            .get("half_size", sample_size_val(local_rng))
                        )
                        size_info["half_size"] = val
                        half_height = val

                    elif obj_type == "sphere":
                        val = float(
                            self.init_settings["disruptor_size_info"]
                            .get(obj_type, {})
                            .get("radius", sample_size_val(local_rng))
                        )
                        size_info["radius"] = val
                        half_height = val

                    elif obj_type == "cylinder":
                        r = float(
                            self.init_settings["disruptor_size_info"]
                            .get(obj_type, {})
                            .get("radius", sample_size_val(local_rng))
                        )
                        h = float(
                            self.init_settings["disruptor_size_info"]
                            .get(obj_type, {})
                            .get("half_length", sample_size_val(local_rng))
                        )
                        size_info["radius"] = r
                        size_info["half_length"] = h
                        half_height = r

                    else:
                        sizes = self.init_settings["disruptor_size_info"].get(
                            obj_type, {}
                        ).get(
                            "half_sizes",
                            local_rng.uniform(
                                *self.object_spawn_half_size_range, size=(3,)
                            ).tolist(),
                        )
                        size_info["half_sizes"] = sizes
                        half_height = sizes[-1]

                    color = sample_color(local_rng) + [1]

                    env_types.append(obj_type)
                    env_size_infos.append({obj_type: size_info})
                    env_half_heights.append(half_height)
                    env_colors.append(color)

                types.append(env_types)
                size_infos.append(env_size_infos)
                half_heights.append(env_half_heights)
                colors.append(env_colors)

        # =========================================================
        # ✅ 返回
        # =========================================================
        return counts, types, size_infos, half_heights, colors

    def _reset_settings(self, env_idx=None):
        def to_torch(x, dtype=torch.float32):
            return torch.from_numpy(np.asarray(x)).to(self.device, dtype=dtype)

        def sample_uniform(rng, low, high, shape):
            if self.global_random:
                val = rng[0].uniform(low, high, size=shape)  # shape=(3,) or ()
                val = np.broadcast_to(val, (b, *shape))
            else:
                val = rng.uniform(low, high, size=shape)
            return val

        def sample_int(rng, low, high, shape=()):
            if self.global_random:
                val = rng[0].randint(low, high, size=shape)
                val = np.broadcast_to(val, (b, *shape))
            else:
                val = rng.randint(low, high, size=shape)
            return val

        # -------- select env batch --------
        if env_idx is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            rng = self.rngs
        else:
            env_ids = env_idx.to(self.device)
            rng = self.rngs[env_idx.tolist()]

        b = len(env_ids)

        # -------- object type --------
        if self.init_settings["object_type"] is None:
            type_idx = sample_int(rng, 0, len(self.object_type_options))
            type_idx = to_torch(type_idx, dtype=torch.long)
            object_types = [self.object_type_options[i.item()] for i in type_idx]
        else:
            object_types = [self.init_settings["object_type"]] * b

        # -------- object mass --------
        if self.init_settings["object_mass"] is None:
            mass = sample_uniform(rng, *self.object_mass_range, shape=())
            object_mass = to_torch(mass).reshape(-1)
        else:
            object_mass = torch.full((b,), self.init_settings["object_mass"], device=self.device)

        # -------- object color --------
        if self.init_settings["object_color"] is None:
            # TODO: 可以改成更丰富的纹理随机化，而不是纯色
            rgb = sample_uniform(rng, 0.0, 1.0, shape=(3,))
            rgb = to_torch(rgb)
            alpha = torch.ones((b, 1), device=self.device)
            object_color = torch.cat([rgb, alpha], dim=1)
        else:
            fixed = torch.tensor(self.init_settings["object_color"], device=self.device)
            object_color = fixed.repeat(b, 1)

        # -------- table randomization --------
        if self.init_settings["randomize_table"] is None:
            tab_flag = sample_int(rng, 0, 2)
            randomize_table = to_torch(tab_flag, dtype=torch.bool).reshape(-1)
        else:
            randomize_table = torch.full((b,), self.init_settings["randomize_table"], device=self.device)

        # -------- camera randomization --------
        if self.init_settings["randomize_camera"] is None:
            cam_flag = sample_int(rng, 0, 2)
            randomize_camera = to_torch(cam_flag, dtype=torch.bool).reshape(-1)
        else:
            randomize_camera = torch.full((b,), self.init_settings["randomize_camera"], device=self.device)

        # -------- lighting --------
        if self.init_settings["ambient_light_temperature"] is None:
            ambient_temp = to_torch(sample_uniform(rng, *self.temperature_range, shape=())).reshape(-1)
        else:
            ambient_temp = torch.full((b,), self.init_settings["ambient_light_temperature"], device=self.device)

        if self.init_settings["ambient_light_intensity"] is None:
            ambient_intensity = to_torch(sample_uniform(rng, *self.ambient_light_intensity_range, shape=())).reshape(-1)
        else:
            ambient_intensity = torch.full((b,), self.init_settings["ambient_light_intensity"], device=self.device)

        if self.init_settings["directional_light_temperature"] is None:
            directional_temp = to_torch(sample_uniform(rng, *self.temperature_range, shape=())).reshape(-1)
        else:
            directional_temp = torch.full((b,), self.init_settings["directional_light_temperature"], device=self.device)

        if self.init_settings["directional_light_direction"] is None:
            directional_dir = to_torch(sample_uniform(rng, *self.directional_light_range, shape=(3,)))
        else:
            fixed = torch.tensor(self.init_settings["directional_light_direction"], device=self.device)
            directional_dir = fixed.repeat(b, 1)

        if self.init_settings["shadow_scale"] is None:
            shadow_scale = to_torch(sample_uniform(rng, *self.shadow_scale_range, shape=())).reshape(-1)
        else:
            shadow_scale = torch.full((b,), self.init_settings["shadow_scale"], device=self.device)

        # -------- object size info --------
        size_infos, half_heights = self._randomize_objects(rng, object_types, b)

        # -------- disruptors --------
        disruptor_counts, disruptor_types, disruptor_size_infos, disruptor_half_heights, disruptor_colors = self._randomize_disruptors(rng, object_types, b)

        # -------- friction randomization (optional, can be done in _partial_load_scene) --------
        if self.init_settings["dynamic_friction"] is None:
            dynamic_friction = to_torch(sample_uniform(rng, *self.friction_range, shape=())).reshape(-1)
        else:
            dynamic_friction = torch.full((b,), self.init_settings["dynamic_friction"], device=self.device)

        if self.init_settings["static_friction"] is None:
            static_friction = to_torch(sample_uniform(rng, *self.friction_range, shape=())).reshape(-1)
        else:
            static_friction = torch.full((b,), self.init_settings["static_friction"], device=self.device)

        if self.init_settings["restitution"] is None:
            restitution = to_torch(sample_uniform(rng, *self.resitution_range, shape=())).reshape(-1)
        else:
            restitution = torch.full((b,), self.init_settings["restitution"], device=self.device)

        # -------- skybox --------
        

        # -------- write back per-env --------
        for j, env_id in enumerate(env_ids.tolist()):
            self.per_env_settings[env_id] = {
                "object_type": object_types[j],
                "object_mass": float(object_mass[j].item()),
                "object_color": object_color[j].tolist() if self.object_color is not None else None,
                "object_size_info": size_infos[j],
                "half_height": half_heights[j],
                "randomize_camera": bool(randomize_camera[j].item()),
                "randomize_table": bool(randomize_table[j].item()),
                "ambient_light_temperature": float(ambient_temp[j].item()),
                "ambient_light_intensity": float(ambient_intensity[j].item()),
                "directional_light_temperature": float(directional_temp[j].item()),
                "directional_light_direction": directional_dir[j].tolist(),
                "shadow_scale": float(shadow_scale[j].item()),
                "dynamic_friction": float(dynamic_friction[j].item()),
                "static_friction": float(static_friction[j].item()),
                "restitution": float(restitution[j].item()),
                "disruptor_count": int(disruptor_counts[j]),
                "disruptor_size_infos": disruptor_size_infos[j],
                "disruptor_type": disruptor_types[j],
                "disruptor_half_heights": disruptor_half_heights[j],
                "disruptor_colors": disruptor_colors[j],
            }

    def _get_random_camera_pose(self, base_eye, target, env_idx=None, max_offset=0.1):
        """
        Vector-env safe camera pose randomization.
        global_random=True -> randomize ONCE and cache globally.
        """

        # -------- env batch --------
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        rng = self.rngs[env_idx.tolist()]
        b = len(env_idx)

        random_camera_mask = torch.tensor(
            [self.per_env_settings[i]["randomize_camera"] for i in env_idx.tolist()],
            device=self.device,
            dtype=torch.bool,
        )

        old_pose = self.cam_mount.pose
        old_pose_tensor = old_pose.raw_pose[env_idx]

        # ============================================================
        # GLOBAL RANDOM MODE: reuse cached camera pose if exists
        # ============================================================
        if self.global_random and self.global_camera_pose is not None:
            raw = self.global_camera_pose.raw_pose
            if raw.shape[0] == 1 and b > 1:
                raw = raw.repeat(b, 1)
            elif raw.shape[0] != b:
                raw = raw[env_idx]
            return Pose.create(raw)

        # ============================================================
        # Sample camera eye position
        # ============================================================
        if self.global_random:
            # cache global camera eye once
            if not hasattr(self, "_cached_camera_eye"):
                r = self.rngs[0]
                offset = np.concatenate([
                    r.uniform(*self.camera_pos_range["x"], size=1),
                    r.uniform(*self.camera_pos_range["y"], size=1),
                    r.uniform(*self.camera_pos_range["z"], size=1),
                ], axis=0)
                self._cached_camera_eye = base_eye + offset

            eye = self._cached_camera_eye
        else:
            offset = np.concatenate([
                rng.uniform(*self.camera_pos_range["x"], size=1),
                rng.uniform(*self.camera_pos_range["y"], size=1),
                rng.uniform(*self.camera_pos_range["z"], size=1),
            ], axis=1)

            eye = torch.tensor(base_eye + offset, device=self.device, dtype=torch.float32)
            target = torch.tensor(target, device=self.device, dtype=torch.float32).expand(b, -1)

        # build base pose
        new_pose = sapien_utils.look_at(eye=eye, target=target, device=self.device)
        new_pose_tensor = new_pose.raw_pose

        # apply randomize_camera mask
        mask = random_camera_mask.view(-1, 1)
        base_pose_raw = torch.where(mask, new_pose_tensor, old_pose_tensor)
        base_pose = Pose.create(base_pose_raw)

        # ============================================================
        # SE(3) noise (freeze if global_random)
        # ============================================================
        if self.global_random:
            if not hasattr(self, "_cached_camera_noise"):
                r = rng[0]

                p = torch.from_numpy(
                    r.uniform(-0.025, 0.025, size=(3,))
                ).to(self.device, dtype=torch.float32)

                q = random_quaternions(
                    rng=r,
                    device=self.device,
                    bounds=(-np.pi / 24, np.pi / 24),
                )

                self._cached_camera_noise = Pose.create_from_pq(p=p, q=q)

            noise_raw = self._cached_camera_noise.raw_pose.repeat(b, 1)

        else:
            p = torch.from_numpy(
                rng.uniform(-0.025, 0.025, size=(3,))
            ).to(self.device, dtype=torch.float32)
            q = random_quaternions(
                rng=rng,
                device=self.device,
                bounds=(-np.pi / 24, np.pi / 24),
            )
            noise_raw = Pose.create_from_pq(p=p, q=q).raw_pose

        # apply mask to noise (identity if no randomization)
        identity = torch.zeros_like(noise_raw)
        identity[..., 3] = 1.0  # quaternion w = 1

        noise_raw = torch.where(mask, noise_raw, identity)
        noise_pose = Pose.create(noise_raw)

        # ============================================================
        # Final pose
        # ============================================================
        final_pose = base_pose * noise_pose

        # cache global final pose (first time only)
        if self.global_random and self.global_camera_pose is None:
            self.global_camera_pose = Pose.create(final_pose.raw_pose.clone())
        
        return final_pose

    def _load_agent(self, options: dict):
        if self.robot_uids == "amazinghand_right":
            q = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([0, -np.pi / 2, -np.pi / 2]), convention="XYZ"))
            o_pose = sapien.Pose(
                p=[-0.615, 0, 0.07], 
                q=q
            )
            super()._load_agent(options, o_pose)
            # print('a')
            o_pose = np.concatenate([o_pose.p, o_pose.q], axis=-1)
            if len(o_pose.shape) == 1:
                o_pose = o_pose[None, :]
            self.o_pose = torch.tensor(o_pose).to(self.device, dtype=torch.float32)

    def _partial_load_lighting(self, options: dict, init: bool = False):
        """Apply per-env lighting settings to the sub-scenes using per_env_settings only."""
        
        # 选择目标 env
        env_idx = options.get("env_idx", None)
        target_envs = range(self.num_envs) if env_idx is None else env_idx.tolist()

        for i in target_envs:
            scene = self.scene.sub_scenes[i]

            # 删除旧方向光
            if not init and self._direct_lights[i] is not None:
                scene.remove_light(self._direct_lights[i])

            cfg = self.per_env_settings[i]  # 使用 per-env 设置

            # ---- ambient light ----
            t_ambient = cfg["ambient_light_temperature"]
            intensity_ambient = cfg["ambient_light_intensity"]
            r = min(1.0, max(0.0, 1.0 - (6500 - t_ambient) / 10000))
            g = min(1.0, max(0.0, 1.0 - abs(5000 - t_ambient) / 20000))
            b = min(1.0, max(0.0, 1.0 - (t_ambient - 3000) / 10000))
            ambient_light = [c * intensity_ambient for c in [r, g, b]]
            scene.ambient_light = ambient_light

            # ---- directional light color ----
            t_directional = cfg["directional_light_temperature"]
            r = min(1.0, max(0.0, 1.0 - (6500 - t_directional) / 10000))
            g = min(1.0, max(0.0, 1.0 - abs(5000 - t_directional) / 20000))
            b = min(1.0, max(0.0, 1.0 - (t_directional - 3000) / 10000))
            directional_light_color = [r, g, b]

            # ---- directional light direction ----
            directional_light_direction = cfg["directional_light_direction"]

            # ---- shadow scale ----
            shadow_scale = cfg["shadow_scale"]

            # ---- apply to sub-scene ----
            self._direct_lights[i] = scene.add_directional_light(
                direction=directional_light_direction,
                color=directional_light_color,
                shadow=True,
                shadow_scale=shadow_scale,
                shadow_map_size=4096,
            )

    def _load_lighting(self, options: dict):
        if not self.default_lighting:
            self._partial_load_lighting(options, init=True)
        else:
            # """Loads lighting into the scene. Called by `self._reconfigure`. If not overriden will set some simple default lighting"""
            shadow = self.enable_shadow
            self.scene.set_ambient_light([0.3, 0.3, 0.3])
            self.scene.add_directional_light(
                [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
            )
            self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _compute_object_radius(self, object_type, size_info):
        if object_type == "box":
            # size_info: {half_size: [x, y, z]}
            half_size = size_info["half_sizes"]
            return float(np.linalg.norm(half_size[:2]))

        elif object_type == "sphere":
            # size_info: {radius: r}
            return float(size_info["radius"])

        elif object_type == "cylinder":
            # size_info: {radius: r, half_length: h}
            r = size_info["radius"]
            return float(r)
        elif object_type == "cube":
            # size_info: {half_size: s}
            half_size = size_info["half_size"]
            return float(np.linalg.norm([half_size, half_size]))
        else:
            # fallback（防炸）
            return 0.05

    def _partial_load_scene(self, options: dict, init: bool = False):
        # 选择目标 env
        env_idx = options.get("env_idx", None)
        target_envs = range(self.num_envs) if env_idx is None else env_idx.tolist()

        if init:
            # if self.global_camera_pose is not None:
            #     print(self.global_camera_pose.raw_pose[0])
            self.global_camera_pose = None
            if hasattr(self, "_cached_camera_eye"):
                del self._cached_camera_eye
            if hasattr(self, "_cached_camera_noise"):
                del self._cached_camera_noise
            builder_mount = self.scene.create_actor_builder()
            builder_mount.set_initial_pose(pose=sapien.Pose())
            self.cam_mount = builder_mount.build_kinematic("camera_mount")

            # ---- disruptors ----
            # TODO：修改为并行环境版本（现在是串行的，后续改成并行的）
            self._disruptors = []
            self._disruptors_radius = []
            for idx, env_id in enumerate(target_envs):
                cfg = self.per_env_settings[env_id]
                shape_types = cfg["disruptor_type"]
                count = cfg["disruptor_count"]
                half_heights = cfg["disruptor_half_heights"]
                size_infos = cfg["disruptor_size_infos"]
                colors = cfg["disruptor_colors"]

                # ⭐ 关键：拿到当前 env 对应的 sub_scene
                sub_scene = self.scene.sub_scenes[idx]
                env_disruptors = []
                radius = []

                for i in range(count):
                    shape_type = shape_types[i]
                    half_height = half_heights[i]
                    size_info = size_infos[i][shape_type]

                    d_r = self._compute_object_radius(shape_type, size_info)
                    radius.append(d_r)
                    # ⭐ 在 sub_scene 上建 builder
                    builder = sub_scene.create_actor_builder()

                    # ===== 几何 =====
                    if shape_type == "box":
                        builder.add_box_collision(half_size=size_info['half_sizes'])
                        builder.add_box_visual(half_size=size_info['half_sizes'])

                    elif shape_type == "sphere":
                        builder.add_sphere_collision(radius=size_info['radius'])
                        builder.add_sphere_visual(radius=size_info['radius'])

                    elif shape_type == "cylinder":
                        builder.add_cylinder_collision(radius=size_info['radius'], half_length=size_info['half_length'])
                        builder.add_cylinder_visual(radius=size_info['radius'], half_length=size_info['half_length'])

                    elif shape_type == "box":
                        builder.add_box_collision(half_size=[size_info['half_size'], size_info['half_size'], size_info['half_size']])
                        builder.add_box_visual(half_size=[size_info['half_size'], size_info['half_size'], size_info['half_size']])

                    else:
                        builder = get_ycb_builder(sub_scene, shape_type, scale=size_info['scale'])

                    actor = builder.build(name=f"disruptor_{env_id}_{i}")

                    # ❗ 不要 set_pose

                    # ===== 颜色 =====
                    render_comp = actor.find_component_by_type(sapien.render.RenderBodyComponent)
                    if render_comp is not None:
                        if colors is not None:
                            color = colors[i]
                            for shape in render_comp.render_shapes:
                                mat = shape.material
                                mat.set_base_color(color)

                    env_disruptors.append(actor)

                self._disruptors.append(env_disruptors)
                self._disruptors_radius.append(radius)

            # self.cam_mount_human = self.scene.create_actor_builder().build_kinematic("camera_mount_human")
        else:
            self.remove_from_state_dict_registry(self.object)

        self.object_r = [0.0] * self.num_envs
        for i in target_envs:
            # scene = self.scene.sub_scenes[i]
            cfg = self.per_env_settings[i]

            object_type = cfg["object_type"]
            object_color = cfg["object_color"]
            size_info = cfg["object_size_info"][object_type]  # 直接取 size_info
            half_height = cfg["half_height"]
            object_r = self._compute_object_radius(object_type, size_info)
            self.object_r[i] = object_r

            # ---- 删除旧对象 ----
            if not init and self.objects[i] is not None:
                temp_pose = self.objects[i].pose.raw_pose
                temp_pose[..., :3] += 99999
                self.objects[i].pose = temp_pose

            # ---- build object ----
            if object_type in self.object_type_options:
                obj = getattr(actors, f"build_{object_type}")(
                    self.scene, 
                    color=object_color if object_color is not None else [1, 0, 0, 1], 
                    scene_idxs=[i], 
                    name=f"object_{i}_{time.time()}",
                    # body_type='kinematic',
                    initial_pose=sapien.Pose(p=[0, 0, half_height]),
                    **size_info)
            else:
                if self.default_grasp_items:
                    builder = get_ycb_builder(
                        self.scene,
                        object_type, 
                        scene_idxs=[i], 
                        scale=size_info['scale']
                    )
                else:
                    builder = get_custom_obj_builder(
                        self.scene,
                        object_type,
                        scene_idxs=[i],
                        scale=size_info['scale']
                    )
                builder.set_initial_pose(sapien.Pose(p=[0, 0, 1]))
                obj = builder.build(name=f"object_{i}_{time.time()}")
            self.objects[i] = obj
            self.remove_from_state_dict_registry(self.objects[i])

        # actors.get_actor_builder

        self.object = Actor.merge(self.objects, name="object")
        
        self.wall = actors.build_box(
            self.scene,
            half_sizes=[0.01, 1, 1],
            color=[0.8, 0.8, 0.8, 1],
            scene_idxs=target_envs,
            name="wall",
            body_type='kinematic',
            initial_pose=sapien.Pose(p=[-1, 0, 0.5]),
            add_collision=False
        )

        # self.scene.create_drive
        # self.object.set_disable_gravity(True)
        self.add_to_state_dict_registry(self.object)
        self.object.set_angular_damping(0.99)
        self.object.set_linear_damping(0.99)

        # table 纹理
        table = self.table_scene.table
        textures = None
        table_colors = []
        for i, obj in enumerate(table._objs):
        # modify the i-th object which is in parallel environment i
            if self.global_random:
                i = 0
            cfg = self.per_env_settings[i]
            randomize_table = cfg["randomize_table"]

            if isinstance(self.object, Link):
                obj = obj.entity
            
            render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
            if randomize_table:
                if self.global_random and textures is None or not self.global_random:
                    textures = generate_material(self.rngs[i], mode='table')
                base_color = textures['base_color']
                roughness = textures['roughness']
                metallic = textures['metallic']
                table_colors.append(textures['color_value'])
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        # you can change color, use texture files etc.
                        # note that textures must use the sapien.render.RenderTexture2D 
                        # object which allows passing a texture image file path
                        part.material.set_base_color_texture(base_color)
                        # part.material.set_normal_texture(None)
                        # part.material.set_emission_texture(None)
                        # part.material.set_transmission_texture(None)
                        part.material.set_metallic_texture(metallic)
                        part.material.set_roughness_texture(roughness)

        # object 纹理
        textures = None
        for i, obj in enumerate(self.object._objs):
        # modify the i-th object which is in parallel environment i
            if self.global_random:
                i = 0
            cfg = self.per_env_settings[i]
            dynamic_friction = cfg["dynamic_friction"]
            static_friction = cfg["static_friction"]
            resitiution = cfg["restitution"]
            object_mass = cfg["object_mass"]
            object_color = cfg["object_color"]

            if isinstance(self.object, Link):
                obj = obj.entity
            rigid_body_component: PhysxRigidBodyComponent = obj.find_component_by_type(PhysxRigidBodyComponent)
            if rigid_body_component is not None:
                # modifying physical properties e.g. randomizing mass from 0.1 to 1kg
                # note the use of _batched_episode_rng instead of torch.rand. _batched_episode_rng helps ensure reproducibility in parallel environments.
                rigid_body_component.mass = object_mass

                # modifying per collision shape properties such as friction values
                if not self.default_friction:
                    for shape in rigid_body_component.collision_shapes:
                        shape.physical_material.dynamic_friction = dynamic_friction
                        shape.physical_material.static_friction = static_friction
                        shape.physical_material.restitution = resitiution
            
            render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)

            if object_color is None:
                if self.global_random and textures is None or not self.global_random:
                    textures = generate_material(self.rngs[i])
                    if self.randomize_table:
                        while np.linalg.norm(textures['color_value'] - table_colors[i]) < 0.3:
                            textures = generate_material(self.rngs[i])
                base_color = textures['base_color']
                roughness = textures['roughness']
                metallic = textures['metallic']
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        # you can change color, use texture files etc.
                        # note that textures must use the sapien.render.RenderTexture2D 
                        # object which allows passing a texture image file path
                        part.material.set_base_color_texture(base_color)
                        # part.material.set_normal_texture(None)
                        # part.material.set_emission_texture(None)
                        # part.material.set_transmission_texture(None)
                        part.material.set_metallic_texture(metallic)
                        part.material.set_roughness_texture(roughness)

    def _load_scene(self, options: dict):
        self.has_inited = False
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._initialize_own_rngs()
        self._reset_settings()
        self._partial_load_scene(options, init=True)

    def _sample_disruptor_positions(self, rng, object_xyz, half_heights, counts, object_radius, disruptor_radius, object_types, disruptor_types):
        b, max_n = half_heights.shape
        device = object_xyz.device
        min_dist = 0.01
        poses = torch.zeros((b, max_n, 3), device=device)

        def is_valid(new_xy, new_r, existing_xy, existing_r):
            for p, r in zip(existing_xy, existing_r):
                if np.linalg.norm(new_xy - p) < (new_r + r + min_dist):
                    return False
            return True

        for j in range(b):
            center = object_xyz[j, :2].cpu().numpy()
            local_rng = rng[j]

            # ✅ 已有物体列表（先放目标物体）
            existing_xy = [center]

            # ⭐ 目标物体半径（必须有！）
            object_r = object_radius[j]  
            existing_r = [object_r]

            for i in range(max_n):

                if i >= counts[j]:
                    poses[j, i] = torch.tensor([999, 999, 0], device=device)
                    continue

                h = float(half_heights[j, i].item())
                r = disruptor_radius[j][i]
                # h > self.large_object_threshold 或者 object_type 和 disruptor_type 相同就放远一点，避免和目标物体混淆
                need_to_place_far = h > self.large_object_threshold or object_types[j] == disruptor_types[j][i]
                placed = False

                for _ in range(30):

                    if need_to_place_far:
                        offset = local_rng.uniform(self.object_spawn_radius, 20 * self.object_spawn_radius, size=(2,))
                        if np.linalg.norm(offset) < self.object_spawn_radius:
                            continue
                    else:
                        offset = local_rng.uniform(-self.object_spawn_radius, self.object_spawn_radius, size=(2,))

                    pos_xy = center + offset

                    # ✅ 统一防重叠（包含目标物体）
                    if is_valid(pos_xy, r, existing_xy, existing_r):
                        existing_xy.append(pos_xy)
                        existing_r.append(r)

                        poses[j, i, 0] = float(pos_xy[0])
                        poses[j, i, 1] = float(pos_xy[1])
                        poses[j, i, 2] = float(h)

                        placed = True
                        break

                if not placed:
                    poses[j, i] = torch.tensor([999, 999, 0], device=device)

        return poses

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        for obj in self._hidden_objects:
            obj.hide_visual()

        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self.capture_sensor_data()

        sensor_obs = dict()

        # ==== 取出每个 env 的 blur sigma ====
        env_ids = range(0, self.num_envs)  # ManiSkill batched env
        if self.randomize_blur:
            sigmas = torch.tensor(
                [self.per_env_settings[eid]["blur_sigma"] for eid in env_ids],
                device=self.device,
                dtype=torch.float32
            )

        for name, sensor in self.scene.sensors.items():
            if isinstance(sensor, Camera):

                if self.obs_mode in ["state", "state_dict"]:
                    obs = sensor.get_obs(
                        position=False,
                        segmentation=False,
                        apply_texture_transforms=apply_texture_transforms
                    )
                else:
                    obs = sensor.get_obs(
                        rgb=self.obs_mode_struct.visual.rgb,
                        depth=self.obs_mode_struct.visual.depth,
                        position=self.obs_mode_struct.visual.position,
                        segmentation=self.obs_mode_struct.visual.segmentation,
                        normal=self.obs_mode_struct.visual.normal,
                        albedo=self.obs_mode_struct.visual.albedo,
                        apply_texture_transforms=apply_texture_transforms
                    )

                # ====== ⭐ 在这里加 blur ======
                if self.randomize_blur and "rgb" in obs:
                    img = obs["rgb"]  # (B, H, W, 3) uint8 or float

                    # 转 tensor
                    if not torch.is_tensor(img):
                        img = torch.from_numpy(img)

                    img = img.to(self.device).float()

                    # normalize
                    if img.max() > 1.0:
                        img = img / 255.0

                    # BCHW
                    img = img.permute(0, 3, 1, 2)

                    # blur
                    img = self._gaussian_blur_batch(img, sigmas)

                    # 还原
                    img = (img.clamp(0, 1) * 255).byte()
                    img = img.permute(0, 2, 3, 1)

                    obs["rgb"] = img

                if hasattr(self.agent, "distortion_params") and "rgb" in obs:
                    assert self.agent.intrinsic_matrix is not None

                    img = obs["rgb"]  # (B, H, W, 3)

                    # 转 numpy（如果需要）
                    if torch.is_tensor(img):
                        img = img.cpu().numpy()

                    K = self.agent.intrinsic_matrix
                    dist = self.agent.distortion_params
                    b, h, w, c = img.shape

                    # === 1. 生成 map（只和分辨率有关，其实可以提前缓存）===
                    map1, map2 = cv2.initUndistortRectifyMap(
                        K, dist, None, K, (w, h), cv2.CV_32FC1
                    )

                    # === 2. 转成 grid ===
                    map1_t = torch.from_numpy(map1)
                    map2_t = torch.from_numpy(map2)

                    grid_x = (map1_t / (w - 1)) * 2 - 1
                    grid_y = (map2_t / (h - 1)) * 2 - 1

                    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
                    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1).to(self.device)  # (B, H, W, 2)

                    # === 3. 图像转 tensor + 正确格式 ===
                    img = torch.from_numpy(img).to(self.device)

                    # (B, H, W, C) → (B, C, H, W)
                    img = img.permute(0, 3, 1, 2)

                    # uint8 → float
                    if img.dtype == torch.uint8:
                        img = img.float() / 255.0
                    else:
                        img = img.float()

                    # === 4. 畸变 ===
                    img = F.grid_sample(img, grid, align_corners=True)

                    # === 5. 转回原格式 ===
                    img = (img * 255).clamp(0, 255).byte()
                    img = img.permute(0, 2, 3, 1)  # (B, H, W, C)

                    obs["rgb"] = img

                sensor_obs[name] = obs

        if self.backend.render_device.is_cuda():
            torch.cuda.synchronize()

        return sensor_obs

    def _gaussian_blur_batch(self, images, sigmas):
        """
        images: (B, C, H, W) float [0,1]
        sigmas: (B,)
        """
        B, C, H, W = images.shape
        out = []

        for i in range(B):
            sigma = sigmas[i].item()

            if sigma < 1e-3:
                out.append(images[i:i+1])
                continue

            ksize = int(2 * round(3 * sigma) + 1)

            x = torch.arange(ksize, device=images.device) - ksize // 2
            gauss = torch.exp(-(x**2) / (2 * sigma**2))
            gauss = gauss / gauss.sum()

            kernel2d = gauss[:, None] * gauss[None, :]
            kernel2d = kernel2d.expand(C, 1, ksize, ksize)

            img = images[i:i+1]
            img = F.pad(img, (ksize//2,)*4, mode="reflect")

            blurred = F.conv2d(img, kernel2d, groups=C)
            out.append(blurred)

        return torch.cat(out, dim=0)

    def _before_control_step(self):
        idxs = torch.where(self._elapsed_steps == 20)[0]
        for idx in idxs:
            # self.object._bodies[idx].set_kinematic(False)
            self.object._bodies[idx].wake_up()

    # _initialize_episode外部有reset_mask防止修改无需重置的 env
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
        # Table reset
        self.table_scene.initialize(env_idx)
        # base_qs = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([0, 0, 0]), convention="XYZ"))
        # base_p = self.table_scene.table.pose.p[0]
        # p = base_p - torch.tensor([0, 0, 0.05]).to(self.device)  # table 前移 10cm，给物体留点空间
        # q = self.table_scene.table.pose.q[0]
        # self.table_scene.table.set_pose(Pose.create_from_pq(p, q))
        self._initialize_own_rngs()
        rng = self.rngs[env_idx.tolist()] # numpy

        half_height = torch.tensor(
            [self.per_env_settings[i]["half_height"] for i in env_idx.tolist()],
            device=self.device,
        )
        object_types = [self.per_env_settings[i]["object_type"] for i in env_idx.tolist()]
        disruptor_types = [self.per_env_settings[i]["disruptor_type"] for i in env_idx.tolist()]
        # ---- random blur sigma ----
        if self.randomize_blur:
            blur_sigma = []

            for j in range(b):
                local_rng = rng[0] if self.global_random else rng[j]

                sigma = local_rng.uniform(0.0, 0.3)  # 👈 控制模糊强度
                blur_sigma.append(sigma)

            blur_sigma = np.array(blur_sigma)
            # blur_sigma[0] = 0.6
            # print(blur_sigma)

            for j, env_id in enumerate(env_idx.tolist()):
                self.per_env_settings[env_id]["blur_sigma"] = float(blur_sigma[j])

        # ---- random object pose ----
        xyz = torch.zeros((b, 3), device=self.device)

        xyz[:, 1] = torch.from_numpy(rng.uniform(-self.object_spawn_radius, self.object_spawn_radius, size=(1,))).to(self.device).squeeze(-1)

        # xyz[:, 0] += self.object_spawn_center[0]
        # xyz[:, 1] += self.object_spawn_center[1]
        # xyz[:, 2] += self.object_spawn_center[2] + 0.01

        # local_axis = torch.tensor(AXIS, device=self.device, dtype=torch.float32)
        # local_axis = local_axis / torch.norm(local_axis)
        # theta = -np.pi / 2
        # rot_q = axis_angle_to_quaternion(local_axis * theta)
        # current_q = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([0, 0, 0]), convention="XYZ"))
        # base_qs = quaternion_raw_multiply(current_q, rot_q)

        # base_qs = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([0, 0, -np.pi/2]), convention="XYZ"))
        # qs = random_quaternions(
        #     rng, lock_z=True, lock_y=True, device=self.device, bounds=(-np.pi / 18, np.pi / 18)
        # )
        
        # for mouse
        # xyz[:, 0] += self.object_spawn_center[0] + 0.03
        # xyz[:, 1] += self.object_spawn_center[1] + 0.01
        # xyz[:, 2] += self.object_spawn_center[2] + 0.02
        # base_qs = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([0, -np.pi/2, -np.pi/2]), convention="XYZ"))
        # qs = random_quaternions(
        #     rng, lock_z=True, lock_y=True, device=self.device, bounds=(-np.pi / 18, np.pi / 18)
        # )

        # for bottle
        xyz[:, 0] += self.object_spawn_center[0] + 0.02
        xyz[:, 1] += self.object_spawn_center[1] + 0.01
        xyz[:, 2] += self.object_spawn_center[2] + 0.01
        # base_qs = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([0, -np.pi/2, -np.pi/2]), convention="XYZ"))
        base_qs = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([0, 0, 0]), convention="XYZ"))

        qs = random_quaternions(
            rng, lock_z=True, lock_y=True, device=self.device, bounds=(np.pi / 37, np.pi / 36)
        )
        
        qs = quaternion_raw_multiply(qs, base_qs)

        self.object.set_pose(Pose.create_from_pq(xyz, qs))
        for env_id in env_idx.tolist():
            self.object._bodies[env_id].put_to_sleep()

        device = self.device

        # ===== 从配置读取 =====
        counts = torch.tensor(
            [self.per_env_settings[i]["disruptor_count"] for i in env_idx.tolist()],
            device=device
        )

        half_heights = torch.stack([
            torch.tensor(self.per_env_settings[i]["disruptor_half_heights"], dtype=torch.float32)
            for i in env_idx.tolist()
        ], dim=0).to(device)

        # ===== 目标物体位置 =====
        object_xyz = xyz  # (b,3)

        # ===== 调用你写好的防重叠采样 =====
        # TODO：修改为并行环境版本（现在是串行的，后续改成并行的）
        if self.randomize_disruptors:
            object_r = [self.object_r[i] for i in env_idx.tolist()]
            disruptor_r = [self._disruptors_radius[j] for j in range(b)]
            poses = self._sample_disruptor_positions(
                rng=rng,
                object_xyz=object_xyz,
                half_heights=half_heights,
                counts=counts,
                object_radius=object_r,
                disruptor_radius=disruptor_r,
                object_types=object_types,
                disruptor_types=disruptor_types,
            )

            # ===== 写回（可选，用于debug/记录）=====
            for j, env_id in enumerate(env_idx.tolist()):
                self.per_env_settings[env_id]["disruptor_poses"] = poses[j]

            # ===== ⭐ 关键：设置 actor pose =====
            # TODO：修改为并行环境版本（现在是串行的，后续改成并行的）
            for j, env_id in enumerate(env_idx.tolist()):
                env_disruptors = self._disruptors[j]
                now_rng = BatchedRNG([rng[j]])
                for i, actor in enumerate(env_disruptors):

                    if actor is None:
                        continue

                    pos = poses[j, i].unsqueeze(0)

                    # skip invalid
                    if pos[0, 0] > 900:
                        actor.set_pose(sapien.Pose(p=[999, 999, 0]))
                        continue
                    
                    qs = random_quaternions(
                        now_rng, lock_x=True, lock_y=True, device=self.device
                    )

                    pose = Pose.create_from_pq(pos, qs)
                    pose = sapien.Pose(pose.p.squeeze(0).cpu().numpy(), pose.q.squeeze(0).cpu().numpy()) 

                    actor.set_pose(pose)

        # ---- camera randomization ----
        if not self.has_inited:
            pose = sapien_utils.look_at(eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos)
            self.cam_mount.set_pose(pose)
            # self.cam_mount_human.set_pose(pose)
            self.has_inited = True

        # if b < self.num_envs:
        #     print('a')

        pose = self._get_random_camera_pose(
            self.sensor_cam_eye_pos,
            self.sensor_cam_target_pos,
            env_idx=env_idx,
        )
        # print(xyz[0], goal_xyz[0], pose.p[0])
        # Ensure batched pose
        assert pose.raw_pose.shape[0] == b, f"Camera pose batch mismatch: {pose.raw_pose.shape}, {b}"

        self.cam_mount.set_pose(pose)
        # self.agent_root_pos = self.agent.base_link.pose.p
        # self.cam_mount_human.set_pose(pose)

        qvel = np.zeros_like(self.agent.keyframes['start'].qpos)

        self.agent.robot.set_qpos(self.agent.keyframes['start'].qpos)
        self.agent.robot.set_qvel(qvel)
        

    def _get_obs_extra(self, info: dict):
        return {
            "ee_link_pose": self.agent.ee_link.pose.raw_pose,
            "ee_link_1_pose": self.agent.ee_link_1.pose.raw_pose,
            "ee_link_2_pose": self.agent.ee_link_2.pose.raw_pose,
            "ee_link_3_pose": self.agent.ee_link_3.pose.raw_pose,
        } 

    def evaluate(self):
        def point_to_axis_distance(point, origin, axis, eps=1e-8):
            """
            point:  (..., 3)
            origin: (..., 3)   轴上一点
            axis:   (..., 3)   轴方向向量

            return:
                distance: (...)
            """

            v = point - origin.unsqueeze(1)

            cross = torch.cross(v, axis.unsqueeze(1), dim=-1)

            dist = torch.linalg.norm(cross, dim=-1) / (
                torch.linalg.norm(axis.unsqueeze(1), dim=-1) + eps
            )

            return dist
        
        is_grasping = self.agent.is_grasping(self.object)
        # print(is_grasping)
        is_static = (self.object.get_linear_velocity().norm(dim=-1) < 0.2) & (self.object.get_angular_velocity().norm(dim=-1) < 0.2)
        up_time = self._elapsed_steps > 20
        fail_1_1 = torch.linalg.norm(self.scene.get_pairwise_contact_forces(self.object, self.table_scene.table), dim=1) > 0.1
        # axis = quaternion_apply(self.object.pose.q, torch.tensor([1, 0, 0], device=self.device)) # axis 请根据物体变化
        axis = quaternion_apply(self.object.pose.q, torch.tensor(AXIS, device=self.device))
        g = torch.tensor([0, 0, -1], device=self.device).unsqueeze(0)
        fail_1_2 = (torch.sum(axis * g, dim=-1).abs() < 0.02) | (torch.sum(axis * g, dim=-1).abs() > 0.98)
        is_touching = self.agent.is_touching(self.object)
        fail_1 = fail_1_1 & fail_1_2 & (~is_touching) & is_static
        fail_2 = torch.linalg.norm(self.object.pose.p - self.o_pose[:, :3], dim=-1) > 0.3
        fail_3 = self.object.pose.p[:, -1] < 0
        fail = fail_1 | fail_2 | fail_3
        ee_link_poses = torch.stack([
            self.agent.ee_link.pose.raw_pose,
            self.agent.ee_link_1.pose.raw_pose,
            self.agent.ee_link_2.pose.raw_pose,
            self.agent.ee_link_3.pose.raw_pose,
        ], dim=1)
        obj_ee_link_vec = ee_link_poses[..., :3] - self.object.pose.p[:, None, :]
        obj_to_ee_link_dist = torch.mean(torch.linalg.norm(obj_ee_link_vec, dim=-1), dim=-1)
        # now_prop = self.agent.get_proprioception()
        # error_qpos = torch.zeros_like(fail)
        # for k, v in now_prop.items():
        #     # v: (B, D) or (B,)
            
        #     # 判断异常（逐元素）
        #     judge = torch.abs(v) > 1.8
            
        #     # 如果是多维关节，需要压到 batch 维
        #     if judge.dim() > 1:
        #         judge = judge.any(dim=1)  # (B,)
            
        #     # 汇总
        #     error_qpos = error_qpos | judge

        # fail = fail | error_qpos

        # if fail.any(dim=0):
        #     print('a')
        center = self.object.pose.p
        ee_link_to_axis_dist = point_to_axis_distance(ee_link_poses[..., :3], center, axis)
        is_closed = ee_link_to_axis_dist.mean(dim=-1) < 0.05
        # print("is_static:", is_static)
        # print("is_grasping:", is_grasping)

        # g = torch.tensor([0, 0, -1], device=self.device).unsqueeze(0)
        # print(self.agent._touch_other_fingers())
        # print(torch.mean(0.5 - 0.5 * torch.tanh(40 * ee_link_to_axis_dist - 3.5), dim=-1))
        # print(ee_link_to_axis_dist)
        # if fail.any():
        #     print(f"fail:", torch.where(fail)[0])
        #     print(f"fail_1_1:", torch.where(fail_1_1)[0])
        #     print(f"fail_3:", torch.where(fail_3)[0])
        #     print(f"fail_2:", torch.where(fail_2)[0])
        return {
            "is_grasped": is_grasping,
            # "success": is_static & is_grasping & is_closed & up_time,
            "success": is_static & is_grasping & up_time,
            "fail": fail,
            "obj_to_ee_link_dist": obj_to_ee_link_dist,
            "ee_link_to_axis_dist": ee_link_to_axis_dist,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # keep_height = 0.07
        # delta_height = self.object.pose.p[:, -1] - keep_height
        # reward = 1 - torch.tanh(50 * delta_height.abs())

        reward = torch.mean(0.5 - 0.5 * torch.tanh(40 * info['ee_link_to_axis_dist'] - 3.5), dim=-1)

        axis = quaternion_apply(self.object.pose.q, torch.tensor(AXIS, device=self.device))
        g = torch.tensor([0, 0, -1], device=self.device).unsqueeze(0)
        horizontal_reward = 0.5 - 0.5 * torch.tanh(15 * torch.sum(axis * g, dim=-1).abs() - 8)
        
        reward += horizontal_reward
        is_grasping, nums = self.agent.is_grasping(self.object, ret_num=True)
        reward += torch.tanh(nums / 4)
        linear_static_reward = 1 - torch.tanh(5 * torch.linalg.norm(self.object.get_linear_velocity(), dim=-1))
        angular_static_reward = 1 - torch.tanh(5 * torch.linalg.norm(self.object.get_angular_velocity(), dim=-1))
        # print(torch.sum(axis * g, dim=-1))
        reward += 0.5 * (linear_static_reward + angular_static_reward) * is_grasping
        qvel = self.agent.robot.get_qvel().norm(dim=-1)
        robot_static_reward = 1 - torch.tanh(5 * qvel)
        reward += robot_static_reward * is_grasping
        reward[info["success"]] = 6.
        reward[info['fail']] -= 2
        reward -= self.agent._touch_other_fingers() * 1.
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.

    @property
    def _default_sim_config(self):
        return SimConfig(
            scene_config=SceneConfig(      
                solver_position_iterations=40, 
                solver_velocity_iterations=20,
                contact_offset=0.002,
                enable_ccd=True
            ),
            sim_freq=400,
            # gpu_memory_config=GPUMemoryConfig(
            #     max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
            #     max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
            #     found_lost_pairs_capacity=2**26,
            # )
        )

GraspObjectRandomEnv.__doc__ = GRASP_OBJECT_DOC_STRING.format(robot_id="panda")

# @register_env("PickObjectRandomDofbot-v1", max_episode_steps=100)
# class PickObjectEnvDofbot(PickObjectEnv):
#     def __init__(self, *args, robot_init_qpos_noise=0.02, object_type=None, object_size_info={}, object_mass=None, object_color=None, randomize_camera=None, ambient_light_temperature=None, ambient_light_intensity=None, directional_light_temperature=None, directional_light_direction=None, shadow_scale=None, reconfiguration_freq=None, global_random=False, restart_id=0, **kwargs):    
#         super().__init__(*args, robot_uids="dofbot_se", robot_init_qpos_noise=robot_init_qpos_noise, object_type=object_type, object_size_info=object_size_info, object_mass=object_mass, object_color=object_color, randomize_camera=randomize_camera, ambient_light_temperature=ambient_light_temperature, ambient_light_intensity=ambient_light_intensity, directional_light_temperature=directional_light_temperature, directional_light_direction=directional_light_direction, shadow_scale=shadow_scale, reconfiguration_freq=reconfiguration_freq, global_random=global_random, restart_id=restart_id, **kwargs)
#         self.prev_tcp_to_obj_dist = None
#         self.prev_obj_to_goal_dist = None

#     @property
#     def _default_sim_config(self):
#         return SimConfig(
#             scene_config=SceneConfig(      
#                 solver_position_iterations=40, 
#                 solver_velocity_iterations=20,
#                 contact_offset=0.002,
#             ),
#             sim_freq=200,
#             # gpu_memory_config=GPUMemoryConfig(
#             #     max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
#             #     max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
#             #     found_lost_pairs_capacity=2**26,
#             # )
#         )

#     def evaluate(self):
#         is_obj_placed = (torch.linalg.norm(self.goal_site.pose.p - self.object.pose.p, axis=1) <= self.goal_thresh)
#         is_grasped = self.agent.is_grasping(self.object)
#         is_robot_static = self.agent.is_static(1)
#         is_falled_off = self.object.pose.p[:, -1] < 0
#         dist = torch.linalg.norm((self.agent_root_pos - self.object.pose.p)[:, :2], axis=1)
#         is_in_dead_zone = (dist <= 0.05) | (dist >= 0.35)
#         # qvel = self.agent.robot.get_qvel()[..., :-6]
#         # print(torch.linalg.norm(qvel, axis=1))
#         return {
#             "success": is_obj_placed & is_robot_static,
#             "fail": is_falled_off | is_in_dead_zone,
#             "is_obj_placed": is_obj_placed,
#             "is_robot_static": is_robot_static,
#             "is_grasped": is_grasped,
#         }

#     def _initialize_episode(self, env_idx: torch.Tensor, options):
#         super()._initialize_episode(env_idx, options)
#         arm_qpos = _quantize_qpos(self.agent.keyframes['start'].qpos[:5], only_arm=True)
#         gripper_qpos = _quantize_qpos(self.agent.keyframes['start'].qpos[5:], only_gripper=True)
#         self.agent.robot.set_qpos(np.concatenate([arm_qpos, gripper_qpos], axis=-1))

#         self.agent.robot.set_qvel(np.array([0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32))
#         self.prev_obj_to_goal_dist = None
#         self.prev_tcp_to_obj_dist = None

#     # @property
#     # def _default_human_render_camera_configs(self):
#     #     config = self.agent._sensor_configs[0]
#     #     config.width = 640
#     #     config.height = 480
#     #     config.pose = sapien.Pose(p=[0, 0, 0.008], q=[0.7071068, 0, -0.7071068, 0])
#     #     config.fov = np.pi / 180 * 20
#     #     return config
        
if __name__ == "__main__":
    print(sapien.Pose(p=[-0.615, 0, 0.005], q=matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([0, 0, -np.pi/2]), convention="XYZ").tolist())))