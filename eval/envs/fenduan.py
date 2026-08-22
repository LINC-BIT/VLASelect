import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import trimesh
import transforms3d as t3d

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from mani_skill.utils.building.ground import build_ground
from .agents.dofbot_se import DOFBOT_SE
from pathlib import Path

@register_env("ShipRustRemoval-v1", max_episode_steps=1000)
class ShipRustRemovalEnv(BaseEnv):
    HULL_THICKNESS = 0.02
    HULL_HEIGHT = 0.6
    HULL_WIDTH = 1.0

    REMOVE_THRESHOLD = 0.04

    SUPPORTED_REWARD_MODES = ["dense","none"]

    SUPPORTED_ROBOTS = ["dofbot_se"]
    agent:DOFBOT_SE

    enable_particle = False
    job_type = 'penqi'

    def __init__(self, *args, robot_uids="dofbot_se", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------- SIM CONFIG ----------------
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=100,
            control_freq=20,
            scene_config=SceneConfig(
                contact_offset=0.01,
                solver_position_iterations=4,
                solver_velocity_iterations=0,
            ),
        )

    # ---------------- CAMERA ----------------

    @property
    def _default_sensor_configs(self):

        pose = sapien_utils.look_at(
            eye=[0.3, 0, 0.8],
            target=[0, 0, 0.3],
        )

        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=320,
                height=240,
                fov=1.2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):

        pose = sapien_utils.look_at(
            eye=[-0.6, -0.5, 0.3],
            target=[-0.6, 0, 0.3],
        )

        return CameraConfig(
            "render_camera",
            pose=pose,
            width=1280,
            height=720,
            fov=1.2,
            near=0.01,
            far=100,
        )

    # ---------------- ROBOT ----------------

    def _load_agent(self, options: dict):

        super()._load_agent(
            options,
            sapien.Pose(p=[-0.6, 0, 0])
        )

        # self.agent.set_action(torch.Tensor([[0,0,0,0,0,-1]]).to(self.device))

    # ---------------- SCENE ----------------

    def _load_lighting(self, options: dict):
        shadow = self.enable_shadow
        if self.job_type == 'penqi':
            self.scene.set_ambient_light([0.5, 0.5, 0.5])
        else:
            self.scene.set_ambient_light([0.4, 0.4, 0.4])
        self.scene.add_directional_light(
            [2.5, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_scene(self, options: dict):
        # for s in self.scene.sub_scenes:
        #     s.render_system.cubemap = sapien.render.RenderCubemap(str(Path(__file__).parent.resolve() / f"assets/texture/empty_warehouse_01_4k.hdr"))
        self.scene.set_env
        f = lambda x: [i / 256. for i in x]
        if self.job_type == 'penqi':
            color1 = f([165, 16, 21, 256])
            color2 = f([21, 136, 53, 256]) # 60 137 93
            color3 = f([16, 42, 61, 256]) # 52 82 109
            color = color3

            self.particle_system = ParticleSystem(self.scene, max_particles=400, radius=0.002, color=color)
            sp_mat = sapien.render.RenderMaterial()
            
            sp_mat.base_color = color
            sp_mat.specular = 0.5
            # sp_mat.metallic = 0.0
            # sp_mat.roughness = 0.5
        elif self.job_type == 'mosha':
            self.particle_system = ParticleSystem(self.scene, max_particles=200, radius=0.003, color=[0.85, 0.80, 0.60, 1], dust=True)
            sp_mat = sapien.render.RenderMaterial()
            sp_mat.normal_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/steel/Metal012_1K-JPG_NormalGL.jpg"))
            sp_mat.metallic_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/mosha_steel/Metal011_1K-JPG_Metalness.jpg"))
            sp_mat.base_color = [0.7, 0.7, 0.7, 1]
            sp_mat.base_color_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/steel/Metal012_1K-JPG_Color.jpg"))
            sp_mat.metallic = 1.0
            sp_mat.roughness = 0.18
        else:
            self.particle_system = None
            sp_mat = sapien.render.RenderMaterial()
            # sp_mat.normal_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/under_rust/Metal061C_1K-JPG_NormalGL.jpg"))
            # sp_mat.metallic_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/under_rust/Metal061C_1K-JPG_Metalness.jpg"))
            sp_mat.base_color = f([6, 7, 10, 255])
            # sp_mat.base_color_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/under_rust/Metal061C_1K-JPG_Color.jpg"))
            sp_mat.metallic = 1.0
            sp_mat.roughness = 0.4
            # sp_mat = sapien.render.RenderMaterial()
            # sp_mat.base_color = [0.0, 0.0, 1.0, 0.8]
            # sp_mat.metallic = 0.0
            # sp_mat.roughness = 0.2
            # sp_mat.specular = 0.6

        # steel plate
        builder_rust = self.scene.create_actor_builder()

        # 平台大小 (半尺寸)
        half_size = [2.0, 2.0, 0.001]

        # 添加碰撞
        builder_rust.add_box_collision(
            half_size=half_size
        )

        # 添加可视化
        # 喷漆结果
        # steel = sapien.render.RenderMaterial()
        # steel.base_color = [0, 0, 0.5, 1]
        # steel.normal_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/steel/Metal012_1K-JPG_NormalGL.jpg"))
        # steel.metallic_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/mosha_steel/Metal011_1K-JPG_Metalness.jpg"))
        # steel.metallic = 1.0
        # steel.roughness = 0.18

        if self.job_type == 'penqi':
            steel = sapien.render.RenderMaterial()
            steel.base_color_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/steel/Metal012_1K-JPG_Color.jpg"))
            steel.normal_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/steel/Metal012_1K-JPG_NormalGL.jpg"))
            steel.metallic_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/mosha_steel/Metal011_1K-JPG_Metalness.jpg"))
            steel.metallic = 1.0
            steel.roughness = 0.18
            builder_rust.add_box_visual(
                half_size=half_size,
                material=steel,
            )
        elif self.job_type == 'mosha':
            builder_rust.add_visual_from_file(
                filename=str(Path(__file__).parent.resolve() / f"assets/steel/steel4.glb"),
                scale=(2, 2, 2)
            )
            # steel = sapien.render.RenderMaterial()
            # steel.base_color_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/steel/Metal012_1K-JPG_Color.jpg"))
            # steel.metallic_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/steel/Metal012_1K-JPG_Metalness.jpg"))
            # steel.metallic = 1.0
            # steel.roughness = 0.191
            # builder_rust.add_box_visual(
            #     half_size=half_size,
            #     material=steel,
            # )
        else:
            steel = sapien.render.RenderMaterial()
            steel.base_color_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/rust.jpg"))
            steel.metallic = 0.6
            steel.roughness = 0.9
            builder_rust.add_visual_from_file(
                filename=str(Path(__file__).parent.resolve() / f"assets/rust/rust4.glb"),
                scale=(1, 1, 1)
            )

        # 磨砂钢
        # steel = sapien.render.RenderMaterial()
        # steel.base_color_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/steel/Metal012_1K-JPG_Color.jpg"))
        # steel.normal_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/steel/Metal012_1K-JPG_NormalGL.jpg"))
        # steel.metallic_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/mosha_steel/Metal011_1K-JPG_Metalness.jpg"))
        # steel.metallic = 1.0
        # steel.roughness = 0.18

        # 创建钢材质
        # steel = sapien.render.RenderMaterial()
        # steel.base_color_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/steel/Metal012_1K-JPG_Color.jpg"))
        # steel.metallic_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/steel/Metal012_1K-JPG_Metalness.jpg"))
        # steel.metallic = 1.0
        # steel.roughness = 0.191

        # 平台位置
        if self.job_type == 'penqi':
            plane_y = 0.33    
            builder_rust.initial_pose = sapien.Pose(p=[-0.6, plane_y, 0.242], q=[0.7071, 0.7071, 0, 0])
            self.spray_system = SpraySystem(self, plane_y, mat=sp_mat)
        elif self.job_type == 'mosha':
            plane_y = 0.4    
            builder_rust.initial_pose = sapien.Pose(p=[-0.6, plane_y, 0.242], q=[0.7071, 0.7071, 0, 0])
            self.spray_system = SpraySystem(self, plane_y, mat=sp_mat)
        else:
            plane_y = 0.21
            builder_rust.initial_pose = sapien.Pose(p=[-0.6, plane_y, 0.242], q=[0.7071, 0.7071, 0, 0])
            self.spray_system = SpraySystem(self, plane_y, mat=sp_mat, thres=0.06)

        self.steel = builder_rust.build_static(name="steel")

        # mosha
        if self.job_type == 'mosha':
            builder = self.scene.create_actor_builder()

            builder.add_visual_from_file(
                filename=str(Path(__file__).parent.resolve() / f"assets/sandblaster/3d66.com_JJI54559488412.obj"),
                scale=(2e-4, 2e-4, 2e-4)
            )

            ee_link = sapien_utils.get_obj_by_name(
                self.agent.robot.get_links(), self.agent.ee_link_name
            )

            builder.initial_pose = sapien.Pose()

            self.sander = builder.build_dynamic(name="sander")

            self.sander.set_disable_gravity(True)

            print(self.sander.mass)

            drive = self.scene.create_drive(
                ee_link,
                sapien.Pose(p=[-0.02, 0.0, 0.02], q = [0.5, -0.5, 0.5, 0.5]),  # 工具在EE前方
                self.sander,
                sapien.Pose()
            )

            drive.set_drive_property_x(stiffness=1e5, damping=1e3)
            drive.set_drive_property_y(stiffness=1e5, damping=1e3)
            drive.set_drive_property_z(stiffness=1e5, damping=1e3)
            [
                x.set_drive_property_swing(1e5, 1e3, 3.4028234663852886e38, "force")
                for x in drive._objs
            ]
            [
                x.set_drive_property_twist(1e5, 1e3, 3.4028234663852886e38, "force")
                for x in drive._objs
            ]

        # sprayer
        elif self.job_type == 'penqi':
            builder = self.scene.create_actor_builder()

            builder.add_visual_from_file(
                filename=str(Path(__file__).parent.resolve() / f"assets/sprayer/sprayer.obj"),
                scale=(6e-4, 6e-4, 6e-4)
            )

            ee_link = sapien_utils.get_obj_by_name(
                self.agent.robot.get_links(), self.agent.ee_link_name
            )

            builder.initial_pose = sapien.Pose()

            self.sander = builder.build_dynamic(name="sander")

            self.sander.set_disable_gravity(True)

            drive = self.scene.create_drive(
                ee_link,
                sapien.Pose(p=[-0.02, -0.05, 0], q = [0.5, -0.5, 0.5, 0.5]),  # 工具在EE前方
                self.sander,
                sapien.Pose()
            )

            drive.set_drive_property_x(stiffness=1e5, damping=1e3)
            drive.set_drive_property_y(stiffness=1e5, damping=1e3)
            drive.set_drive_property_z(stiffness=1e5, damping=1e3)
            [
                x.set_drive_property_swing(1e5, 1e3, 3.4028234663852886e38, "force")
                for x in drive._objs
            ]
            [
                x.set_drive_property_twist(1e5, 1e3, 3.4028234663852886e38, "force")
                for x in drive._objs
            ]

        else:
            builder = self.scene.create_actor_builder()

            builder.add_visual_from_file(
                filename=str(Path(__file__).parent.resolve() / f"assets/sander/3d66.com_JCI54559042907.obj"),
                scale=(4e-4, 4e-4, 4e-4)
            )

            ee_link = sapien_utils.get_obj_by_name(
                self.agent.robot.get_links(), self.agent.ee_link_name
            )

            builder.initial_pose = sapien.Pose()

            self.sander = builder.build_dynamic(name="sander")

            self.sander.set_disable_gravity(True)

            print(self.sander.mass)
            self.sander.set_mass(1e-20)

            drive = self.scene.create_drive(
                ee_link,
                sapien.Pose(p=[-0.05, 0, 0], q = [0.5, -0.5, 0.5, 0.5]),  # 工具在EE前方
                self.sander,
                sapien.Pose()
            )

            drive.set_drive_property_x(stiffness=1e5, damping=1e3)
            drive.set_drive_property_y(stiffness=1e5, damping=1e3)
            drive.set_drive_property_z(stiffness=1e5, damping=1e3)
            [
                x.set_drive_property_swing(1e5, 1e3, 3.4028234663852886e38, "force")
                for x in drive._objs
            ]
            [
                x.set_drive_property_twist(1e5, 1e3, 3.4028234663852886e38, "force")
                for x in drive._objs
            ]

        builder = self.scene.create_actor_builder()
        
        # 平台大小 (半尺寸)
        half_size = [0.2, 0.2, 0.01]

        # 添加碰撞
        builder.add_box_collision(
            half_size=half_size
        )

        # 添加可视化
        # 创建钢材质
        plat = sapien.render.RenderMaterial()
        # plat.base_color_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/platform/DiamondPlate001_1K-JPG_Color.jpg"))
        plat.base_color = [0.3, 0.3, 0.3, 1]
        # plat.normal_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/platform/DiamondPlate001_1K-JPG_NormalGL.jpg"))
        # plat.roughness_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/platform/DiamondPlate001_1K-JPG_Roughness.jpg"))
        plat.metallic_texture = sapien.render.RenderTexture2D(str(Path(__file__).parent.resolve() / f"assets/texture/platform/DiamondPlate001_1K-JPG_Metalness.jpg"))

        # 视觉
        builder.add_box_visual(
            half_size=half_size,
            material=plat
        )

        # 平台位置
        builder.initial_pose = sapien.Pose(p=[-0.6, -0.14, -0.005])

        # 创建静态物体
        self.platform = builder.build_static(name="platform")


    # ---------------- RESET ----------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):

        self.removed_rust = 0

        with torch.device(self.device):
            pass
            # self.table_scene.initialize(env_idx)

            # for rust in self.rust_dots:

            #     y = torch.rand(1).item() * self.HULL_WIDTH - self.HULL_WIDTH / 2
            #     z = torch.rand(1).item() * self.HULL_HEIGHT + 0.05
            #     x = 0.2 - self.HULL_THICKNESS / 2

            #     rust.set_pose(
            #         sapien.Pose(
            #             p=[x, y, z],
            #             q=euler2quat(0, np.pi / 2, 0),
            #         )
            #     )

    # ---------------- CONTROL STEP ----------------
    def start_particle(self):
        self.enable_particle = True


    def _after_control_step(self):

        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

        tcp_pose = self.agent.tcp.pose
        # # 如果是 torch tensor
        # tcp_pose = tcp_pose.raw_pose.detach().cpu().numpy()

        # # 保证形状正确
        # tcp_pose = np.squeeze(tcp_pose).astype(np.float32)

        if self.enable_particle and self.spray_system is not None:
            # 单环境
            p = tcp_pose.p[0]
            q = tcp_pose.q[0]

            # 四元数转旋转矩阵
            R = t3d.quaternions.quat2mat(q)

            # 喷嘴相对 TCP 的偏移
            if self.job_type == 'penqi':
                nozzle_offset_local = np.array([-0.05, 0, 0.047], dtype=np.float32)
                nozzle_offset_world = R @ nozzle_offset_local
            elif self.job_type == 'mosha':
                nozzle_offset_local = np.array([-0.035, 0, 0.13], dtype=np.float32)
                nozzle_offset_world = R @ nozzle_offset_local
            else:
                nozzle_offset_local = np.array([-0.05, 0, 0], dtype=np.float32)
                nozzle_offset_world = R @ nozzle_offset_local

            # 粒子发射位置
            p_world = p + nozzle_offset_world

            # 粒子速度方向（局部 z 轴）
            forward = R @ np.array([0, 0, 1], dtype=np.float32)

            # 发射粒子
            if self.job_type == 'penqi':
                self.particle_system.emit(p_world, forward, count=20, vel_scale=0.4)
            elif self.job_type == 'mosha':
                self.particle_system.emit(p_world, forward, count=10, vel_scale=0.4)
            # else:
            #     self.particle_system.emit(p_world, forward, count=10, vel_scale=0.4)

            # 更新粒子
            if self.particle_system is not None:
                self.particle_system.update()

            self.spray_system.spray(p_world, forward)

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    # ---------------- REWARD ----------------

    def compute_dense_reward(self, obs, action, info):

        reward = self.removed_rust * 2.0

        return reward

    # ---------------- SUCCESS ----------------

    def evaluate(self):

        remaining = torch.zeros(self.num_envs, device=self.device)

        # for rust in self.rust_dots:

        #     z = rust.pose.p[:, 2]

        #     remaining += (z > 0).float()

        success = remaining < 10

        return dict(
            success=success,
            remaining_rust=remaining,
        )

    # ---------------- OBS ----------------

    def _get_obs_extra(self, info: dict):
        builder = self.scene.create_actor_builder()
        builder.add_sphere_visual()
        return dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

# ---------------- 粒子系统类 ----------------
from mani_skill.envs.scene import ManiSkillScene
class ParticleSystem:
    def __init__(self, scene:ManiSkillScene, max_particles=100, radius=0.001, color=[1.0, 0.8, 0.1, 1.0], dust=False):
        """
        scene: SAPIEN scene
        max_particles: 粒子总数
        radius: 粒子半径
        color: RGBA 粒子颜色
        """
        self.scene:ManiSkillScene = scene
        self.particles = []
        self.max_particles = max_particles
        self.radius = radius
        self.color = color
        self.lifetimes = [0] * max_particles
        self.max_age = 20  # 帧数寿命

        # -------- 在场景中预创建粒子池 --------
        for i in range(self.max_particles):
            builder = self.scene.create_actor_builder()
            mat = sapien.render.RenderMaterial()
            mat.base_color = self.color
            mat.metallic = 0
            # -------- dust模式：随机颗粒形状 --------
            if dust:
                mat.roughness = 0.9   # 沙粒比较粗糙
                shape_type = np.random.choice(["sphere", "box"])

                # 随机尺寸（让颗粒不规则）
                r = self.radius * np.random.uniform(0.6, 1.4)

                if shape_type == "sphere":
                    builder.add_sphere_visual(radius=r, material=mat)

                else:
                    # 不规则box
                    sx = r * np.random.uniform(0.5, 1.5)
                    sy = r * np.random.uniform(0.5, 1.5)
                    sz = r * np.random.uniform(0.5, 1.5)

                    builder.add_box_visual(
                        half_size=[sx, sy, sz],
                        material=mat
                    )

            # -------- 普通粒子 --------
            else:
                builder.add_sphere_visual(radius=self.radius, material=mat)

            # 随机初始姿态（让沙粒方向不同）
            q = t3d.quaternions.axangle2quat(
                np.random.randn(3), np.random.uniform(0, np.pi)
            )
            builder.set_initial_pose(sapien.Pose(p=[0,0,-10], q=q))
            particle = builder.build_dynamic(name=f"particle_{i}")

            particle.set_disable_gravity(True)

            self.particles.append(particle)

    def emit(self, pos, forward, count=5, vel_scale=0.1):
        pos = np.squeeze(np.asarray(pos, dtype=np.float32))
        forward = np.asarray(forward, dtype=np.float32)
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        for _ in range(count):
            for i, particle in enumerate(self.particles):
                if self.lifetimes[i] <= 0:

                    particle.set_pose(sapien.Pose(p=pos))

                    # 🔥 速度沿前方 + 少量扰动
                    noise = np.random.randn(3).astype(np.float32) * 0.2
                    vel = (forward + noise) * vel_scale

                    particle.set_linear_velocity(vel)

                    self.lifetimes[i] = self.max_age
                    break

    def update(self):
        # 更新粒子寿命并隐藏到不可见位置
        for i, particle in enumerate(self.particles):
            if self.lifetimes[i] > 0:
                self.lifetimes[i] -= 1
                if self.lifetimes[i] <= 0:
                    # 超过寿命，隐藏
                    particle.set_pose(sapien.Pose(p=[0, 0, -10]))

class SpraySystem:

    MAX_DOTS = 2000
    DOT_RADIUS = 0.03
    DOT_THICKNESS = 0.005

    def __init__(self, env, plane_y, mat, thres=0.12):
        self.env = env
        self.scene:ManiSkillScene = env.scene
        self.device = env.device
        self.plane_y = plane_y
        self.mat = mat
        self.thres = thres

        self.draw_step = 0
        self.dots = []

        self._build_dot_pool()

    # -------------------------------------------------
    # 创建喷涂点池
    # -------------------------------------------------
    def _build_dot_pool(self):

        dot_builder = self.scene.create_actor_builder()

        dot_builder.add_cylinder_visual(
            radius=self.DOT_RADIUS,
            half_length=self.DOT_THICKNESS,
            material=self.mat,
        )

        for i in range(self.MAX_DOTS):

            dot = dot_builder.build_static(name=f"spray_dot_{i}")

            dot.set_pose(
                sapien.Pose(
                    p=[0, 0, -100]  # 初始隐藏
                )
            )

            self.dots.append(dot)

    def ray_plane_intersection(self, nozzle_pos, forward, plane_y):

        if abs(forward[1]) < 1e-6:
            return None

        t = (plane_y - nozzle_pos[1]) / forward[1]

        if t < 0 or t > self.thres:
            return None

        
        hit = nozzle_pos + t * forward
        print('hit!', hit)
        return hit

    # -------------------------------------------------
    # 喷涂
    # -------------------------------------------------
    def spray(self, nozzle_pos, forward):
        hit = self.ray_plane_intersection(
            nozzle_pos,
            forward,
            self.plane_y
        )

        if hit is None:
            return

        pos = hit

        pos_tensor = torch.tensor(
            pos,
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)

        dot = self.dots[self.draw_step]

        dot.set_pose(
            Pose.create_from_pq(
                pos_tensor,
                euler2quat(0, 0, np.pi / 2)
            )
        )

        self.draw_step += 1

        if self.draw_step >= self.MAX_DOTS:
            self.draw_step = 0