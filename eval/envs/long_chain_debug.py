from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig


@register_env("LongChainDebug-v1", max_episode_steps=200)
class LongChainDebugEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]
    SUPPORTED_REWARD_MODES = ["none", "normalized_dense"]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        chain_num_links=8,
        chain_link_length=0.08,
        chain_link_half_width=0.012,
        chain_anchor_height=0.65,
        **kwargs,
    ):
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.chain_num_links = int(chain_num_links)
        self.chain_link_length = float(chain_link_length)
        self.chain_link_half_width = float(chain_link_half_width)
        self.chain_anchor_height = float(chain_anchor_height)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.9, -0.8, 0.8], target=[0.0, 0.0, 0.25])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 3)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[1.2, -1.2, 1.0], target=[0.0, 0.0, 0.25])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=np.pi / 3
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0.45, -0.35, 0.0]))

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)

        builder = self.scene.create_articulation_builder()
        builder.set_name("long_chain")
        builder.disable_self_collisions = True
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, self.chain_anchor_height])

        root = builder.create_link_builder()
        root.set_name("anchor")
        root.set_joint_name("anchor_joint")
        root.set_joint_properties(
            "fixed",
            [],
            pose_in_parent=sapien.Pose(),
            pose_in_child=sapien.Pose(),
        )
        root.add_sphere_visual(
            radius=self.chain_link_half_width * 1.5,
            material=sapien.render.RenderMaterial(base_color=[0.9, 0.2, 0.2, 1.0]),
        )

        parent = root
        for idx in range(self.chain_num_links):
            link = builder.create_link_builder(parent=parent)
            link.set_name(f"segment_{idx}")
            link.set_joint_name(f"joint_{idx}")
            link.set_joint_properties(
                "revolute",
                [[-np.pi / 2, np.pi / 2]],
                pose_in_parent=sapien.Pose(p=[0.0, 0.0, -self.chain_link_length]),
                pose_in_child=sapien.Pose(),
                damping=0.05,
            )
            segment_pose = sapien.Pose(p=[0.0, 0.0, -self.chain_link_length / 2])
            link.add_box_collision(
                pose=segment_pose,
                half_size=[
                    self.chain_link_half_width,
                    self.chain_link_half_width,
                    self.chain_link_length / 2,
                ],
            )
            link.add_box_visual(
                pose=segment_pose,
                half_size=[
                    self.chain_link_half_width,
                    self.chain_link_half_width,
                    self.chain_link_length / 2,
                ],
                material=sapien.render.RenderMaterial(
                    base_color=[0.2, 0.5 + 0.04 * idx, 0.85 - 0.04 * idx, 1.0]
                ),
            )
            parent = link

        self.chain = builder.build(fix_root_link=True)
        self.chain_end_link = self.chain.links_map[f"segment_{self.chain_num_links - 1}"]
        self.chain_mid_link = self.chain.links_map[f"segment_{self.chain_num_links // 2}"]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        all_qpos = self.chain.get_qpos().clone()
        all_qvel = self.chain.get_qvel().clone()
        b = len(env_idx)
        qpos = torch.zeros((b, self.chain_num_links), device=self.device)
        if self.chain_num_links > 0:
            qpos[:, 0] = 0.15
            qpos += 0.03 * torch.randn_like(qpos)
        qvel = torch.zeros_like(qpos)
        all_qpos[env_idx] = qpos
        all_qvel[env_idx] = qvel
        self.chain.set_qpos(all_qpos)
        self.chain.set_qvel(all_qvel)

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }

    def _get_obs_extra(self, info: dict):
        return dict(
            chain_qpos=self.chain.get_qpos(),
            chain_end_pos=self.chain_end_link.pose.p,
            chain_mid_pos=self.chain_mid_link.pose.p,
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
