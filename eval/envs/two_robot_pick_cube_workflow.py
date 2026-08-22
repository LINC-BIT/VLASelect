# =========================================================
# TwoRobotPickCubeHRL-v1
# =========================================================

from typing import Any, Tuple, Union

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
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from envs.sub_task import SubTask


@register_env("TwoRobotPickCubeHRL-v1", max_episode_steps=100)
class TwoRobotPickCubeHRL(BaseEnv):

    SUPPORTED_ROBOTS = [
        ("panda_wristcam", "panda_wristcam")
    ]

    agent: MultiAgent[Tuple[Panda, Panda]]

    cube_half_size = 0.02

    goal_thresh = 0.025

    PUSH_SUBTASK = 0

    PICK_SUBTASK = 1

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        

        # =================================================
        # runtime states
        # =================================================

        self.current_subtask = torch.full(
            (kwargs['num_envs'],),
            -1,
            dtype=torch.long
        )

        self.subtask_step = torch.zeros(
            kwargs['num_envs'],
            dtype=torch.long
        )

        super().__init__(
            *args,
            robot_uids=robot_uids,
            **kwargs
        )

        self.subtasks = []

        self.current_subtask = self.current_subtask.to(self.device)
        self.subtask_step = self.subtask_step.to(self.device)
        

    # =====================================================
    # sim config
    # =====================================================
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    # =====================================================
    # camera
    # =====================================================
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            [1.0, 0, 0.75],
            [0.0, 0.0, 0.25]
        )

        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            [1.4, 0.8, 0.75],
            [0.0, 0.1, 0.1]
        )

        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    # =====================================================
    # load agents
    # =====================================================
    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            [
                sapien.Pose(p=[0, -1, 0]),
                sapien.Pose(p=[0, 1, 0])
            ]
        )

    # =====================================================
    # build subtasks
    # =====================================================
    def _build_sub_task(self):
        def push_termination():
            target_p = torch.tensor([0, 0.15, 0], device=self.device).expand_as(self.cube.pose.p)
            return torch.linalg.norm(self.cube.pose.p - target_p, axis=1) < 0.03
            

        def pick_termination():
            return torch.linalg.norm(self.cube.pose.p - self.goal_site.pose.p, axis=1) < self.goal_thresh
            

        self.subtasks = [
            SubTask(
                id=0,
                name="push_cube_to_other_side",
                agents=[self.left_agent.uid],
                cooperative=False,
                max_episode_steps=50,
                termination_fn=push_termination
            ),

            SubTask(
                id=1,
                name="pick_cube",
                agents=[self.right_agent.uid],
                cooperative=False,
                max_episode_steps=50,
                termination_fn=pick_termination
            )
        ]

    # =====================================================
    # load scene
    # =====================================================
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )

        self.table_scene.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(
                p=[0, 0, 0.02]
            ),
        )

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

        self._hidden_objects.append(self.goal_site)

    # =====================================================
    # initialize episode
    # =====================================================
    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict
    ):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.left_init_qpos = self.left_agent.robot.get_qpos()

            # -------------------------------------------------
            # cube
            # -------------------------------------------------
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            xyz[:, 1] = -0.15 - torch.rand((b,)) * 0.1 + 0.05
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # -------------------------------------------------
            # goal
            # -------------------------------------------------
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            goal_xyz[:, 1] = 0.15 + torch.rand((b,)) * 0.1 - 0.05
            goal_xyz[:, 2] = torch.rand((b,)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # -------------------------------------------------
            # reset runtime states
            # -------------------------------------------------
            self.current_subtask[env_idx] = -1
            self.subtask_step[env_idx] = 0

    # =====================================================
    # helper
    # =====================================================
    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    # =====================================================
    # set subtask
    # =====================================================
    def set_subtask(
        self,
        env_ids,
        subtask_id
    ):
        self.current_subtask[env_ids] = subtask_id
        self.subtask_step[env_ids] = 0

    # =====================================================
    # step
    # =====================================================
    def step(self, action):
        active_mask = (self.current_subtask >= 0)
        self.subtask_step[active_mask] += 1

        return super().step(action)

    # =====================================================
    # evaluate
    # =====================================================
    def evaluate(self):
        is_obj_placed = (torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1) <= self.goal_thresh)
        is_right_arm_static = self.right_agent.is_static(0.2)

        # =================================================
        # subtask done
        # =================================================
        subtask_done = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device
        )

        # -------------------------------------------------
        # push subtask
        # -------------------------------------------------
        push_mask = (self.current_subtask == self.PUSH_SUBTASK)

        if push_mask.any():
            push_done = self.subtasks[self.PUSH_SUBTASK].termination_fn()
            push_timeout = (self.subtask_step >= self.subtasks[self.PUSH_SUBTASK].max_episode_steps)
            subtask_done[push_mask] = torch.logical_or(push_done[push_mask], push_timeout[push_mask])

        # -------------------------------------------------
        # pick subtask
        # -------------------------------------------------
        pick_mask = (self.current_subtask == self.PICK_SUBTASK)

        if pick_mask.any():
            pick_done = self.subtasks[self.PICK_SUBTASK].termination_fn()
            pick_timeout = (self.subtask_step >= self.subtasks[self.PICK_SUBTASK].max_episode_steps)
            subtask_done[pick_mask] = torch.logical_or(pick_done[pick_mask], pick_timeout[pick_mask])

        return {
            "success": torch.logical_and(is_obj_placed, is_right_arm_static),
            "is_obj_placed": is_obj_placed,
            "is_right_arm_static": is_right_arm_static,
            "subtask_done": subtask_done,
        }

    # =====================================================
    # task-conditioned obs
    # =====================================================
    def _get_obs_extra(self, info: dict):
        obs = {}

        # =================================================
        # high-level obs (always exists)
        # =================================================
        obs["high_level"] = {
            "goal_pose": self.goal_site.pose.raw_pose,
            "cube_pose": self.cube.pose.raw_pose,
        }

        # =================================================
        # push subtask
        # =================================================
        push_mask = (self.current_subtask == self.PUSH_SUBTASK)
        if push_mask.any():
            obs["push_cube_to_other_side"] = {
                "left_tcp_pose": self.left_agent.tcp.pose.raw_pose[push_mask],
                "left_tcp_to_cube": self.cube.pose.p[push_mask] - self.left_agent.tcp.pose.p[push_mask],
                "cube_pose": self.cube.pose.raw_pose[push_mask],
            }

        # =================================================
        # pick subtask
        # =================================================
        pick_mask = (self.current_subtask == self.PICK_SUBTASK)
        if pick_mask.any():
            obs["pick_cube"] = {
                "right_tcp_pose": self.right_agent.tcp.pose.raw_pose[pick_mask],
                "cube_to_goal": self.goal_site.pose.p[pick_mask] - self.cube.pose.p[pick_mask],
                "goal_pose": self.goal_site.pose.raw_pose[pick_mask],
            }

        return obs

    # =====================================================
    # task-conditioned reward
    # =====================================================
    def compute_dense_reward(
        self,
        obs: Any,
        action,
        info: dict
    ):

        reward = torch.zeros(
            self.num_envs,
            device=self.device
        )

        # =================================================
        # masks
        # =================================================
        push_mask = (self.current_subtask == self.PUSH_SUBTASK)
        pick_mask = (self.current_subtask == self.PICK_SUBTASK)

        # =================================================
        # push reward
        # =================================================
        if push_mask.any():
            tcp_to_obj_dist = torch.linalg.norm(self.cube.pose.p[push_mask] - self.left_agent.tcp.pose.p[push_mask], axis=1)
            reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
            push_reward = 1 - torch.tanh(5 * torch.clamp(0.05 - self.cube.pose.p[push_mask, 1], min=0))
            reward[push_mask] = (reaching_reward + push_reward) / 2

        # =================================================
        # pick reward
        # =================================================
        if pick_mask.any():
            tcp_to_obj_dist = torch.linalg.norm(self.cube.pose.p[pick_mask] - self.right_agent.tcp.pose.p[pick_mask], axis=1)
            reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
            obj_to_goal_dist = torch.linalg.norm(self.goal_site.pose.p[pick_mask] - self.cube.pose.p[pick_mask], axis=1)
            place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
            is_grasped = self.right_agent.is_grasping(self.cube)[pick_mask]
            reward[pick_mask] = (reaching_reward + place_reward + 2 * is_grasped.float())

        # =================================================
        # success bonus
        # =================================================
        reward[info["success"]] += 10

        return reward

    # =====================================================
    # normalized reward
    # =====================================================
    def compute_normalized_dense_reward(
        self,
        obs,
        action,
        info
    ):
        return self.compute_dense_reward(obs, action, info) / 10.0