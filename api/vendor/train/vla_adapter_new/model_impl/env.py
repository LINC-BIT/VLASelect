import math

import torch
import torch.nn.functional as F

from mani_skill.envs.tasks.dexterity.rotate_single_object_in_hand import (
    RotateSingleObjectInHandLevel0,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("EasierRotateSingleObjectInHandLevel0-v1", max_episode_steps=300)
class EasierRotateSingleObjectInHandLevel0(RotateSingleObjectInHandLevel0):
    """A lighter version of RotateSingleObjectInHandLevel0-v1.

    Relative to ManiSkill's original Level0 task, this environment only lowers the
    success threshold from 720 degrees (4 * pi) to 60 degrees (pi / 3). All robot,
    object, observation, reward, and physics settings are inherited unchanged.
    """

    success_threshold_degrees: float = 180.0
    success_tip_distance_threshold: float = 0.12
    success_min_close_tips: int = 2

    def _initialize_actors(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            obj_heights = self.obj_heights
            if obj_heights.numel() > 1:
                obj_heights = obj_heights[env_idx]

            new_pos = torch.randn((b, 3), device=self.device) * self.obj_init_pos_noise
            new_pos[:, 2] = torch.abs(new_pos[:, 2]) + self.hand_init_height + obj_heights
            obj_pose = Pose.create_from_pq(
                p=new_pos,
                q=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(b, -1),
            )
            self.obj.set_pose(obj_pose)

            if self.difficulty_level <= 2:
                axis = torch.ones((b,), dtype=torch.long, device=self.device) * 2
            else:
                axis = torch.randint(0, 3, (b,), dtype=torch.long, device=self.device)
            rot_dir = F.one_hot(axis, num_classes=3).to(torch.float32)
            vector_axis = (axis + 1) % 3
            unit_vector = F.one_hot(vector_axis, num_classes=3).to(torch.float32)

            if not hasattr(self, "rot_dir") or self.rot_dir.shape != (self.num_envs, 3):
                self.rot_dir = torch.zeros((self.num_envs, 3), dtype=rot_dir.dtype, device=self.device)
            if not hasattr(self, "unit_vector") or self.unit_vector.shape != (self.num_envs, 3):
                self.unit_vector = torch.zeros((self.num_envs, 3), dtype=unit_vector.dtype, device=self.device)
            if not hasattr(self, "prev_unit_vector") or self.prev_unit_vector.shape != (self.num_envs, 3):
                self.prev_unit_vector = torch.zeros((self.num_envs, 3), dtype=unit_vector.dtype, device=self.device)
            if not hasattr(self, "cum_rotation_angle") or self.cum_rotation_angle.shape != (self.num_envs,):
                self.cum_rotation_angle = torch.zeros((self.num_envs,), device=self.device)

            self.rot_dir[env_idx] = rot_dir
            self.unit_vector[env_idx] = unit_vector
            self.prev_unit_vector[env_idx] = unit_vector.clone()
            self.cum_rotation_angle[env_idx] = 0.0

            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = int(dof[0].item())
            stiffness = torch.tensor(self.agent.controller.config.stiffness, device=self.device)
            damping = torch.tensor(self.agent.controller.config.damping, device=self.device)
            force_limit = torch.tensor(self.agent.controller.config.force_limit, device=self.device)
            if (
                not hasattr(self, "controller_param")
                or len(self.controller_param) != 3
                or self.controller_param[0].shape != (self.num_envs, dof)
            ):
                self.controller_param = (
                    torch.zeros((self.num_envs, dof), device=self.device),
                    torch.zeros((self.num_envs, dof), device=self.device),
                    torch.zeros((self.num_envs, dof), device=self.device),
                )
            self.controller_param[0][env_idx] = stiffness.expand(b, dof)
            self.controller_param[1][env_idx] = damping.expand(b, dof)
            self.controller_param[2][env_idx] = force_limit.expand(b, dof)

        self.success_threshold = torch.tensor(
            math.radians(self.success_threshold_degrees),
            device=self.device,
        )

    def evaluate(self, **kwargs) -> dict:
        info = super().evaluate(**kwargs)

        close_tip_count = torch.sum(
            info["obj_tip_dist"] < self.success_tip_distance_threshold,
            dim=-1,
        )
        object_in_hand = (~info["obj_fall"]) & (
            close_tip_count >= self.success_min_close_tips
        )

        info["success_raw"] = info["success"]
        info["object_in_hand"] = object_in_hand
        info["success"] = info["success"] & object_in_hand
        return info

    def compute_dense_reward(self, obs, action, info: dict):
        reward = 20 * info["rotation_angle"] * info["object_in_hand"].to(torch.float32)

        obj_vel = info["obj_vel"]
        reward += -0.1 * obj_vel

        obj_fall = info["obj_fall"]
        reward += -50.0 * obj_fall

        power = torch.abs(info["power"])
        reward += -0.0003 * power

        qf = info["qf"]
        qf_norm = torch.linalg.norm(qf, dim=-1)
        reward += -0.0003 * qf_norm

        obj_tip_dist = info["obj_tip_dist"]
        distance_rew = 0.1 / (0.02 + 4 * obj_tip_dist)
        reward += torch.mean(torch.clip(distance_rew, 0, 1), dim=-1)

        return reward


@register_env("HoldCubeInHand-v1", max_episode_steps=100)
class HoldCubeInHand(EasierRotateSingleObjectInHandLevel0):
    """A simple AllegroHandRightTouch task that only requires holding the cube.

    The scene, robot, cube, and reset pose are inherited from the Level0 in-hand
    rotation task. Success is based on keeping the cube from falling while enough
    fingertips stay close to the object for a short continuous window.
    """

    hold_tip_distance_threshold: float = 0.12
    hold_min_close_tips: int = 2
    hold_success_steps: int = 10

    def _initialize_actors(self, env_idx: torch.Tensor):
        super()._initialize_actors(env_idx)

        if not hasattr(self, "hold_steps") or self.hold_steps.shape != (self.num_envs,):
            self.hold_steps = torch.zeros(
                (self.num_envs,), dtype=torch.long, device=self.device
            )
        self.hold_steps[env_idx] = 0

    def evaluate(self, **kwargs) -> dict:
        info = RotateSingleObjectInHandLevel0.evaluate(self, **kwargs)

        close_tip_count = torch.sum(
            info["obj_tip_dist"] < self.hold_tip_distance_threshold,
            dim=-1,
        )
        object_in_hand = (~info["obj_fall"]) & (
            close_tip_count >= self.hold_min_close_tips
        )

        if not hasattr(self, "hold_steps") or self.hold_steps.shape != (self.num_envs,):
            self.hold_steps = torch.zeros(
                (self.num_envs,), dtype=torch.long, device=self.device
            )
        self.hold_steps = torch.where(
            object_in_hand,
            self.hold_steps + 1,
            torch.zeros_like(self.hold_steps),
        )

        info["close_tip_count"] = close_tip_count
        info["object_in_hand"] = object_in_hand
        info["hold_steps"] = self.hold_steps
        info["success"] = object_in_hand & (self.hold_steps >= self.hold_success_steps)
        info["fail"] = info["obj_fall"]
        return info

    def compute_dense_reward(self, obs, action, info: dict):
        close_tip_count = info["close_tip_count"].to(torch.float32)
        object_in_hand = info["object_in_hand"].to(torch.float32)

        obj_tip_dist = info["obj_tip_dist"]
        distance_rew = 0.1 / (0.02 + 4 * obj_tip_dist)
        reward = torch.mean(torch.clip(distance_rew, 0, 1), dim=-1)

        reward += 0.5 * close_tip_count
        reward += 2.0 * object_in_hand
        reward += 0.05 * torch.clip(info["hold_steps"].to(torch.float32), 0, 20)

        obj_vel = info["obj_vel"]
        reward += -0.1 * obj_vel

        obj_fall = info["obj_fall"]
        reward += -50.0 * obj_fall

        power = torch.abs(info["power"])
        reward += -0.0003 * power

        qf = info["qf"]
        qf_norm = torch.linalg.norm(qf, dim=-1)
        reward += -0.0003 * qf_norm

        return reward

    def compute_normalized_dense_reward(self, obs, action, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0





@register_env("HoldCubeInHandLevel2-v1", max_episode_steps=100)
class HoldCubeInHandLevel2(EasierRotateSingleObjectInHandLevel0):
    """A simple AllegroHandRightTouch task that only requires holding the cube.

    The scene, robot, cube, and reset pose are inherited from the Level0 in-hand
    rotation task. Success is based on keeping the cube from falling while enough
    fingertips stay close to the object for a short continuous window.
    """

    hold_tip_distance_threshold: float = 0.09
    hold_min_close_tips: int = 3
    hold_success_steps: int = 30

    def _initialize_actors(self, env_idx: torch.Tensor):
        super()._initialize_actors(env_idx)

        if not hasattr(self, "hold_steps") or self.hold_steps.shape != (self.num_envs,):
            self.hold_steps = torch.zeros(
                (self.num_envs,), dtype=torch.long, device=self.device
            )
        self.hold_steps[env_idx] = 0

    def evaluate(self, **kwargs) -> dict:
        info = RotateSingleObjectInHandLevel0.evaluate(self, **kwargs)

        close_tip_count = torch.sum(
            info["obj_tip_dist"] < self.hold_tip_distance_threshold,
            dim=-1,
        )
        object_in_hand = (~info["obj_fall"]) & (
            close_tip_count >= self.hold_min_close_tips
        )

        if not hasattr(self, "hold_steps") or self.hold_steps.shape != (self.num_envs,):
            self.hold_steps = torch.zeros(
                (self.num_envs,), dtype=torch.long, device=self.device
            )
        self.hold_steps = torch.where(
            object_in_hand,
            self.hold_steps + 1,
            torch.zeros_like(self.hold_steps),
        )

        info["close_tip_count"] = close_tip_count
        info["object_in_hand"] = object_in_hand
        info["hold_steps"] = self.hold_steps
        info["success"] = object_in_hand & (self.hold_steps >= self.hold_success_steps)
        info["fail"] = info["obj_fall"]
        return info

    def compute_dense_reward(self, obs, action, info: dict):
        close_tip_count = info["close_tip_count"].to(torch.float32)
        object_in_hand = info["object_in_hand"].to(torch.float32)

        obj_tip_dist = info["obj_tip_dist"]
        distance_rew = 0.1 / (0.02 + 4 * obj_tip_dist)
        reward = torch.mean(torch.clip(distance_rew, 0, 1), dim=-1)

        reward += 0.5 * close_tip_count
        reward += 2.0 * object_in_hand
        reward += 0.05 * torch.clip(info["hold_steps"].to(torch.float32), 0, 20)

        obj_vel = info["obj_vel"]
        reward += -0.1 * obj_vel

        obj_fall = info["obj_fall"]
        reward += -50.0 * obj_fall

        power = torch.abs(info["power"])
        reward += -0.0003 * power

        qf = info["qf"]
        qf_norm = torch.linalg.norm(qf, dim=-1)
        reward += -0.0003 * qf_norm

        return reward

    def compute_normalized_dense_reward(self, obs, action, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0
