import argparse
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import sapien
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import envs.pick_cube_alt_camera  # noqa: F401
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def image_to_numpy(image):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def move_to_grasped_cube(env, seed):
    env.reset(seed=seed)
    base_env = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base_env.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )

    try:
        obb = get_actor_obb(base_env.cube)
        approaching = np.array([0, 0, -1])
        target_closing = (
            base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
            .cpu()
            .numpy()
        )
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.025,
        )
        grasp_pose = base_env.agent.build_grasp_pose(
            approaching, grasp_info["closing"], base_env.cube.pose.sp.p
        )

        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        planner.move_to_pose_with_screw(reach_pose)
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper(t=10)

        lift_pose = sapien.Pose(grasp_pose.p + np.array([0.0, 0.0, 0.12]), grasp_pose.q)
        planner.move_to_pose_with_screw(lift_pose, refine_steps=2)
    finally:
        planner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PickCubeAltCamera-v1")
    parser.add_argument("--output", default="vis/pick_cube_alt_camera_grasp.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
    )
    try:
        move_to_grasped_cube(env, args.seed)
        image = image_to_numpy(env.render())
        Image.fromarray(image).save(output_path)
    finally:
        env.close()

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
