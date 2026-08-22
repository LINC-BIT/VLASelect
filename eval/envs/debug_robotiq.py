import argparse

import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer
import agents.jetcobot # imports your robot and registers it
import agents.dofbot_pro
import agents.mycobot_pro
import agents.dofbot_se
import sys
import os
sys.path.append(os.getcwd())
import envs.pick_obj_random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--model_constraint",
        action="store_true",
        default=False,
        help="Whether to model the joint constraint for the gripper.",
    )
    parser.add_argument(
        "-p",
        "--pause",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def add_gripper_constraint(robot, scene):
    outer_knuckle = next(
        j for j in robot.get_active_joints() if j.name == "right_outer_knuckle_joint"
    )
    outer_finger = next(
        j for j in robot.get_active_joints() if j.name == "right_inner_finger_joint"
    )
    inner_knuckle = next(
        j for j in robot.get_active_joints() if j.name == "right_inner_knuckle_joint"
    )
    pad = outer_finger.get_child_link()
    lif = inner_knuckle.get_child_link()

    T_pw = pad.pose.inv().to_transformation_matrix()
    p_w = (
        outer_finger.get_global_pose().p
        + inner_knuckle.get_global_pose().p
        - outer_knuckle.get_global_pose().p
    )
    T_fw = lif.pose.inv().to_transformation_matrix()
    p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
    p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]
    right_drive = scene.create_drive(lif, sapien.Pose(p_f), pad, sapien.Pose(p_p))
    right_drive.set_limit_x(low=0, high=0, stiffness=0, damping=0)
    right_drive.set_limit_y(low=0, high=0, stiffness=0, damping=0)
    right_drive.set_limit_z(low=0, high=0, stiffness=0, damping=0)

    outer_knuckle = next(
        j for j in robot.get_active_joints() if j.name == "finger_joint"
    )
    outer_finger = next(
        j for j in robot.get_active_joints() if j.name == "left_inner_finger_joint"
    )
    inner_knuckle = next(
        j for j in robot.get_active_joints() if j.name == "left_inner_knuckle_joint"
    )

    pad = outer_finger.get_child_link()
    lif = inner_knuckle.get_child_link()

    T_pw = pad.pose.inv().to_transformation_matrix()
    p_w = (
        outer_finger.get_global_pose().p
        + inner_knuckle.get_global_pose().p
        - outer_knuckle.get_global_pose().p
    )
    T_fw = lif.pose.inv().to_transformation_matrix()
    p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
    p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]

    left_drive = scene.create_drive(lif, sapien.Pose(p_f), pad, sapien.Pose(p_p))
    left_drive.set_limit_x(low=0, high=0, stiffness=0, damping=0)
    left_drive.set_limit_y(low=0, high=0, stiffness=0, damping=0)
    left_drive.set_limit_z(low=0, high=0, stiffness=0, damping=0)
    
    print("Finished adding gripper constraint.")

args = parse_args()
engine = sapien.Engine()
# import pdb; pdb.set_trace()
renderer = sapien.SapienRenderer(offscreen_only=False)
engine.set_renderer(renderer)
scene = engine.create_scene()
scene.set_environment_map("../assets/robots/robotiq/env.ktx")
scene.add_directional_light([0.2, 0.2, -1], [1, 1, 1], True, shadow_scale=3, shadow_near=-5, shadow_far=5)
scene.add_ground(-0.9, render=True)

# Viewer
viewer = Viewer(renderer)
viewer.set_scene(scene)
viewer.set_camera_xyz(0, 0, 0.3)
viewer.set_camera_rpy(0, -1.57, 0)
viewer.control_window.toggle_joint_axes(0)


# Articulation
urdf_file = "../assets/robots/robotiq/robotiq_85_original.urdf"
loader = scene.create_urdf_loader()
builder = loader.load_file_as_articulation_builder(urdf_file)

# Disable self collision for simplification
for link_builder in builder.link_builders:
    link_builder.collision_groups = [1, 1, 2, 0]
robot = builder.build() #fix_root_link=True)
robot.set_qpos(np.zeros([robot.dof]))
robot.set_pose(sapien.Pose(p=np.array([0.0, 0, 0]), q=np.array([0.7071, 0, 0.7071, 0])))

# MAdd constraints
if args.model_constraint:
    add_gripper_constraint(robot, scene)

right_joint = next(
    j for j in robot.get_active_joints() if j.name == "right_outer_knuckle_joint"
)
print(right_joint.name)
right_joint.set_drive_property(1e4, 2000, 0.1)
right_joint.set_drive_target(0.5)

left_joint = next(
    j for j in robot.get_active_joints() if j.name == "finger_joint"
)
print(left_joint.name)
left_joint.set_drive_property(1e4, 2000, 0.1) #(1e3, 1e2, 100)
left_joint.set_drive_target(0.5)

viewer.paused = args.pause
while not viewer.closed:
    # qpos = robot.get_qpos()
    # qf = np.zeros_like(qpos)
    # correction = min((qpos[0] - qpos[3]) * 10, 1)
    # qf[3] += correction
    # qf[0] -= correction
    # robot.set_qf(qf)

    scene.update_render()
    scene.step()
    viewer.render()