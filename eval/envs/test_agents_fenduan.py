import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
sys.path.append(os.getcwd())
import agents.jetcobot # imports your robot and registers it
import agents.dofbot_pro
import agents.mycobot_pro
import agents.dofbot_se
import my_empty_env
import envs.pick_obj_random
import envs.fenduan
from dataclasses import dataclass
from typing import Annotated, Optional
import tyro
import numpy as np
import gymnasium as gym
import mani_skill
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers import RecordEpisode
from pathlib import Path

@dataclass
class Args:
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "dofbot_se"
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "cpu"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    keyframe: Annotated[Optional[str], tyro.conf.arg(aliases=["-k"])] = None
    shader: str = "default"
    keyframe_actions: bool = False
    random_actions: bool = False
    none_actions: bool = False
    zero_actions: bool = True
    sim_freq: int = 100
    control_freq: int = 20
    seed: Annotated[Optional[int], tyro.conf.arg(aliases=["-s"])] = None

def main(args: Args):
    # =========================
    # 1. 创建原始环境
    # =========================
    env_settings = dict(
        object_type="cube",                                     # "cube", "sphere", "cylinder", "box", 或者 None 表示随机选择
        object_size_info={                                      # 一个字典，预设物体尺寸参数 (m)，设置为 {} 表示随机选择
            'cube': {'half_size': 0.015},                        # cube 的边长为 0.06m
        },                                      
        object_mass=0.1,                                        # 物体质量 (kg)，None 表示随机选择
        object_color=[1, 0, 0, 1],                              # 物体颜色，输入一个 RGBA 列表，None 表示随机选择，各维度 0 ~ 1 之间
        randomize_camera=False,                                 # 是否随机摄像头位置, None 表示部分随机
    )
    base_env = gym.make(
        "ShipRustRemoval-v1",
        # "PickObjectRandomDofbot-v1",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode=args.control_mode,
        # robot_uids=args.robot_uid,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        render_mode="rgb_array",  # ⚠️ 录视频必须是 rgb_array
        sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
        sim_backend=args.sim_backend,
        max_episode_steps=220,
        # **env_settings
    )

    # =========================
    # 2. 用 RecordEpisode 包装
    # =========================
    video_dir = Path("videos")
    video_dir.mkdir(exist_ok=True)

    rec_env = RecordEpisode(
        base_env,
        output_dir=video_dir,
        save_trajectory=False,      # True 可保存 npz
        save_video=True,
        video_fps=args.control_freq,
        info_on_video=False
    )

    # =========================
    # 3. reset
    # =========================
    rec_env.reset(seed=0)
    env: BaseEnv = rec_env.unwrapped

    print(f"Selected robot {args.robot_uid}. Control mode: {args.control_mode}")
    print("Selected Robot has the following keyframes to view: ")
    print(env.agent.keyframes.keys())

    env.agent.robot.set_qpos(env.agent.robot.qpos * 0)

    kf = None
    if len(env.agent.keyframes) > 0:
        kf_name = None
        if args.keyframe is not None:
            kf_name = args.keyframe
            kf = env.agent.keyframes[kf_name]
        else:
            for kf_name, kf in env.agent.keyframes.items():
                break

        if kf.qpos is not None:
            env.agent.robot.set_qpos(kf.qpos)
            env.agent.controller.reset()
        if kf.qvel is not None:
            env.agent.robot.set_qvel(kf.qvel)

        env.agent.robot.set_pose(kf.pose)
        if kf_name is not None:
            print(f"Viewing keyframe {kf_name}")

    if env.gpu_sim_enabled:
        env.scene._gpu_apply_all()
        env.scene.px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()

    # =========================
    # 4. 主循环（录视频）
    # =========================
    step = 0
    stage = 0
    pre_action = np.zeros(6)
    while True:
        print(f'step={step}')
        mins = np.array([-1.571, -1.571, -1.571, -1.571, -1.571, -1])
        maxs = np.array([1.571, 1.571, 1.571, 1.571, 3.142, 1])
        f = lambda x:2 * (x - mins) / (maxs - mins) - 1
        if args.random_actions:
            action = rec_env.action_space.sample()
        elif args.none_actions:
            action = None
        elif args.zero_actions:
            # action = np.array([0, -0.8, -1, 0])
            action = np.zeros(6)
            # action = np.array([0, 0.9, 0.9, 0.5, 0, -1])
            # action = np.array([0, -0.5, -0.5, -0.5, 0, -1])

            # action = f(np.array([-1.571, -0.394, -0.710, -1.310, 0, 0.5]))
            # action = np.array([0, 0, 0, 0, 0, -1])
        elif args.keyframe_actions:
            assert kf is not None, "this robot has no keyframes"
            if isinstance(rec_env.agent.controller, DictController):
                action = rec_env.agent.controller.from_qpos(kf.qpos)
            else:
                action = kf.qpos
        else:
            action = None
        
        type = "zong_r"

        # rust
        tar_action1 = np.array([-1.571, -0.145, -0.880, -1.571, 0, 0.5])
        tar_action1_l = np.array([1.571, -0.145, -0.880, -1.571, 0, 0.5])
        tar_action2 = np.array([0, 0.644, -0.999, -1.209, 0, 0.5])

        # heng (ee_pose -0.75, 0.177, 0.264, 0.317, -0.611, 0.284, 0.667)
        if type == "heng_r":
            # 针对 rust right
            # l1 = np.array([0.682, -0.555, -0.454, -0.684, 0, 0.5])
            # m1 = np.array([0, -0.245, 0.223, -1.380, 0, 0.5])
            # r1 = np.array([-0.682, -0.205, -0.454, -0.684, 0, 0.5])
            # l2 = np.array([0.762, 0.020, -0.854, -0.684, 0, 0.5])
            # m2 = np.array([0, 0.005, -0.223, -1.380, 0, 0.5])
            # r2 = np.array([-0.762, 0.015, -0.854, -0.684, 0, 0.5])

            # 针对mosha / penqi
            l1 = np.array([0.762, 0.065, -0.854, -0.684, 0, 0.5])
            m1 = np.array([0, 0.055, -0.223, -1.380, 0, 0.5])
            r1 = np.array([-0.762, 0.065, -0.854, -0.684, 0, 0.5])
            l2 = np.array([0.682, -0.34, -0.554, -0.684, 0, 0.5])
            m2 = np.array([0, -0.33, 0.023, -1.380, 0, 0.5])
            r2 = np.array([-0.682, -0.34, -0.554, -0.684, 0, 0.5])

        # left
        elif type == "heng_l":
            # 针对 rust right
            # l1 = np.array([0.682, -0.155, -0.454, -0.684, 0, 0.5])
            # m1 = np.array([0, -0.245, 0.223, -1.380, 0, 0.5])
            # r1 = np.array([-0.782, -0.225, -0.454, -0.684, 0, 0.5])
            # l2 = np.array([0.762, -0.1, -0.854, -0.684, 0, 0.5])
            # m2 = np.array([0, 0.005, -0.223, -1.380, 0, 0.5])
            # r2 = np.array([-0.762, 0.015, -0.854, -0.684, 0, 0.5])

            # 针对mosha / penqi
            l1 = np.array([0.762, 0.065, -0.854, -0.684, 0, 0.5])
            m1 = np.array([0, 0.055, -0.223, -1.380, 0, 0.5])
            r1 = np.array([-0.552, 0.065, -0.854, -0.684, 0, 0.5])
            l2 = np.array([0.802, -0.34, -0.554, -0.684, 0, 0.5])
            m2 = np.array([0, -0.33, 0.023, -1.380, 0, 0.5])
            r2 = np.array([-0.502, -0.34, -0.554, -0.684, 0, 0.5])
        
        elif type == "zong_l":
            # for rust
            # ul1 = np.array([0.83, -0.542, -0.123, -0.486, 0, 0.5])
            # dl1 = np.array([0.821, -0, -1.515, -0.483, 0, 0.5])
            # dl2 = np.array([0.701, 0.2, -1.515, -0.683, 0, 0.5])
            # ul2 = np.array([0.703, -0.05, -0.5, -0.8, 0, 0.5])

            ul1 = np.array([0.553, -0.692, -0.223, -0.486, 0, 0.5])
            dl1 = np.array([0.561, -0.7, -0.715, -0.583, 0, 0.5])
            dl2 = np.array([0.481, -0.4, -1.115, -0.483, 0, 0.5])
            ul2 = np.array([0.473, -0.642, -0.223, -0.483, 0, 0.5])

        elif type == "zong_r":
            # for rust
            # ur1 = np.array([-0.723, -0.542, -0.123, -0.486, 0, 0.5])
            # dr1 = np.array([-0.731, -0, -1.515, -0.483, 0, 0.5])
            # dr2 = np.array([-0.601, -0, -1.515, -0.483, 0, 0.5])
            # ur2 = np.array([-0.593, -0.05, -0.5, -0.8, 0, 0.5])
            
            # for mosha / penqi
            ur1 = np.array([-0.553, -0.692, -0.223, -0.486, 0, 0.5])
            dr1 = np.array([-0.561, -0.7, -0.715, -0.583, 0, 0.5])
            dr2 = np.array([-0.481, -0.4, -1.115, -0.483, 0, 0.5])
            ur2 = np.array([-0.473, -0.642, -0.223, -0.483, 0, 0.5])

        tar_action3 = f(np.array([-0.05, -0.5, -0.5, -0.95, 0, 0.5]))
        
        # action = np.zeros(4)
        # gain = 0.8
        # ############################################
        # # Stage 0: 打开夹爪
        # ############################################
        # if stage == 0:
        #     action[-1] = -1.0   # 张开
        #     if step > 10:
        #         stage = 1

        # ############################################
        # # Stage 1: 移动到物体上方
        # ############################################
        # elif stage == 1:
        #     target = object_pos + np.array([0, 0, 0.01])  # 上方8cm
        #     delta = target - ee_pos
        #     action[:3] = np.clip(gain * delta / 0.1, -1, 1)
        #     action[-1] = -1.0

        #     if np.linalg.norm(delta) < 0.005:
        #         stage = 2

        # ############################################
        # # Stage 2: 下移靠近物体
        # ############################################
        # elif stage == 2:
        #     target = object_pos + np.array([0, 0, 0.005])
        #     delta = target - ee_pos
        #     action[:3] = np.clip(gain * delta / 0.1, -1, 1)
        #     action[-1] = -1.0

        #     if np.linalg.norm(delta) < 0.003:
        #         stage = 3

        # ############################################
        # # Stage 3: 闭合夹爪
        # ############################################
        # elif stage == 3:
        #     action[-1] = 1.0
        #     if step > 20:
        #         stage = 4

        # ############################################
        # # Stage 4: 抬起
        # ############################################
        # elif stage == 4:
        #     target = object_pos + np.array([0, 0, 0.01])
        #     delta = target - ee_pos
        #     action[:3] = np.clip(gain * delta / 0.1, -1, 1)
        #     action[-1] = 1.0

        #     if np.linalg.norm(delta) < 0.005:
        #         stage = 5

        # ############################################
        # # Stage 5: 移到goal
        # ############################################
        # elif stage == 5:
        #     target = goal_pos
        #     delta = target - ee_pos
        #     action[:3] = np.clip(gain * delta / 0.1, -1, 1)
        #     action[-1] = 1.0

        # if step > 20:
        #     action[-1] = 0.6
        # if step > 10:
        #     if step % 4 < 2:
        #         action[-1] = -1
        #     else:
        #         action[-1] = 1

        # 越小越慢，推荐 0.02~0.1
        
        # print(rec_env.enable_particle)

        # heng right
        if type == "heng_r":
            if step < 30:
                action = tar_action1
            elif step >= 30 and step < 50:
                action = pre_action + 0.2 * (r1 - pre_action)
            elif step >= 50 and step < 90:
                rec_env.start_particle()
                action = pre_action + 0.04 * (m1 - pre_action)
            elif step >= 90 and step < 130:
                action = pre_action + 0.04 * (l1 - pre_action)
            elif step >= 130 and step < 140:
                action = l2
            elif step >= 140 and step < 180:
                action = pre_action + 0.04 * (m2 - pre_action)
            elif step >= 180 and step < 220:
                action = pre_action + 0.04 * (r2 - pre_action)

        elif type == "heng_l":
            if step < 30:
                action = tar_action1_l
            elif step >= 30 and step < 50:
                action = pre_action + 0.2 * (l1 - pre_action)
            elif step >= 50 and step < 90:
                rec_env.start_particle()
                action = pre_action + 0.04 * (m1 - pre_action)
            elif step >= 90 and step < 130:
                action = pre_action + 0.04 * (r1 - pre_action)
            elif step >= 130 and step < 140:
                action = r2
            elif step >= 140 and step < 180:
                action = pre_action + 0.04 * (m2 - pre_action)
            elif step >= 180 and step < 220:
                action = pre_action + 0.04 * (l2 - pre_action)
        elif type == "zong_r":
            if step < 30:
                action = tar_action1
            elif step >= 30 and step < 50:
                action = pre_action + 0.2 * (ur1 - pre_action)
            elif step >= 50 and step < 130:
                rec_env.start_particle()
                action = pre_action + 0.04 * (dr1 - pre_action)
            elif step >= 130 and step < 140:
                action = dr2
            elif step >= 140 and step < 220:
                action = pre_action + 0.04 * (ur2 - pre_action)
        elif type == "zong_l":
            if step < 30:
                action = tar_action1_l
            elif step >= 30 and step < 50:
                action = pre_action + 0.2 * (ul1 - pre_action)
            elif step >= 50 and step < 130:
                rec_env.start_particle()
                action = pre_action + 0.04 * (dl1 - pre_action)
            elif step >= 130 and step < 140:
                action = dl2
            elif step >= 140 and step < 220:
                action = pre_action + 0.04 * (ul2 - pre_action)


        # if step > 50 and step <= 60:
        #     action = np.array([0, 0.6, 0.4, 0.1, 0, 1]) 
        # if step > 60:
        #     if step % 20 < 10:
        #         action = np.array([0 - (step % 10) * 0.15, 0.6, 0.4, 0.1, 0, 0.6])
        #     else:
        #         action = np.array([-1.5 + (step % 10) * 0.15, 0.6, 0.4, 0.1, 0, 0.6])
        # env.agent.robot
        pre_action = action
        obs, reward, terminated, truncated, info = rec_env.step(action)
        
        step += 1

        if truncated:
            break

    # =========================
    # 5. 关闭并写文件
    # =========================
    rec_env.close()
    print(f"Video saved to: {video_dir}")

if __name__ == "__main__":
    main(tyro.cli(Args))
