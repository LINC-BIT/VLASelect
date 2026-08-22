# import threading
# import time
# import traceback
# import sys

# def monitor_threads(log_file="thread_log.txt", interval=2):
#     """周期性打印所有线程状态和堆栈"""
#     with open(log_file, "w") as f:
#         while True:
#             f.write("="*40 + "\n")
#             for t in threading.enumerate():
#                 f.write(f"Thread {t.name} (id={t.ident}): alive={t.is_alive()}\n")
#                 if t.ident in sys._current_frames():
#                     stack = ''.join(traceback.format_stack(sys._current_frames()[t.ident]))
#                     f.write(stack + "\n")
#             f.write("="*40 + "\n\n")
#             f.flush()  # 保证实时写入文件
#             time.sleep(interval)

# threading.Thread(target=monitor_threads, daemon=True, name="ThreadMonitor").start() 

import os
import sys
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
from datetime import datetime
import argparse
import time
from tqdm import tqdm
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import torch
# print(torch.cuda.device_count())
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

from accelerate import Accelerator
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from mani_skill.utils.io_utils import load_json, dump_json
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import gymnasium as gym

import random

from train.octo.model import Agent
from train.reinforcement_learning.utils import collect_rollout, compute_gae, ppo_update_on_policy, get_step_infos, get_stage
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.evaluate import evaluate
import envs.grasp_obj_random
import copy

# =====================================================
# Preprocess
# =====================================================
model_name = 'octo'
robot_name = 'amazinghand_right'
sensor_configs = {
    "shader_pack": "default",
    "width": 128,
    "height": 128
}
cameras=("hand_camera",)
env_settings_1 = dict(
    object_type="computer_mouse",                           
    object_size_info={                                      
        "computer_mouse": {
            'scale': 0.05,
        },                                                  
    },                                      
    object_mass=0.01,                                       
    object_color=[0, 0, 1, 1],                              
    randomize_camera=False,                                 
    randomize_table=False,
    default_friction=True,
    randomize_skybox=False,
    randomize_disruptors=False,
    randomize_blur=True,
    default_lighting=True,
    default_grasp_items=False
)

env_settings_2 = dict(
    object_type="plastic_water_bottle",                                     
    object_size_info={                                      
        'plastic_water_bottle': {
            'scale': 0.0065,
        },                        
    },                                      
    object_mass=0.01,                                        
    object_color=[1, 0, 0, 1],                              
    randomize_camera=False,                                 
    randomize_table=False,
    default_friction=True,
    randomize_skybox=False,
    randomize_disruptors=False,
    randomize_blur=True,
    default_lighting=True,
    default_grasp_items=False
)

env_settings_3 = dict(
    object_type="game_controller",                                     
    object_size_info={                                      
        'game_controller': {
            'scale': 0.35,
        },                        
    },                                      
    object_mass=0.01,                                        
    object_color=[0, 1, 0, 1],                              
    randomize_camera=False,                                 
    randomize_table=False,
    default_friction=True,
    randomize_skybox=False,
    randomize_disruptors=False,
    randomize_blur=True,
    default_lighting=True,
    default_grasp_items=False
)

env_settings_4 = dict(
    object_type="mobile_charger",                                     
    object_size_info={                                      
        'mobile_charger': {
            'scale': 0.025,
        },                        
    },                                      
    object_mass=0.01,                                        
    object_color=[1, 1, 0, 1],                              
    randomize_camera=False,                                 
    randomize_table=False,
    default_friction=True,
    randomize_skybox=False,
    randomize_disruptors=False,
    randomize_blur=True,
    default_lighting=True,
    default_grasp_items=False
)

env_settings_5 = dict(
    object_type="computer_mouse",                           
    object_size_info={                                      
        "computer_mouse": {
            'scale': 0.05,
        },                                                  
    },                                      
    object_mass=0.01,                                        
    object_color=[0, 0, 1, 1],                              
    randomize_camera=False,                                 
    randomize_table=False,
    default_friction=True,
    randomize_skybox=False,
    randomize_disruptors=False,
    randomize_blur=True,
    default_lighting=False,
    ambient_light_temperature=6500,
    ambient_light_intensity=0.05,
    directional_light_direction=[1, 1, 1],
    default_grasp_items=False
)

env_settings_6 = dict(
    object_type="plastic_water_bottle",                                     
    object_size_info={                                      
        'plastic_water_bottle': {
            'scale': 0.005,
        },                        
    },                                      
    object_mass=0.01,                                         
    object_color=[1, 0, 0, 1],                              
    randomize_camera=False,                                 
    randomize_table=False,
    default_friction=True,
    randomize_skybox=False,
    randomize_disruptors=False,
    randomize_blur=True,
    default_lighting=False,
    ambient_light_temperature=6500,
    ambient_light_intensity=0.05,
    directional_light_direction=[1, 1, 1],
    default_grasp_items=False
)

env_settings_7 = dict(
    object_type="game_controller",                                     
    object_size_info={                                      
        'game_controller': {
            'scale': 0.35,
        },                        
    },                                      
    object_mass=0.01,                                        
    object_color=[0, 1, 0, 1],                              
    randomize_camera=False,                                 
    randomize_table=False,
    default_friction=True,
    randomize_skybox=False,
    randomize_disruptors=False,
    randomize_blur=True,
    default_lighting=False,
    ambient_light_temperature=6500,
    ambient_light_intensity=0.05,
    directional_light_direction=[1, 1, 1],
    default_grasp_items=False
)

env_settings_8 = dict(
    object_type="computer_mouse",                           
    object_size_info={                                      
        "computer_mouse": {
            'scale': 0.05,
        },                                                  
    },                                      
    object_mass=0.01,                                        
    object_color=[0, 0, 1, 1],                              
    randomize_camera=False,                                 
    randomize_table=False,
    default_friction=True,
    randomize_skybox=False,
    randomize_disruptors=False,
    randomize_blur=True,
    default_lighting=False,
    ambient_light_temperature=6500,
    ambient_light_intensity=1,
    directional_light_temperature=6000,
    default_grasp_items=False
)

env_settings_9 = dict(
    object_type="plastic_water_bottle",                                     
    object_size_info={                                      
        'plastic_water_bottle': {
            'scale': 0.005,
        },                        
    },                                      
    object_mass=0.01,                                         
    object_color=[1, 0, 0, 1],                              
    randomize_camera=False,                                 
    randomize_table=False,
    default_friction=True,
    randomize_skybox=False,
    randomize_disruptors=False,
    randomize_blur=True,
    default_lighting=False,
    ambient_light_temperature=6500,
    ambient_light_intensity=1,
    directional_light_temperature=6000,
    default_grasp_items=False
)

env_settings_10 = dict(
    object_type="game_controller",                                     
    object_size_info={                                      
        'game_controller': {
            'scale': 0.35,
        },                        
    },                                      
    object_mass=0.01,                                        
    object_color=[0, 1, 0, 1],                              
    randomize_camera=False,                                 
    randomize_table=False,
    default_friction=True,
    randomize_skybox=False,
    randomize_disruptors=False,
    randomize_blur=True,
    default_lighting=False,
    ambient_light_temperature=6500,
    ambient_light_intensity=1,
    directional_light_temperature=6000,
    default_grasp_items=False
)
# class ActionRepeatWrapperForPosControl(gym.Wrapper):
#     def __init__(self, env, real_freq, delta_pos=False, control_freq=20):
#         self.base_env: BaseEnv = env.unwrapped
#         super().__init__(env)

#         self.repeat = control_freq // real_freq
#         self.delta_pos = delta_pos

#         self.held_action = None

#     def step(self, action):
        
#         def dict_assign(dict_a, dict_b, mask):
#             for k in dict_a.keys():
#                 assert k in dict_b
#                 if isinstance(dict_a[k], torch.Tensor):
#                     dict_a[k][mask] = dict_b[k][mask]
#                 elif isinstance(dict_a[k], dict):
#                     dict_assign(dict_a[k], dict_b[k], mask)
        
#         if isinstance(action, np.ndarray):
#             action = torch.from_numpy(action).to(self.base_env.device)

#         self.held_action = action.clone()

#         obs, reward, term, trunc, infos = None, None, None, None, None

#         final_info_accum = None  # ✔ 用来保存 final_info
#         final_observation_accum = None  # ✔ 用来保存 final_observation（如果需要的话）
#         _final_info_accum = None  # 用于存储 _final_info

#         for i in range(self.repeat):

#             t_obs, t_reward, t_term, t_trunc, t_infos = self.env.step(self.held_action)

#             if i == 0:
#                 obs = t_obs
#                 reward = t_reward
#                 term = t_term
#                 trunc = t_trunc
#                 infos = t_infos

#             done_mask = t_term | t_trunc

#             # ===== action reset =====
#             if done_mask.any():
#                 if self.delta_pos:
#                     self.held_action[done_mask] = self.held_action[done_mask] * 0
#                 else:
#                     tmp_action = self.base_env.agent.start_action.to(action.device)
#                     tmp_action = tmp_action.unsqueeze(0).repeat(action.shape[0], 1)
#                     self.held_action[done_mask] = tmp_action[done_mask]
                
#                 dict_assign(obs, t_obs, done_mask)
#                 reward[done_mask] = t_reward[done_mask]
#                 term[done_mask] = t_term[done_mask]
#                 trunc[done_mask] = t_trunc[done_mask]
#                 dict_assign(infos, t_infos, done_mask)

#                 # ===== ✔ 关键：处理 final_info =====
#                 if "final_info" in t_infos:
#                     if final_info_accum is None:
#                         final_info_accum = copy.deepcopy(t_infos["final_info"])
#                     else:
#                         dict_assign(final_info_accum, t_infos["final_info"], done_mask)
                    
#                     if _final_info_accum is None:
#                         _final_info_accum = copy.deepcopy(t_infos["_final_info"])
#                     else:
#                         _final_info_accum[done_mask] = t_infos["_final_info"][done_mask]
                    
#                     if final_observation_accum is None:
#                         final_observation_accum = copy.deepcopy(t_infos["final_observation"])
#                     else:
#                         dict_assign(final_observation_accum, t_infos["final_observation"], done_mask)

#         # ===== ✔ 写回 final_info =====
#         if final_info_accum is not None:
#             infos["final_info"] = final_info_accum
#             infos["_final_info"] = _final_info_accum
#             infos["final_observation"] = final_observation_accum

#         return obs, reward, term, trunc, infos

from PIL import Image
class FlattenRGBObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation

    Note that the returned observations will have a "rgb" key depending on the rgb bool flags, and will
    always have a "state" key. 
    """

    def __init__(self, env, rgb=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_state = state

        # check if rgb data exists in first camera's sensor data
        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "rgb" not in first_cam:
            self.include_rgb = False
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        rgb_images = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                rgb_images.append(cam_data["rgb"])

        if len(rgb_images) > 0:
            # rgb_images = torch.concat(rgb_images, axis=-1)
            rgb_images = rgb_images[0]
        # flatten the rest of the data which should just be state data
        # print(observation)
        id = np.random.choice(rgb_images.shape[0])
        Image.fromarray(rgb_images[0].cpu().numpy()).save("debug_rgb.png")

        # observation["ee_link_pose"] = observation['extra']["ee_link_pose"].to(self.base_env.device)
        # observation["ee_link_1_pose"] = observation['extra']["ee_link_1_pose"].to(self.base_env.device)
        # observation["ee_link_2_pose"] = observation['extra']["ee_link_2_pose"].to(self.base_env.device)
        # observation["ee_link_3_pose"] = observation['extra']["ee_link_3_pose"].to(self.base_env.device)
        # for j in self.base_env.agent.robot.get_active_joints():
        #     print(j.name)
        # print(self.base_env.agent.robot.get_qpos())
        del observation["extra"]
        observation = common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
        )
        ret = dict()
        if self.include_state:
            ret["state"] = observation
        if self.include_rgb:
            ret["rgb"] = rgb_images
        return ret

def resolve_ckpt_dir(args, model_name):
    task_dir = os.path.join(args.save_dir, f'{args.task_name}/ppo/{args.robot_name}/{model_name}')
    root_dir = os.path.join(task_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    os.makedirs(task_dir, exist_ok=True)

    if args.resume_dir is not None:
        resume_dir = args.resume_dir
    else:
        resume_dir = root_dir

    log_dir = os.path.join(resume_dir, "tb")
    video_dir=os.path.join(resume_dir, 'videos')
    latest_agent = os.path.join(resume_dir, "latest_agent.pt")
    latest_opt = os.path.join(resume_dir, "latest_opt.pt")
    actor_path = os.path.join(args.actor_ckpt_path if args.actor_ckpt_path is not None else '', "last.pt")

    return {
        "task_dir": task_dir,
        "log_dir": log_dir,
        "root_dir": root_dir,
        "video_dir": video_dir,
        "latest_agent": latest_agent,
        "latest_opt": latest_opt,
        "best_agent": os.path.join(resume_dir, "best_agent.pt"),
        "metrics": os.path.join(resume_dir, "metrics.json"),
        "actor_path": actor_path,
        "ft_agent_path": os.path.join(args.ft_agent_path if args.ft_agent_path is not None else '', "best_agent.pt"),
    }

def make_collate_fn(args, cameras, device):
    # device 指将obs移动到什么device, 不是obs是什么device
    def collate_fn(obs):
        if isinstance(obs['rgb'], np.ndarray):
            rgb = torch.from_numpy(obs['rgb'] / 255.0).permute(0, 3, 1, 2).float()
            state = torch.from_numpy(obs["state"])
        else:
            rgb = obs["rgb"].permute(0, 3, 1, 2).float() / 255.0  # (env, H, W, 3)
            state = obs["state"]
        
        def _resize(img, size=128):
            # img = img.unsqueeze(0)          # (1, C, H, W)
            img = F.interpolate(
                img,
                size=size,
                mode='bilinear',
                # align_corners=align_corners if mode != "nearest" else None,
            )
            return img

        rgb = _resize(rgb).to(device)
        state = state.to(device)

        return {
            'rgb': rgb,
            'state': state,
        }

    return collate_fn

def make_sample_fn(args, agent_model, cameras, accelerator, deterministic=True):
    device = accelerator.device
    # device 指将obs移动到什么device, 不是obs是什么device
    def sample_fn(obs):
        if isinstance(obs['rgb'], np.ndarray):
            rgb = torch.from_numpy(obs['rgb'] / 255.0).permute(0, 3, 1, 2).float()
            state = torch.from_numpy(obs["state"])
        else:
            rgb = obs["rgb"].permute(0, 3, 1, 2).float() / 255.0  # (env, H, W, 3)
            state = obs["state"]
        
        def _resize(img, size=128):
            # img = img.unsqueeze(0)          # (1, C, H, W)
            img = F.interpolate(
                img,
                size=size,
                mode='bilinear',
                # align_corners=align_corners if mode != "nearest" else None,
            )
            return img

        rgb = _resize(rgb).to(device)
        state = state.to(device)

        batch = {
            'rgb': rgb,
            'state': state,
        }

        action = agent_model.get_action(batch, deterministic=deterministic)
        # action = torch.zeros_like(action)
        return action

    return sample_fn

def get_agent_info(args, env_kwargs, collate_fn):
    test_env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend='gpu',
        env_kwargs=env_kwargs,
        wrappers=[FlattenRGBObservationWrapper]
    )
    obs, _ = test_env.reset()
    batch = collate_fn(obs)
    infos = dict(state_dim=batch['state'].shape[-1], action_dim=test_env.single_action_space.shape[0])
    return infos

# =====================================================
# Main
# =====================================================
def main(args):
    ckpt = resolve_ckpt_dir(args, model_name)
    step_infos = get_step_infos(args)
    env_kwargs = {
      "obs_mode": "rgb+state_dict",
      "control_mode": "pd_joint_delta_pos",
      "render_mode": "rgb_array",
      "reward_mode": "normalized_dense",
      "shader_dir": "minimal",
      "sim_backend": "physx_cuda",
      "max_episode_steps": args.max_episode_steps,
    }
    env_kwargs.update(**env_settings_2)
    env_kwargs_for_eval = env_kwargs.copy()
    env_kwargs_for_eval.pop('sim_backend')
    env_kwargs['render_mode'] = 'none'
    env_kwargs_for_eval['shader_dir'] = 'default'
    
    tensorboard_port = 6007
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = not args.ignore_torch_deterministic
    accelerator = Accelerator(mixed_precision="bf16" if args.use_amp else "no")
    device = accelerator.device

    if accelerator.is_main_process:
        writer = SummaryWriter(ckpt["log_dir"])
    else:
        writer = None

    infos = get_agent_info(args, env_kwargs_for_eval, make_collate_fn(args, cameras, device))
    agent = Agent(**infos, normalize_state=args.normalize_state, use_depth=False)
    if os.path.exists(ckpt["latest_agent"]):
        print(f"[Train] Resume agent from {ckpt['latest_agent']}")
        agent_dict = torch.load(ckpt["latest_agent"], map_location="cpu")
        agent.load_state_dict(agent_dict)
    elif os.path.exists(ckpt["actor_path"]):
        print(f"[Train] load actor from {ckpt['actor_path']}")
        agent.load_actor(ckpt['actor_path'])
    elif os.path.exists(ckpt['ft_agent_path']):
        print(f"[Train] load ft agent from {ckpt['ft_agent_path']}")
        agent_dict = torch.load(ckpt['ft_agent_path'], map_location="cpu")
        agent.load_state_dict(agent_dict)
        # agent.reset_value_head()

    sta_steps = global_steps = 0
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    start_time = time.time()
    if os.path.exists(ckpt["latest_opt"]):
        print(f"[Train] Resume optimizer from {ckpt['latest_opt']}")
        resume_opt = torch.load(ckpt["latest_opt"], map_location="cpu")
        optimizer.load_state_dict(resume_opt['opt'])
        sta_steps = global_steps = resume_opt['step']

    agent, optimizer = accelerator.prepare(agent, optimizer)

    if args.reset_logstd:
        print(f"[Train] Reset actor log std to -0.5")
        agent.reset_logstd(-0.5)

    best_score = -1
    metrics_log = []

    if args.evaluate_mode:
        if args.eval_agent_dir is None:
            raise ValueError("Please provide --eval-agent-dir for evaluation mode")
        
        if args.num_eval_envs > 16:
            video_dir = None
        else:
            video_dir = f'{ckpt["video_dir"]}_train' if not args.evaluate_mode else os.path.join(args.eval_agent_dir, 'videos_eval')

        eval_envs = make_eval_envs(
            env_id=args.task_name,
            num_envs=args.num_eval_envs,
            sim_backend='gpu',
            # action_repeat=control_freq // real_control_freq,
            env_kwargs=env_kwargs_for_eval,
            video_dir=video_dir,
            wrappers=[
                FlattenRGBObservationWrapper,
                # lambda env: ActionRepeatWrapperForPosControl(env, real_freq=real_control_freq, delta_pos=False)
            ],
        )
        
        agent.load_state_dict(torch.load(os.path.join(args.eval_agent_dir, "best_agent.pt"), map_location="cpu"))
        # agent.load_state_dict(torch.load(os.path.join(args.eval_agent_dir, "latest_agent.pt"), map_location="cpu"))
        agent = accelerator.prepare(agent)
        print("[Evaluate] Start evaluation only mode")
        agent.eval()
        eval_metrics = evaluate(
            n=1,
            sample_fn=make_sample_fn(args, agent, cameras, accelerator, deterministic=True),
            eval_envs=eval_envs,
        )
        for k, v in eval_metrics.items():
            mean = v.mean()
            print(f"eval_{k}_mean={mean}")
        dump_json(os.path.join(ckpt['root_dir'], 'eval_metrics.json'), {k: float(v.mean()) for k, v in eval_metrics.items()})
        return

    pbar = tqdm(total=step_infos['total_steps'], initial=global_steps, ascii=True)
    writer = SummaryWriter(ckpt["log_dir"], purge_step=global_steps)
    print(f"[TensorBoard] Logging to {ckpt['log_dir']}")
    print(f"[TensorBoard] Using `tensorboard --logdir {ckpt['log_dir']} --port {tensorboard_port}` to show the logs")

    eval_envs = make_eval_envs(
        env_id=args.task_name,
        num_envs=args.num_eval_envs,
        sim_backend='gpu',
        # action_repeat=control_freq // real_control_freq,
        env_kwargs=env_kwargs_for_eval,
        video_dir=f'{ckpt["video_dir"]}_train' if not args.evaluate_mode else f'{ckpt["video_dir"]}_eval',
        wrappers=[
            FlattenRGBObservationWrapper,
            # lambda env: ActionRepeatWrapperForPosControl(env, real_freq=real_control_freq, delta_pos=False)    
        ],
    )

    _, _ = eval_envs.reset(seed=args.seed)
    envs = gym.make(args.task_name, num_envs=args.num_envs, **env_kwargs)
    envs = FlattenRGBObservationWrapper(envs)
    # envs = ActionRepeatWrapperForPosControl(envs, real_freq=real_control_freq, delta_pos=False)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=args.ignore_partial_reset, record_metrics=True)
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    if os.path.exists(ckpt["metrics"]):
        metrics_log = load_json(ckpt["metrics"])
    # resume_skip = True if args.resume_dir is not None else False
    resume_skip = False

    if args.minibatch_size == 0:
        args.minibatch_size = step_infos['rollot_steps'] // args.num_minibatch // args.grad_accum_steps

    while global_steps < step_infos['total_steps']:
        agent.eval()
        if not resume_skip and global_steps % (step_infos['save_interval_steps']) == 0:
            torch.save(agent.state_dict(), ckpt['latest_agent'])
            torch.save({'opt': optimizer.state_dict(), 'step': global_steps}, ckpt["latest_opt"])
            
            # ---------------- evaluate ----------------
            eval_metrics = evaluate(
                n=1,
                sample_fn=make_sample_fn(args, agent, cameras, accelerator),
                eval_envs=eval_envs,
            )
            for k, v in eval_metrics.items():
                mean = v.mean()
                writer.add_scalar(f"eval/{k}", mean, global_steps)
                print(f"eval_{k}_mean={mean}")
            score = eval_metrics.get(
                "success_rate", eval_metrics[list(eval_metrics.keys())[0]]
            ).mean()
            pbar.set_postfix(eval_score=score)
            metrics_log.append(dict(step=global_steps, score=float(score)))
            dump_json(ckpt["metrics"], metrics_log)

            if score >= best_score:
                best_score = score
                torch.save(agent.state_dict(), ckpt['best_agent'])
                print(f"[Eval] New best model saved (score={score:.3f})")

        resume_skip = False

        if args.dynamic_lr:
            pass
            # adjust_lr_cnn(optimizer, global_steps, step_infos, init_lr)
        if args.dynamic_clip:
            frac = 1.0 - (global_steps / step_infos["total_steps"])
            args.clip_eps = 0.1 + (args.clip_eps - 0.1) * frac
        if args.dynamic_ent_coef:
            frac = 1.0 - (global_steps / step_infos["total_steps"])
            args.ent_coef = args.ent_coef / 10 + (args.ent_coef * 0.9) * frac

        collate_fn = make_collate_fn(args, cameras, device)
        rollout = collect_rollout(
            args=args, 
            agent=agent, 
            collate_fn=collate_fn, 
            accelerator=accelerator,
            envs=envs,
            next_obs=next_obs,
            next_done=next_done,
            writer=writer,
            global_step=global_steps,
        )

        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, final_val_buf, next_obs, next_done = rollout

        adv_buf, ret_buf = compute_gae(rew_buf, done_buf, val_buf, final_val_buf, next_obs, next_done, agent, collate_fn, args, accelerator)

        data = (
            obs_buf.reshape((-1,)),
            act_buf.reshape((-1,) + envs.single_action_space.shape),
            logp_buf.reshape(-1),
            adv_buf.reshape(-1),
            ret_buf.reshape(-1),
            val_buf.reshape(-1),
        )

        agent.train()
        stats = ppo_update_on_policy(args, agent, optimizer, data, collate_fn, accelerator, get_stage(global_steps, step_infos), writer, -1)
        
        global_steps += step_infos['rollot_steps']
        pbar.update(step_infos['rollot_steps'])

        y_pred, y_true = val_buf.flatten(0, 1).cpu().numpy(), ret_buf.flatten(0, 1).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = (global_steps - sta_steps) / (time.time() - start_time)
        writer.add_scalar("charts/SPS", sps, global_steps)
        writer.add_scalar("loss/policy", stats["policy_loss"], global_steps)
        writer.add_scalar("loss/value", stats["value_loss"], global_steps)
        writer.add_scalar("loss/entropy", stats["entropy"], global_steps)
        writer.add_scalar("loss/approx_kl", stats["approx_kl"], global_steps)
        writer.add_scalar("loss/old_approx_kl", stats["old_approx_kl"], global_steps)
        writer.add_scalar("loss/clip_frac", stats["clip_frac"], global_steps)
        writer.add_scalar("loss/explained_var", explained_var, global_steps)
    writer.close()
    torch.save(agent.state_dict(), ckpt['latest_agent'])
    torch.save({'opt': optimizer.state_dict(), 'step': global_steps}, ckpt["latest_opt"])
# batchsize=num_envs * rollout_steps
# minibatch_size = batchsize // num_minibatch
# 若envs中存在好样本，minibatch_size应该足够大以覆盖好样本，但过大会稀释好样本的影响
# 若资源不够，可以采用grad_accum_steps来累积梯度，相当于增大了minibatch_size
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--actor-ckpt-path", type=str, default='ckpt/PickCube-v1/ours/octo/pretrain_large_model/20260121-092802/checkpoints')
    parser.add_argument("--actor-ckpt-path", type=str, default=None)
    parser.add_argument("--ft-agent-path", type=str, default=None)
    parser.add_argument("--task-name", type=str, default="GraspObjectRandom-v1")
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--total-steps", type=int, default=50_000_000)
    parser.add_argument("--critic-warmup-rollouts", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-eval-envs", type=int, default=10)
    parser.add_argument("--ignore-partial-reset", action="store_true")
    parser.add_argument("--ignore-torch-deterministic", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num_minibatch", type=int, default=32)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--reward-scale", type=int, default=1)
    parser.add_argument("--rollout-minibatch-size", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--dynamic-lr", action="store_true")
    parser.add_argument("--dynamic-clip", action="store_true")
    parser.add_argument("--dynamic-ent-coef", action="store_true")
    parser.add_argument("--normalize-state", action="store_true")
    parser.add_argument("--not-normalize-adv", action="store_true")
    parser.add_argument("--reset-logstd", action="store_true")
    parser.add_argument("--finite-horizon-gae", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    # g 0.956 l 0.966
    parser.add_argument("--gamma", type=float, default=0.956)
    parser.add_argument("--gae-lambda", type=float, default=0.966)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.2)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--save-dir", type=str, default="ckpt")
    parser.add_argument("--resume-dir", type=str, default=None)
    
    parser.add_argument("--save-interval-per-rollout", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--evaluate-mode", action="store_true")
    parser.add_argument("--eval-agent-dir", type=str, default=None)
    
    parser.add_argument("--robot-name", type=str, default="amazinghand_right")
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # 启动监控线程（守护线程，不阻塞主程序）
    main(parse_args())
