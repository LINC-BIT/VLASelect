import os

from ours.libs.train_with_fbs.lib_transformer import svd_decompose_linear
from ours.pretrain_fbs_model.main import add_FBS_into_cnn
from ours.utils.dl.common.model import get_module, set_module
print('PID: ', os.getpid())


import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torch.multiprocessing
# # 将策略改为 file_system，这就不会占用 /dev/shm 了
# torch.multiprocessing.set_sharing_strategy('file_system')
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from behavior_cloning.evaluate import evaluate
# from behavior_cloning.make_env import make_eval_envs


from collections import defaultdict
from typing import Callable
import numpy as np
import torch

def evaluate(n: int, sample_fn: Callable, eval_envs):
    """
    Evaluate the agent on the evaluation environments for at least n episodes.

    Args:
        n: The minimum number of episodes to evaluate.
        sample_fn: The function to call to sample actions from the agent by passing in the observations
        eval_envs: The evaluation environments.

    Returns:
        A dictionary containing the evaluation results.
    """

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n:
            action = sample_fn(obs)
            obs, _, _, truncated, info = eval_envs.step(action)
            # print(obs)
            # exit()
            # note as there are no partial resets, truncated is True for all environments at the same time
            if truncated.any():
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics


from typing import Optional
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers import RecordEpisode, CPUGymWrapper


def make_eval_envs(env_id, num_envs: int, sim_backend: str, env_kwargs: dict, video_dir: Optional[str] = None, wrappers: list[gym.Wrapper] = []):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "cpu":
        def cpu_make_env(env_id, seed, video_dir=None, env_kwargs = dict()):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True, source_type="behavior_cloning", source_desc="behavior_cloning evaluation rollout")
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk
        vector_cls = gym.vector.SyncVectorEnv if num_envs == 1 else lambda x : gym.vector.AsyncVectorEnv(x, context="forkserver")
        env = vector_cls([cpu_make_env(env_id, seed, video_dir if seed == 0 else None, env_kwargs) for seed in range(num_envs)])
    else:
        env = gym.make(env_id, num_envs=num_envs, sim_backend=sim_backend, reconfiguration_freq=1, **env_kwargs)
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, save_video=True, source_type="behavior_cloning", source_desc="behavior_cloning evaluation rollout", max_steps_per_video=max_episode_steps)
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v0"
    """the id of the environment"""
    demo_path: str = "data/ms2_official_demos/rigid_body/PegInsertionSide-v0/trajectory.state.pd_ee_delta_pose.h5"
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""

    # Behavior cloning specific arguments
    normalize_states: bool = False
    """if toggled, states are normalized to mean 0 and standard deviation 1"""
    lr: float = 3e-4
    """the learning rate for the actor"""
    normalize_states: bool = False
    """if toggled, states are normalized to mean 0 and standard deviation 1"""

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 1000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None

    eval_model_only: Optional[str] = None

    scheduler_step_size: int = 400000
    scheduler_gamma: float = 0.2

    continue_train_from: Optional[str] = None

    pretrained_model: str = ''

    load_small_data_for_quick_val: bool = False

    max_norm: Optional[float] = None

    use_bn: Optional[bool] = False

    tag: Optional[str] = None


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


def flatten_state_dict_with_space(state_dict: dict) -> np.ndarray:
    states = []
    for key in state_dict.keys():
        value = state_dict[key]
        if isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            state = int(value)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
        elif isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            if value.ndim > 2:
                raise AssertionError(
                    "The dimension of {} should not be more than 2.".format(key)
                )
            state = value
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
        if state is not None:
            states.append(state)
    if len(states) == 0:
        return np.empty(0)
    else:
        if isinstance(states[0], torch.Tensor):
            try:
                return torch.hstack(states)
            except:
                return torch.column_stack(states)
        else:
            try:
                return np.hstack(states)
            except:  # dirty fix for concat trajectory of states
                return np.column_stack(states)


# class ManiSkillDataset(Dataset):
#     def __init__(self, dataset_file: str, device: torch.device, load_count) -> None:
#         self.dataset_file = dataset_file
#         # for details on how the code below works, see the
#         # quick start tutorial
#         self.data = h5py.File(dataset_file, "r")
#         json_path = dataset_file.replace(".h5", ".json")
#         self.json_data = load_json(json_path)
#         self.episodes = self.json_data["episodes"]

#         self.env_info = self.json_data["env_info"]
#         self.env_id = self.env_info["env_id"]
#         self.env_kwargs = self.env_info["env_kwargs"]

#         self.camera_data = defaultdict(list)
#         self.actions = []
#         self.dones = []
#         self.states = []
#         self.total_frames = 0
#         self.device = device

#         if load_count is None:
#             load_count = len(self.episodes)

#         if os.path.exists(dataset_file + '.processed'):
#             self.states, self.actions, self.camera_data = torch.load(dataset_file + '.processed')
#             print(f'load processed data from {dataset_file + ".processed"}')
#         else:
#             for eps_id in tqdm(range(load_count), desc='loading dataset...'):
#                 eps = self.episodes[eps_id]
#                 trajectory = self.data[f"traj_{eps['episode_id']}"]
#                 trajectory = load_h5_data(trajectory)
#                 agent = trajectory["obs"]["agent"]
#                 extra = trajectory["obs"]["extra"]

#                 state = np.hstack(
#                     [
#                         flatten_state_dict_with_space(agent),
#                         flatten_state_dict_with_space(extra),
#                     ]
#                 )
#                 self.states.append(state)

#                 # we use :-1 here to ignore the last observation as that
#                 # is the terminal observation which has no actions
#                 for camera_name, camera_data in trajectory["obs"]["sensor_data"].items():
#                     self.camera_data[camera_name + "_rgb"].append(camera_data["rgb"][:-1])
#                     self.camera_data[camera_name + "_depth"].append(camera_data["depth"][:-1])
        
#                 self.actions.append(trajectory["actions"])
#             for key in self.camera_data.keys():
#                 if "rgb" in key:
#                     self.camera_data[key] = np.vstack(self.camera_data[key]) / 255.0
#                 else:
#                     self.camera_data[key] = np.vstack(self.camera_data[key]) / 1024.0
                
#             self.states = np.vstack(self.states)
#             self.actions = np.vstack(self.actions)
#             for key in self.camera_data.keys():
#                 assert self.camera_data[key].shape[0] == self.actions.shape[0]

#             # torch.save((self.states, self.actions, self.camera_data), dataset_file + '.processed')

#     def __len__(self):
#         return len(self.camera_data[list(self.camera_data.keys())[0]])

#     def __getitem__(self, idx):
#         out = {}
#         out["action"] = (
#             torch.from_numpy(self.actions[idx]).float().to(device=self.device)
#         )
#         out["state"] = torch.from_numpy(self.states[idx]).float().to(device=self.device)
#         rgbd_data = []
#         for key in sorted(self.camera_data.keys()):
#             rgbd_data.append(torch.from_numpy(self.camera_data[key][idx]).float().to(device=self.device))
#         out["rgbd"] = torch.cat(rgbd_data, dim=-1)

#         return out



def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


# class ManiSkillDataset(Dataset):
#     def __init__(
#         self,
#         dataset_file: str,
#         cameras=("base_camera",),
#         load_count=-1,
#         normalize_states=False,
#         need_states=True,
#         task_name="maniskill",
#         task_instruction="pick up the cube",
#         preprocess_fn=None
#     ):
#         self.dataset_file = dataset_file
#         self.cameras = cameras
#         self.task_name = task_name
#         self.task_instruction = task_instruction
#         self.need_states = need_states
#         self.preprocess_fn = preprocess_fn

#         self.h5 = h5py.File(dataset_file, "r")
#         json_path = dataset_file.replace(".h5", ".json")
#         self.json_data = load_json(json_path)

#         self.episodes = self.json_data["episodes"]

#         if load_count is None or load_count < 0:
#             load_count = len(self.episodes)

#         # ---------- build index ----------
#         self.index = []
#         for eps in self.episodes[:load_count]:
#             traj_key = f"traj_{eps['episode_id']}"
#             T = self.h5[traj_key]["actions"].shape[0]
#             for t in range(T - 1):
#                 self.index.append((traj_key, t))

#         # ---------- optional state normalization ----------
#         self.state_mean = None
#         self.state_std = None
#         if normalize_states:
#             self._load_or_compute_state_stats()

#     # def _load_or_compute_state_stats(self):
#     #     states = []
#     #     for i in len(self):
#     #         batch = self.__getitem__(i)
#     #         state = batch['state']
#     #         states += [state]
#     #     states = np.vstack(states)
#     #     self.state_mean = np.mean(states, axis=0)
#     #     self.state_std = np.std(states, axis=0) + 1e-6

#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, idx):
#         traj_key, t = self.index[idx]
#         traj = self.h5[traj_key]

#         # ------------------------------------------------
#         # RGB (choose ONE camera for VLA compatibility)
#         # ------------------------------------------------
#         if len(self.cameras) == 1:
#             cam = self.cameras[0]  # ⭐ 明确选择一个 camera（如 base_camera）
#         else:
#             raise NotImplementedError('暂不支持多摄像机输入')

#         rgb = traj["obs"]["sensor_data"][cam]["rgb"][t]  # (H, W, 3), uint8
#         rgb = rgb / 255.0

#         depth = traj["obs"]["sensor_data"][cam]["depth"][t]
#         depth = depth / 1024.0

#         def _resize(img, size=128):
#             img = img.unsqueeze(0)          # (1, C, H, W)
#             img = F.interpolate(
#                 img,
#                 size=size,
#                 mode='bilinear',
#                 # align_corners=align_corners if mode != "nearest" else None,
#             )
#             return img.squeeze(0)

#         rgb = _resize(torch.from_numpy(rgb).permute(2, 1, 0)).float()
#         depth = _resize(torch.from_numpy(depth).permute(2, 1, 0)).float()

#         # 不知道为什么，回放出来的图片角度不对
#         rgb = torch.rot90(rgb, k=3, dims=(1, 2))
#         depth = torch.rot90(depth, k=3, dims=(1, 2))
#         rgb = torch.flip(rgb, dims=(2,))
#         depth = torch.flip(depth, dims=(2,))

#         action = torch.from_numpy(traj["actions"][t]).float()

#         state = []

#         # print(traj['obs'])
#         # print({})

#         agent = traj["obs"]["agent"]
#         extra = traj["obs"]["extra"]

#         # print(agent['qpos'][t], 
#         #       agent['qvel'][t],
#         #       (extra['is_grasped'][t]),
#         #       (extra['tcp_pose'][t]),
#         #       (extra['goal_pos'][t]))

#         def _process(value):
#             if isinstance(value, (tuple, list)):
#                 state = None if len(value) == 0 else value
#             elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
#                 # x = np.array(1) > 0 is np.bool_ instead of ndarray
#                 state = int(value)
#             elif isinstance(value, (float, np.float32, np.float64)):
#                 state = np.float32(value)
#             elif isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
#                 if value.ndim > 2:
#                     raise AssertionError(
#                         "The dimension should not be more than 2."
#                     )
#                 state = value
#             else:
#                 raise TypeError("Unsupported type: {}".format(type(value)))
#             return state

#         for k in ['qpos', 'qvel']:
#             state += [_process(agent[k][t])]
#             # print(k, agent[k][t])

#         for k in ['is_grasped', 'tcp_pose', 'goal_pos']:
#             state += [_process(extra[k][t])]
#             # print(k, extra[k][t])
#         # exit()
#         state = np.hstack(state)
#         # state = np.hstack(
#         #     [
#         #         flatten_state_dict_with_space(agent),
#         #         flatten_state_dict_with_space(extra),
#         #     ]
#         # )
#         state = torch.from_numpy(state).float()

#         # print(state)

#         # exit()
#         # print(state.size())

#         # return (image, state), action
#         # print(image.size())

#         # if self.state_mean is not None:
#         #     state = (state - self.state_mean) / self.state_std

#         # from ours.utils.dl.common.vis import save_tensor_image
#         # save_tensor_image(rgb, f'./getitem-rgb-{idx}.png')
#         # save_tensor_image(depth, f'./getitem-depth-{idx}.png')

#         return {
#             'rgb': rgb,
#             'depth': depth,
#             'state': state,
#             'action': action
#         }



import torch
import h5py
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import json

class ManiSkillDataset(Dataset):
    def __init__(
        self,
        dataset_file: str,
        cameras=("base_camera",),
        load_count=-1,
        normalize_states=False,
        cache_on_ram=False,  # 新增：是否将数据全量加载到内存
    ):
        self.dataset_file = dataset_file
        self.cameras = cameras
        self.cache_on_ram = cache_on_ram
        
        #         {'agent': {'qpos': tensor([[ 0.0061,  0.3453, -0.0167, -1.9487, -0.0195,  2.3135, -0.8176,  0.0400,
        #   0.0400]]), 'qvel': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.]])}, 'extra': {'is_grasped': tensor([False]), 'tcp_pose': tensor([[ 4.0905e-03, -8.5431e-03,  2.0151e-01,  1.3959e-02,  6.9430e-01,
        #   7.1955e-01, -5.4613e-04]]), 'goal_pos': tensor([[0.0303, 0.0163, 0.0578]]), 'obj_pose': tensor([[-0.0315,  0.0957,  0.0200,  0.8642,  0.0000,  0.0000, -0.5031]]), 'tcp_to_obj_pos': tensor([[-0.0356,  0.1042, -0.1815]]), 'obj_to_goal_pos': tensor([[ 0.0618, -0.0794,  0.0378]])}}
                

        # 预先计算好 state 的 key，避免在 getitem 中重复创建列表
        self.agent_keys = ['qpos', 'qvel']
        self.extra_keys = ['is_grasped', 'tcp_pose', 'goal_pos', 'obj_pose', 'tcp_to_obj_pos', 'obj_to_goal_pos']

        # 1. 临时打开一次文件构建索引
        with h5py.File(dataset_file, "r") as h5_file:
            # 假设 json 与 h5 同名
            json_path = dataset_file.replace(".h5", ".json")
            # 建议这里直接用 json 库读取，不要依赖外部函数 load_json 以保持独立性
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)

            self.episodes = self.json_data["episodes"]
            if load_count is None or load_count < 0:
                load_count = len(self.episodes)

            # ---------- build index ----------
            self.index = []

            self.max_episodes_length = 0

            print(f"Building index for {load_count} episodes...")
            for eps in self.episodes[:load_count]:
                traj_key = f"traj_{eps['episode_id']}"
                # 只需要读取一次 shape，不需要读数据
                if traj_key in h5_file:
                    T = h5_file[traj_key]["actions"].shape[0]

                    self.max_episodes_length = max(self.max_episodes_length, T)

                    # 使用 numpy 直接生成索引，比循环 append 快
                    # 存储 (traj_key, t) 元组消耗较大，改为存储 (traj_idx, t)
                    # 但为了兼容你原有的逻辑，这里先保持 (traj_key, t)
                    for t in range(T):
                        self.index.append((traj_key, t))
            
        print(f"Dataset loaded. Total frames: {len(self.index)}")

        # 多进程句柄容器
        self.h5 = None
        
        # 预定义 Resize 尺寸，避免重复创建参数
        self.img_size = 128

        self.debug = True

    def _get_h5_file(self):
        """懒加载：确保每个 worker 进程拥有独立的文件句柄"""
        if self.h5 is None:
            # 如果内存够大，使用 driver='core' 将文件读入内存，速度会有数量级提升
            # backing_store=False 表示关闭写入时的回写，纯内存模式
            if self.cache_on_ram:
                self.h5 = h5py.File(self.dataset_file, "r", driver='core', backing_store=False)
            else:
                # 开启 rdcc (Raw Data Chunk Cache) 优化读取
                self.h5 = h5py.File(self.dataset_file, "r", rdcc_nbytes=1024*1024*4)
        return self.h5

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        h5 = self._get_h5_file()
        traj_key, t = self.index[idx]
        
        # 缓存层级引用，避免重复通过字符串查找 group
        # 注意：h5py 访问 group 也是有开销的
        traj = h5[traj_key]
        obs = traj["obs"]
        sensor_data = obs["sensor_data"]
        
        # ------------------------------------------------
        # 1. Image Optimization
        # ------------------------------------------------
        cam = self.cameras[0]
        cam_data = sensor_data[cam]

        # 直接读取 numpy (uint8)
        rgb_np = cam_data["rgb"][t]
        depth_np = cam_data["depth"][t]

        # 优化流程：转 Tensor -> Permute -> Float -> Div -> Resize
        # 这样可以在 Resize 前保持一定的计算连续性，但 F.interpolate 需要 float
        
        # RGB
        rgb = torch.from_numpy(rgb_np) # (H, W, 3) uint8
        rgb = rgb.permute(2, 0, 1).float().div_(255.0) # (3, H, W)
        
        # Depth
        depth = torch.from_numpy(depth_np) # (H, W, 1) or (H, W)
        if depth.ndim == 2:
            depth = depth.unsqueeze(2)
        depth = depth.permute(2, 0, 1).float().div_(1024.0) # (1, H, W)

        # 图像处理函数优化
        # 使用 interpolate 处理 4D tensor (1, C, H, W)
        def _process_img(img):
            img = img.unsqueeze(0)
            img = F.interpolate(img, size=self.img_size, mode='bilinear', align_corners=False)
            return img.squeeze(0)

        rgb = _process_img(rgb)
        depth = _process_img(depth)

        # 旋转和翻转优化：
        # 原始逻辑：rot90(k=3) + flip(dims=2)
        # rot90(k=3) 等价于逆时针90度 (transpose + flip)
        # 组合操作可以直接通过 tensor 的索引完成，比调用函数快
        # 假设原始是 (C, H, W)
        # k=3 (270度) -> (C, W, H) 且 W 倒序 -> Flip dim 2 (H)
        # 建议：如果必须这样做，保持原样即可，因为 Resize 后图很小，开销不大了。
        # 但如果能离线处理掉这个逻辑最好。
        # rgb = torch.flip(torch.rot90(rgb, k=3, dims=(1, 2)), dims=(2,))
        # depth = torch.flip(torch.rot90(depth, k=3, dims=(1, 2)), dims=(2,))

        # ------------------------------------------------
        # 2. State Optimization (移除 _process 和循环)
        # ------------------------------------------------
        # 假设数据已经是 clean 的，移除 isinstance 检查
        # 直接读取并拼接 list
        
        agent = obs["agent"]
        extra = obs["extra"]

        #         {'agent': {'qpos': tensor([[ 0.0061,  0.3453, -0.0167, -1.9487, -0.0195,  2.3135, -0.8176,  0.0400,
        #   0.0400]]), 'qvel': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.]])}, 'extra': {'is_grasped': tensor([False]), 'tcp_pose': tensor([[ 4.0905e-03, -8.5431e-03,  2.0151e-01,  1.3959e-02,  6.9430e-01,
        #   7.1955e-01, -5.4613e-04]]), 'goal_pos': tensor([[0.0303, 0.0163, 0.0578]]), 'obj_pose': tensor([[-0.0315,  0.0957,  0.0200,  0.8642,  0.0000,  0.0000, -0.5031]]), 'tcp_to_obj_pos': tensor([[-0.0356,  0.1042, -0.1815]]), 'obj_to_goal_pos': tensor([[ 0.0618, -0.0794,  0.0378]])}}
                
        
        state_parts = []
        
        # 显式读取，减少 getattr 和 key 查找开销
        # 注意：h5py 读取标量需要 [t]，读取出来可能是 numpy scalar
        
        # qpos, qvel
        for k in self.agent_keys:
            val = agent[k][t]
            # 如果是标量，转为 1D array；如果是 array，保持原样
            # 很多时候 h5 存的是 (T, D)，读出来是 (D,)
            state_parts.append(np.atleast_1d(val))

        self.grasped_data_idx = sum([len(sp) for sp in state_parts])
            
        # extra
        for k in self.extra_keys:
            val = extra[k][t]
            state_parts.append(np.atleast_1d(val))
            
        # 一次性 concat，然后转 tensor
        state_np = np.concatenate(state_parts).astype(np.float32)
        state = torch.from_numpy(state_np)

        # Action
        action = torch.from_numpy(traj["actions"][t]).float()

        if self.debug:
            from ours.utils.dl.common.vis import save_tensor_image
            save_tensor_image(rgb, f'ckpt/{run_name}/getitem-rgb-{idx}.png')
            save_tensor_image(depth, f'ckpt/{run_name}/getitem-depth-{idx}.png')
            self.debug = False

        return {
            'rgb': rgb,
            'depth': depth,
            'state': state,
            'action': action
        }


# taken from here
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def save_ckpt(run_name, tag):
    os.makedirs(f"ckpt/{run_name}/checkpoints", exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iteration': iteration
        },
        f"ckpt/{run_name}/checkpoints/{tag}.pt",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        
        # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        from datetime import datetime
        run_name = f'{args.env_id}/ours/octo/{args.exp_name}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        if args.eval_model_only is not None:
            run_name += '-eval'
        if args.tag is not None:
            run_name += f'-{args.tag}'
    else:
        run_name = args.exp_name

    import shutil
    os.makedirs(f"ckpt/{run_name}", exist_ok=True)
    shutil.copyfile(__file__, f"ckpt/{run_name}/script.py")

    if args.demo_path.endswith(".h5"):
        import json

        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert (
                control_mode == args.control_mode
            ), f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

            env_kwargs = demo_info['env_info']['env_kwargs']

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.use_deterministic_algorithms(args.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    control_mode = os.path.split(args.demo_path)[1].split(".")[2]

    # env setup
    # env_kwargs = dict(
    #     control_mode=args.control_mode,
    #     reward_mode="dense",
    #     obs_mode="rgbd",
    #     render_mode="all",
    #     robot_uids='panda_wristcam',
    #     max_episode_step=ds.max_episodes_length,
    # )
    # env_kwargs = {
    #   "obs_mode": "rgb+depth+state_dict",
    #   "control_mode": "pd_ee_delta_pos",
    #   "render_mode": "rgb_array",
    #   "reward_mode": "dense",
    #   "shader_dir": "default",
    #   "sim_backend": "physx_cpu",
    #   "robot_uids": "panda_wristcam",
    #   "sensor_configs": {
    #     "shader_pack": "default",
    #     "width": 224,
    #     "height": 224
    #   },
    #   "num_envs": 1
    # }
    

    if args.track:
        import wandb

        config = vars(args)
        config["eval_env_cfg"] = env_kwargs
        wandb.tensorboard.patch(root_logdir=f"ckpt/{run_name}/tb")
        wandb.init(
            project='EuroSys2026',
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group=f'Maniskill/{args.env_id}/ours/octo/{args.exp_name}',
            # tags=[f'Maniskill/{args.env_id}/ours/octo/{args.exp_name}'.split('/')],
        )
    writer = SummaryWriter(f"ckpt/{run_name}/tb")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    print(f"[TensorBoard] Using `tensorboard --logdir 'ckpt/{run_name}/tb' --port 6009` to show the logs")

    # ds = ManiSkillDataset(
    #     args.demo_path,
    #     device=device,
    #     load_count=args.num_demos,
    # )
    ds = ManiSkillDataset(
        dataset_file=args.demo_path,
        cache_on_ram=True
    )
    ds_dict = {
        'rgb': [],
        'depth': [],
        'state': [],
        'action': []
    }
    n_samples = len(ds) if not args.load_small_data_for_quick_val else 2 * args.batch_size
    pbar = range(n_samples)
    for ds_item_i in tqdm(pbar, total=n_samples, desc='loading samples...'):
        ds_item = ds[ds_item_i]
        for k, v in ds_item.items():
            ds_dict[k] += [v.to(device)]
    for k, v in ds_dict.items():
        ds_dict[k] = torch.stack(v, dim=0)

    state_max, state_min = ds_dict['state'].max(dim=0).values, ds_dict['state'].min(dim=0).values
    torch.save((state_max, state_min), f'train/octo/ours/{args.env_id}-state-max-min.pth')
    exit()
    
    def minmax_normalize(x, eps=1e-8):
        return (x - state_min) / (state_max - state_min + eps)
    if args.normalize_states:
        print('enable state normalization')
        ds_dict['state'] = minmax_normalize(ds_dict['state'])
    # n_samples = len(ds)
    

    env_kwargs['max_episode_steps'] = ds.max_episodes_length
    # if args.max_episode_steps is not None:
    #     env_kwargs["max_episode_steps"] = args.max_episode_steps

    print(env_kwargs)

    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        video_dir=f"ckpt/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
        # wrappers=[]
    )

    obs, _ = envs.reset(seed=args.seed)

    sampler = RandomSampler(ds)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    # camera_count = len(ds.camera_data.keys()) // 2 # each camera has rgb and depth
    camera_count = 1
    iter_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)

    data_loader = DataLoader(ds, batch_sampler=iter_sampler, num_workers=0)
    from train.octo.model import Actor
    actor = Actor(ds[0]['state'].size(0), envs.single_action_space.shape[0], camera_count, args.use_bn).to(
        device=device
    )
    if args.use_bn:
        print('add bn layers')

    if args.continue_train_from is None:
        assert args.pretrained_model != ''

        state_dict = torch.load(args.pretrained_model)['actor']
        # print('\n\nWARNING: remove state_encoder\'s pretrained weight because the architecture of state_encoder is changed\n\n')
        # state_dict = {n: p for n, p in state_dict.items() if 'state_encoder' not in n and 'decoder' not in n}
        actor.load_state_dict(state_dict, strict=False)

    # add FBS
    set_module(actor, 'rgb_encoder.fc.0', svd_decompose_linear(
        get_module(actor, 'rgb_encoder.fc.0')
    ))
    set_module(actor, 'depth_encoder.fc.0', svd_decompose_linear(
        get_module(actor, 'depth_encoder.fc.0')
    ))
    example_sample = {}
    for k, v in ds_dict.items():
        example_sample[k] = v[[0]]

    if not args.use_bn:
        add_FBS_into_cnn(
            actor,
            [f'rgb_encoder.cnn.{i}' for i in [0, 6, 12]] + [f'depth_encoder.cnn.{i}' for i in [0, 6, 12]],
            # [f'state_encoder.{i}' for i in [2]] + ['decoder.0'] + ['rgb_encoder.fc.0.0', 'depth_encoder.fc.0.0'],
            ['decoder.0'] + ['rgb_encoder.fc.0.0', 'depth_encoder.fc.0.0'],
            example_sample,
            0.9,
            8,
            lambda model, sample: model(sample['rgb'], sample['depth'], sample['state'])
        )
    else:
        add_FBS_into_cnn(
            actor,
            [f'rgb_encoder.cnn.{i}' for i in [0, 8, 16]] + [f'depth_encoder.cnn.{i}' for i in [0, 8, 16]],
            # [f'state_encoder.{i}' for i in [2]] + ['decoder.0'] + ['rgb_encoder.fc.0.0', 'depth_encoder.fc.0.0'],
            ['decoder.0'] + ['rgb_encoder.fc.0.0', 'depth_encoder.fc.0.0'],
            example_sample,
            0.9,
            8,
            lambda model, sample: model(sample['rgb'], sample['depth'], sample['state'])
        )

    if args.eval_model_only is not None:
        print('load ckpt for evaluation')
        # print(ds[0])
        actor.load_state_dict(torch.load(args.eval_model_only)['actor'])

    optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    best_eval_metrics = defaultdict(float)

    # print(actor)
    # exit()

    start_iter_idx = 0
    if args.continue_train_from is not None:
        ckpt = torch.load(args.continue_train_from)

        # print('\n\nWARNING: remove optimizer state because the architecture of state_encoder is changed\n\n')
        # del ckpt['optimizer']
        # del ckpt['scheduler']
        # del ckpt['iteration']
        

        if 'actor' in ckpt:
            print('load pretrained actor to continue training')

            # print('\n\nWARNING: remove state_encoder\'s pretrained weight because the architecture of state_encoder is changed\n\n')
            # ckpt['actor'] = {n: p for n, p in ckpt['actor'].items() if 'state_encoder' not in n and 'decoder' not in n}
            print(ckpt['actor'].keys())
            num_bns = 0
            for n, p in actor.rgb_encoder.named_parameters():
                m = get_module(actor.rgb_encoder, '.'.join(n.split('.')[0: -1]))

                if 'fc.' in n:
                    continue

                if isinstance(m, nn.BatchNorm2d) and n.endswith('.weight'):
                    num_bns += 1
                    continue
                
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    # print(n)
                    a = int(n.split('.')[1])

                    print('rgb_encoder.' + n, 'rgb_encoder.' + n.replace('cnn.' + str(a), 'cnn.' + str(a - num_bns)))
                    
                    ckpt['actor']['rgb_encoder.' + n] = ckpt['actor']['rgb_encoder.' + n.replace('cnn.' + str(a), 'cnn.' + str(a - num_bns))]
                    if num_bns > 0:
                        del ckpt['actor']['rgb_encoder.' + n.replace('cnn.' + str(a), 'cnn.' + str(a - num_bns))]
                    ckpt['actor']['depth_encoder.' + n] = ckpt['actor']['depth_encoder.' + n.replace('cnn.' + str(a), 'cnn.' + str(a - num_bns))]
                    if num_bns > 0:
                        del ckpt['actor']['depth_encoder.' + n.replace('cnn.' + str(a), 'cnn.' + str(a - num_bns))]

            print(actor.load_state_dict(ckpt['actor'], strict=False))
            optimizer = optim.Adam(actor.parameters(), lr=args.lr)
        if 'optimizer' in ckpt:
            print('load pretrained optimizer to continue training')

            # print('\n\nWARNING: remove state_encoder\'s pretrained weight because the architecture of state_encoder is changed\n\n')
            # def remove_optimizer_state_by_prefix(model, optimizer, prefix: str):
            #     """
            #     删除 optimizer 中，对应 model 参数名以 prefix 开头的 state
            #     """
            #     to_remove = []

            #     for name, param in model.named_parameters():
            #         if name.startswith(prefix):
            #             if param in optimizer.state:
            #                 to_remove.append(param)

            #     for param in to_remove:
            #         del optimizer.state[param]

            #     print(f"Removed optimizer state for {len(to_remove)} parameters with prefix '{prefix}'")

            # remove_optimizer_state_by_prefix(actor, optimizer, 'state_encoder')
            # remove_optimizer_state_by_prefix(actor, optimizer, 'decoder')

            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as e:
                print(f'WARNING: load optimizer error: {e}')
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        if 'scheduler' in ckpt:
            print('load pretrained scheduler to continue training')
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'iteration' in ckpt:
            print('load pretrained iteration to continue training')
            start_iter_idx = ckpt['iteration']

    # pbar = tqdm(enumerate(data_loader), total=args.total_iters)
    pbar = tqdm(range(start_iter_idx, args.total_iters), total=args.total_iters - start_iter_idx, initial=start_iter_idx, dynamic_ncols=True)

    # for iteration, batch in pbar:
    for iteration in pbar:

        from ours.libs.train_with_fbs.lib import set_sparsity
        if iteration % 4 == 0:
            cur_sparsity = 0.
        elif 1 <= iteration % 4 <= 2:
            cur_sparsity = random.random() * (0.9 - 0.) + 0.
        elif iteration % 4 == 3:
            cur_sparsity = 0.9
        set_sparsity(actor, cur_sparsity)

        samples_idx = torch.randperm(n_samples, device=device)[0: args.batch_size]
        batch = {}
        for k, v in ds_dict.items():
            # batch[k] = torch.stack([v[si] for si in samples_idx], dim=0)
            batch[k] = v[samples_idx]

        log_dict = {}

        optimizer.zero_grad()
        preds = actor(batch["rgb"], batch["depth"], batch["state"])
        loss = F.mse_loss(preds, batch["action"])
        loss.backward()

        if args.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_norm)

        optimizer.step()
        scheduler.step()

        pbar.set_description(f'Iter {iteration} | Loss: {loss:.6f}')

        if iteration % args.log_freq == 0:
            # print(f"Iteration {iteration}, loss: {loss.item()}")
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("losses/total_loss", loss.item(), iteration)
            # if args.track:
            #     wandb.log({'loss': loss.item()}, iteration)
            #     wandb.log({'lr': optimizer.param_groups[0]["lr"]}, iteration)

        if iteration % args.eval_freq == 0:
            save_ckpt(run_name, f"last")
            actor.debuged = False

            actor.eval()
            def sample_fn(obs):
                # state for panda_wristcam
        #         {'agent': {'qpos': tensor([[ 0.0061,  0.3453, -0.0167, -1.9487, -0.0195,  2.3135, -0.8176,  0.0400,
        #   0.0400]]), 'qvel': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.]])}, 'extra': {'is_grasped': tensor([False]), 'tcp_pose': tensor([[ 4.0905e-03, -8.5431e-03,  2.0151e-01,  1.3959e-02,  6.9430e-01,
        #   7.1955e-01, -5.4613e-04]]), 'goal_pos': tensor([[0.0303, 0.0163, 0.0578]]), 'obj_pose': tensor([[-0.0315,  0.0957,  0.0200,  0.8642,  0.0000,  0.0000, -0.5031]]), 'tcp_to_obj_pos': tensor([[-0.0356,  0.1042, -0.1815]]), 'obj_to_goal_pos': tensor([[ 0.0618, -0.0794,  0.0378]])}}
                # print(obs)
                # print({k: v.shape for k, v in obs.items()})
                # obs['rgbd'] = torch.concat([obs['rgb'], obs['depth']], axis=-1)
                # if isinstance(obs["rgbd"], np.ndarray):
                #     for k, v in obs.items():
                #         obs[k] = torch.from_numpy(v).float().to(device)
                # else:
                #     obs["rgbd"] = obs["rgbd"].float().to(device)

                # rgb = traj["obs"]["sensor_data"][cam]["rgb"][t]  # (H, W, 3), uint8
                rgb = torch.from_numpy(obs['rgb'] / 255.0).permute(0, 3, 1, 2)[:, 0: 3].float()

                # depth = traj["obs"]["sensor_data"][cam]["depth"][t]
                depth = torch.from_numpy(obs['depth'] / 1024.0).permute(0, 3, 1, 2)[:, 0: 1].float()

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
                depth = _resize(depth).to(device)

                if iteration == 0 and not actor.debuged:
                    from ours.utils.dl.common.vis import save_tensor_image
                    save_tensor_image(rgb, f'ckpt/{run_name}/sample-rgb.png')
                    save_tensor_image(depth, f'ckpt/{run_name}/sample-depth.png')
                    actor.debuged = True

                obs['state'] = torch.from_numpy(obs['state']).to(device)
                if args.normalize_states:
                    obs['state'] = minmax_normalize(obs['state'])

                action = actor(rgb, depth, obs["state"])
                if args.sim_backend == "cpu":
                    action = action.cpu().numpy()
                return action


            val_acc = 0.
            val_accs = {}
            import numpy as np

            eval_metrics_of_sparsities = {}

            sparsities = np.linspace(0, 0.9, 3).tolist()

            for sparsity in sparsities:
                set_sparsity(actor, sparsity)
                
                with torch.no_grad():
                    eval_metrics = evaluate(args.num_eval_episodes, sample_fn, envs)
                    for k in eval_metrics.keys():
                        eval_metrics[k] = np.mean(eval_metrics[k])
                eval_metrics_of_sparsities[sparsity] = eval_metrics
                # print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")

            
            actor.train()

            def get_eval_metric(k, sparsity):
                return eval_metrics_of_sparsities[sparsity][k]

            def get_eval_metric_averaged(k):
                res = [get_eval_metric(k, s) for s in sparsities]
                return sum(res) / len(res)
                
            for k in eval_metrics.keys():
                writer.add_scalars(f"eval/{k}", 
                                   {f'{s:.4f}': get_eval_metric(k, s) for s in sparsities}, 
                                   iteration)
                # if args.track:
                #     wandb.log({f"eval/{k}": {f'{s:.4f}': get_eval_metric(k, s) for s in sparsities}}, iteration)
                # print(f"{k}: {eval_metrics[k]:.4f}")

            if args.eval_model_only is not None:
                exit()

            save_on_best_metrics = ["success_once", "success_at_end", "reward"]
            for k in save_on_best_metrics:
                if k in eval_metrics and get_eval_metric_averaged(k) > best_eval_metrics[k]:
                    best_eval_metrics[k] = get_eval_metric_averaged(k)
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(
                        f"New best {k}_rate: {best_eval_metrics[k]:.4f}. Saving checkpoint."
                    )

        # if args.save_freq is not None and iteration % args.save_freq == 0:
        #     save_ckpt(run_name, str(iteration))
    envs.close()
    if args.track:
        wandb.finish()