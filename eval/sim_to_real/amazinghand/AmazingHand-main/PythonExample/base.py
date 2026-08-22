import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

# ==============================
# Kinematics (using pinocchio)
# ==============================
# pip install pin
import pinocchio as pin
import numpy as np
import xml.etree.ElementTree as ET
from AmazingHand_Demo import set_action, read_qpos
import cv2
from PIL import Image

class Kinematics:
    def __init__(self, urdf_path: str, ee_frame_name: str):
        # 1. 加载模型
        self.model = pin.buildModelsFromUrdf(urdf_path, geometry_types=[])[0]
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        
        # 2. 解析 URDF 获取 Mimic 关系
        # 我们需要知道：哪些关节是 Mimic，它们模仿谁，系数是多少
        self.mimic_map = self._parse_mimic_joints(urdf_path)
        
        # 3. 建立关节名称到 q 索引的映射
        # name_to_idx: {"joint_1": 0, "joint_2": 1, ...}
        self.name_to_idx = {}
        for i in range(self.model.njoints):
            idx = self.model.joints[i].idx_q
            if idx != -1: # 排除固定关节
                self.name_to_idx[self.model.names[i]] = idx

        print(f"模型加载完成:")
        print(f" - 总配置维度 (nq): {self.model.nq}")
        print(f" - 检测到 Mimic 关节数量: {len(self.mimic_map)}")

    def _parse_mimic_joints(self, urdf_path):
        """
        解析 URDF 文件，提取 mimic 关节的信息。
        返回格式: { "mimic_joint_name": {"master": "master_name", "multiplier": 1.0, "offset": 0.0} }
        """
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        mimic_map = {}
        for joint in root.findall('.//joint'):
            mimic_tag = joint.find('mimic')
            if mimic_tag is not None:
                joint_name = joint.get('name')
                master_name = mimic_tag.get('joint')
                multiplier = float(mimic_tag.get('multiplier', 1.0))
                offset = float(mimic_tag.get('offset', 0.0))
                
                mimic_map[joint_name] = {
                    "master": master_name,
                    "multiplier": multiplier,
                    "offset": offset
                }
        return mimic_map

    def fk(self, q_active: np.ndarray, goal_pos: np.ndarray, base_pose: np.ndarray):
        """
        正运动学计算。
        输入 q_active: 仅包含主动关节角度的数组 (例如长度为 6)。
        内部会自动补全 Mimic 关节角度并计算。
        """
        # 1. 如果输入的长度等于 model.nq，说明已经补全过了，直接计算
        if len(q_active) == self.model.nq:
            q_full = q_active
        else:
            # 2. 否则，假设输入的是主动关节，需要补全
            # 初始化全零向量
            q_full = np.zeros((q_active.shape[0], self.model.nq))
            
            # 我们需要追踪主动关节的索引，以便从 q_active 中取值
            # 注意：这里假设 q_active 的顺序与模型中主动关节出现的顺序一致
            # 如果你的 q_active 顺序是固定的（比如对应特定的电机），
            # 你可能需要根据 self.name_to_idx 来更严格地映射。
            
            # 为了安全起见，我们先填充所有已知的主动关节
            # 这里做一个简单的假设：q_active 按照 model 中 joint 的顺序依次填充非-mimic 关节
            active_input_idx = 0
            
            # 第一次遍历：填充所有非 Mimic 关节 (Master Joints)
            # 同时记录每个关节实际填入的角度，方便后面 Mimic 关节读取
            filled_angles = {} 
            
            for joint_name, q_idx in self.name_to_idx.items():
                if joint_name not in self.mimic_map:
                    # 这是一个主动关节
                    if active_input_idx < q_active.shape[-1]:
                        val = q_active[..., active_input_idx]
                        q_full[..., q_idx] = val
                        filled_angles[joint_name] = val
                        active_input_idx += 1
                    else:
                        # 输入数据不足，报错或处理异常
                        raise ValueError(f"输入的主动关节数量不足。期望至少 {len(self.name_to_idx) - len(self.mimic_map)} 个，实际得到 {len(q_active)} 个")
                else:
                    # 这是一个 Mimic 关节，暂时跳过，稍后填充
                    pass
            
            # 第二次遍历：根据 Mimic 关系填充 Mimic 关节
            for mimic_name, info in self.mimic_map.items():
                master_name = info["master"]
                multiplier = info["multiplier"]
                offset = info["offset"]
                
                if master_name in filled_angles:
                    master_val = filled_angles[master_name]
                    mimic_val = multiplier * master_val + offset
                    
                    # 找到 mimic 关节在 q_full 中的索引并赋值
                    if mimic_name in self.name_to_idx:
                        q_idx = self.name_to_idx[mimic_name]
                        q_full[..., q_idx] = mimic_val
                    else:
                        # 理论上不会发生，除非 URDF 解析不一致
                        pass 

        # 3. 调用 Pinocchio 计算
        pin.forwardKinematics(self.model, self.data, q_full[0])
        pin.updateFramePlacements(self.model, self.data)

        t_base = base_pose[:3]
        q_base = base_pose[3:]

        ee_pose = self.data.oMf[self.ee_frame_id]
        # print(ee_pose)
        # q_base: (w, x, y, z) → pinocchio (x, y, z, w)
        R_base = pin.Quaternion(
             q_base[0], q_base[1], q_base[2], q_base[3],
        ).toRotationMatrix()

        # ===== local tcp =====
        pos_local = ee_pose.translation
        R_local = ee_pose.rotation

        # ===== world tcp =====
        pos = R_base @ pos_local + t_base
        rot = R_base @ R_local

        # Pinocchio 的 Quaternion 默认是 (x, y, z, w)
        quat = pin.Quaternion(rot).coeffs()  
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        tcp_to_goal_pos = goal_pos - pos

        return np.concatenate([pos, quat_wxyz])[None, :], tcp_to_goal_pos[None, :]


# ==============================
# Velocity Estimator
# ==============================
class VelocityEstimator:
    def __init__(self, dof, dt):
        self.prev_qpos = None
        self.prev_time = None
        self.dt = dt

    def compute(self, qpos):
        if self.prev_qpos is None:
            self.prev_qpos = qpos.copy()
            return np.zeros_like(qpos)

        qvel = (qpos - self.prev_qpos) / (self.dt + 1e-8)
        self.prev_qpos = qpos.copy()

        return qvel

# ==============================
# Low-pass filter
# ==============================
class LowPassFilter:
    def __init__(self, alpha=0.2, dim=6):
        self.alpha = alpha
        self.state = np.zeros(dim)
        self.initialized = False

    def filter(self, x):
        if not self.initialized:
            self.state = x
            self.initialized = True
            return x

        self.state = self.alpha * x + (1 - self.alpha) * self.state
        return self.state


# ==============================
# Policy
# ==============================
class PlainConv(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_dim=256,
        max_pooling=True,
        inactivated_output=False,  # False for ConvBody, True for CNN
        use_bn=False
    ):
        super().__init__()

        if not use_bn:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [64, 64]
                nn.Conv2d(16, 16, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [32, 32]
                nn.Conv2d(16, 32, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [16, 16]
                nn.Conv2d(32, 64, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [8, 8]
                nn.Conv2d(64, 128, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [4, 4]
                nn.Conv2d(128, 128, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [64, 64]
                nn.Conv2d(16, 16, 3, padding=1, bias=True),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [32, 32]
                nn.Conv2d(16, 32, 3, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [16, 16]
                nn.Conv2d(32, 64, 3, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [8, 8]
                nn.Conv2d(64, 128, 3, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [4, 4]
                nn.Conv2d(128, 128, 1, padding=0, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )

        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=not inactivated_output)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4, [out_dim], last_act=not inactivated_output)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class F_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def encode(self, x):
        return torch.cat([
            x,
            torch.sin(x),
            torch.cos(x),
            torch.sin(2*x),
            torch.cos(2*x)
        ], dim=-1)

    def forward(self, x):
        x = self.encode(x)
        return self.net(x)


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

def make_mlp_with_orth_init(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True, is_actor=False):
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        if is_actor and idx == len(mlp_channels) - 1:
            module_list.append(layer_init(nn.Linear(c_in, c_out), std=0.01 * np.sqrt(2)))
        else:
            module_list.append(layer_init(nn.Linear(c_in, c_out)))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder(inplace=True))
        c_in = c_out
    return nn.Sequential(*module_list)

class RunningMeanStd(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(eps))
        self.eps = eps
        self.training_stats = True  # 区分 nn.Module.training

    @torch.no_grad()
    def update(self, x):
        if not self.training_stats:
            return

        x = x.float()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def forward(self, x, clip_range=None):
        x = (x - self.mean) / torch.sqrt(self.var + self.eps)
        if clip_range is not None:
            x = torch.clamp(x, -clip_range, clip_range)
        return x

    def freeze(self):
        self.training_stats = False

    def unfreeze(self):
        self.training_stats = True

class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, camera_count=1, normalize_state=True, use_depth=True):
        super().__init__()

        # 输入 (B, 3, 128, 128)，输出 (B, 256)
        self.rgb_encoder = PlainConv(
            in_channels=3 * camera_count, out_dim=256, max_pooling=False, inactivated_output=False
        )

        # 输入 (B, 1, 128, 128)，输出 (B, 256)
        if use_depth:
            self.depth_encoder = PlainConv(
                in_channels=1 * camera_count, out_dim=256, max_pooling=False, inactivated_output=False
            )
        else:
            self.depth_encoder = None

        # 输入 (B, state_dim)，输出 (B, state_dim)
        self.state_encoder = make_mlp(
            state_dim, [state_dim, 256], last_act=False
        )

        if use_depth:
            self.actor_mean = make_mlp_with_orth_init(
                256 * 3, [512, action_dim], last_act=False, is_actor=True
            )
            self.critic = make_mlp_with_orth_init(
                256 * 3, [512, 1], last_act=False
            )
        else:
            self.actor_mean = make_mlp_with_orth_init(
                256 * 2, [512, action_dim], last_act=False, is_actor=True
            )
            self.critic = make_mlp_with_orth_init(
                256 * 2, [512, 1], last_act=False
            )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

        if normalize_state:
            self.state_rms = RunningMeanStd(shape=(state_dim,))
        else:
            self.state_rms = nn.Sequential()

    def get_feature(self, batch):
        if self.depth_encoder is not None:
            rgb, depth, state = batch['rgb'], batch['depth'], batch['state']
            state = self.state_rms(state)
            rgb, depth, state = self.rgb_encoder(rgb), self.depth_encoder(depth), self.state_encoder(state)
            x = torch.cat([rgb, depth, state], dim=1)
        else:
            rgb, state = batch['rgb'], batch['state']
            state = self.state_rms(state)
            rgb, state = self.rgb_encoder(rgb), self.state_encoder(state)
            x = torch.cat([rgb, state], dim=1)
        return x

    def forward(self, batch, action=None):
        x = self.get_feature(batch)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        action = action if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=action_logstd.dtype, device=action_logstd.device)
        return (action.detach().cpu().numpy()), probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x).squeeze(-1)

    def get_action(self, batch, deterministic=True):
        x = self.get_feature(batch)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_value(self, batch):
        x = self.get_feature(batch)
        return self.critic(x).squeeze(-1)

    @torch.no_grad()
    def reset_logstd(self, new_logstd=-0.5):
        self.actor_logstd.data.fill_(new_logstd)

    def load_actor(self, path, strict=True):
        actor_state = torch.load(path, map_location="cpu")['actor']

        # 1. encoder
        self.rgb_encoder.load_state_dict(
            {k.replace("rgb_encoder.", ""): v
            for k, v in actor_state.items()
            if k.startswith("rgb_encoder.")},
            strict=strict
        )

        self.depth_encoder.load_state_dict(
            {k.replace("depth_encoder.", ""): v
            for k, v in actor_state.items()
            if k.startswith("depth_encoder.")},
            strict=strict
        )

        self.state_encoder.load_state_dict(
            {k.replace("state_encoder.", ""): v
            for k, v in actor_state.items()
            if k.startswith("state_encoder.")},
            strict=strict
        )

        # 2. decoder → actor_mean
        self.actor_mean.load_state_dict(
            {k.replace("decoder.", ""): v
            for k, v in actor_state.items()
            if k.startswith("decoder.")},
            strict=strict
        )

    @torch.no_grad()
    def update_state_stats(self, obs):
        if isinstance(self.state_rms, RunningMeanStd):
            self.state_rms.update(obs['state'])

    def freeze_state_stats(self):
        self.state_rms.freeze()

    def unfreeze_state_stats(self):
        self.state_rms.unfreeze()

# ==============================
# Utils
# ==============================

def deg2rad(q):
    return np.deg2rad(q)


def rad2deg(q):
    return np.rad2deg(q)

def crop_image(img):
    h, w = img.shape[1:3]
    crop_sz = min(h, w)
    start_x = (w - crop_sz) // 2
    start_y = (h - crop_sz) // 2
    
    cropped_img = img[:, start_y:start_y + crop_sz, start_x:start_x + crop_sz, :]

    return cropped_img


def preprocess_image(img):
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img, (0, 3, 1, 2)))  # HWC -> CHW
    return img

def flatten_state_dict(state_dict: dict):
    """Flatten a dictionary containing states recursively. Expects all data to be either torch or numpy

    Args:
        state_dict: a dictionary containing scalars or 1-dim vectors.
        use_torch (bool): Whether to convert the data to torch tensors.

    Raises:
        AssertionError: If a value of @state_dict is an ndarray with ndim > 2.

    Returns:
        np.ndarray | torch.Tensor: flattened states.

    Notes:
        The input is recommended to be ordered (e.g. dict).
        However, since python 3.7, dictionary order is guaranteed to be insertion order.
    """
    states = []

    for key, value in state_dict.items():
        if isinstance(value, dict):
            state = flatten_state_dict(value)
            if state.shape[0] == 0:
                state = None
        elif isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            state = int(value)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
        elif isinstance(value, np.ndarray):
            if value.ndim > 2:
                raise AssertionError(
                    "The dimension of {} should not be more than 2.".format(key)
                )
            state = value if value.size > 0 else None
        elif isinstance(value, torch.Tensor):
            state = value
            if len(state.shape) == 1:
                state = state[:, None]
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
        if state is not None:
            states.append(state)

    if len(states) == 0:
        return np.empty(0)
    else:
        return np.hstack(states)

def build_obs(image, roll, pitch, device, size=128):
    def _resize(img, size=size):
        # img = img.unsqueeze(0)          # (1, C, H, W)
        img = F.interpolate(
            img,
            size=size,
            mode='bilinear',
            # align_corners=align_corners if mode != "nearest" else None,
        )
        return img
    # 1. 预处理图像 (假设 preprocess_image 返回的是 numpy 数组)
    # 如果它返回的是 tensor，则不需要后面的 from_numpy 转换
    image = crop_image(image)
    rgb = preprocess_image(image)
    rgb = _resize(rgb).to(device)
    
    
    # 2. 拼接状态向量
    # 确保所有输入都是 float32 类型，防止精度不匹配报错
    # state_vector = np.concatenate([
    #     qpos.astype(np.float32), 
    #     qvel.astype(np.float32), 
    #     tcp_pose.astype(np.float32)
    # ], axis=-1)
    state_dict = dict(
        agent=dict(
            roll=roll,
            pitch=pitch
        ),
    )
    state_vector = flatten_state_dict(state_dict)
    state = torch.from_numpy(state_vector).to(device)
    state = state.to(dtype=torch.float32)

    # 3. 转换为 PyTorch Tensor 并移动到指定设备 (CPU/GPU)
    # 如果 rgb_np 已经是 tensor，可以直接 torch.as_tensor(rgb_np).to(device)
    obs_dict = {
        "rgb": rgb,
        "state": state,
    }
    
    return obs_dict

#    
# ==============================
# Main Loop
# ==============================
def make_dark(image, gamma=10, clip_limit=1.0, tile_size=(16, 16)):

    """

    统一函数：模拟关灯效果并抑制反光

    :param image: 输入的 BGR 彩色图像 (numpy array)

    :param gamma: 伽马值，>1 使图像变暗（模拟关灯）

    :param clip_limit: CLAHE的对比度限制参数，控制反光平滑程度

    :param tile_size: CLAHE的分块大小

    :return: 处理后的 BGR 图像

    """

    # 步骤1：伽马变换（模拟关灯，整体降暗）

    # 归一化到 0-1 之间

    img_normalized = image / 255.0

    # 幂运算并还原回 0-255

    darkened_img = np.power(img_normalized, gamma) * 255

    darkened_img = darkened_img.astype(np.uint8)


    # 步骤2：转换到 LAB 颜色空间，分离亮度通道（L）

    lab = cv2.cvtColor(darkened_img[0], cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)


    # 步骤3：应用 CLAHE 局部自适应直方图均衡（平滑光线，抑制反光）

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    l_eq = clahe.apply(l)


    # 步骤4：合并通道，转回 BGR 格式

    lab_eq = cv2.merge((l_eq, a, b))

    final_img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    final_img = final_img[None, ...]

    return final_img

import os
class Runner:
    def __init__(self, model_path, interface, freq=20, dof=6, device='cuda'):
        # TODO:添加接口
        # self.robot = RobotInterface()
        self.robot = interface
        self.policy = Agent(state_dim=8, action_dim=8, use_depth=False)
        self.policy.load_state_dict(torch.load(os.path.join(model_path, 'best_agent2.pt'), map_location=device))
        self.policy = self.policy.to(device)

        self.finger_models = []
        for i in range(1, 5):
            f_model = F_Model()
            f_model.load_state_dict(torch.load(os.path.join(model_path, f'finger{i}_f.pth'), map_location=device))
            f_model = f_model.to(device)
            self.finger_models.append(f_model)

        self.device = device
        # self.goal_pos = np.array(goal_pos) if isinstance(goal_pos, (list, tuple)) else goal_pos
        
        self.freq = freq
        self.dt = 1.0 / self.freq

        # self.vel_est = VelocityEstimator(dof=dof, dt=self.dt)
        # self.vel_filter = LowPassFilter(dim=dof, alpha=0.02)
        # self.action_filter = LowPassFilter(dim=dof, alpha=1)
        self.dof = dof

        

    def run(self, step=2):
        self.policy.eval()  # 切换到评估模式，关闭 dropout 和 batchnorm 的训练行为
        for i in range(step):
            start = time.time()

            # 1. Read sensors
            img = self.robot.get_image()
            # img = make_dark(img)
            Image.fromarray(crop_image(img)[0]).save('1.jpg')
            # 2. Process state
            roll, pitch = self.robot.get_joint_angles(self.finger_models, self.device)

            # t = time.time()
            # qvel = self.vel_est.compute(qpos_rad)
            # qvel = self.vel_filter.filter(qvel)
            # print("qpos:", qpos_rad)
            # print("qvel:", qvel)
            

            # tcp_pose, tcp_to_goal_pos = self.kin.fk(qpos_rad, self.goal_pos, self.robot.o_pose)
            # print("tcp:", tcp_pose)
            # 3. Build obs
            # goal_pos = self.goal_pos[None, :]
            obs = build_obs(img, roll, pitch, device=self.device)
            # actual_tcp_pose = self.robot.env.base_env.agent.tcp_pose.raw_pose.detach().cpu().numpy()
            # err = np.linalg.norm(actual_tcp_pose - tcp_pose)

            # 4. Inference
            with torch.no_grad():
                action = self.policy.get_action(obs)
            # action = self.action_filter.filter(action)

            # 5. Send
            self.robot.step(action)

            # 6. Control frequency
            # sleep dt 时间以让后续在读取数据时机器臂已经执行到位，避免数据时序错乱导致的性能下降
            time.sleep(self.dt * 1.1)


# ==============================
# Robot Interface (TO IMPLEMENT)
# ==============================
def bgr8_to_jpeg(value, quality=75):
    return bytes(cv2.imencode('.jpg', value)[1])

def clip_and_scale_action(action, low, high):
    """Clip action to [-1, 1] and scale according to a range [low, high]."""
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    action = np.clip(action, -1, 1)
    return 0.5 * (high + low) + 0.5 * (high - low) * action

class RobotInterface:
    def __init__(self, com, freq=20):
        time.sleep(.1)
        self.camera = cv2.VideoCapture(0,cv2.CAP_V4L2)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        current_focus = self.camera.get(cv2.CAP_PROP_FOCUS)
        print(f"当前对焦值: {current_focus}")

        self.prev_arm_action = np.zeros((1, 8))
        self.low = np.array([-.4, -.4, -.4, -.4, -.4, -.4, -.4, -.4])  # 这里的范围需要根据实际情况调整
        self.high = np.array([.4, .4, .4, .4, .4, .4, .4, .4])
        # self.deg_low = np.array([0, 0, 0, 0, 0, 0])
        # self.deg_high = np.array([180, 180, 180, 180, 270, 135])
        # self.dt = 1 / freq * 1e3
        # self.o_pose = np.array(o_pose)

        start_qpos:np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        set_action(start_qpos)
        time.sleep(1)

        img = self.get_image()
        img = crop_image(img)
        Image.fromarray(img[0]).save('1.jpg')

    def get_image(self):
        # 丢弃缓存图像
        ret, frame = self.camera.read()

        ret, frame = self.camera.read()     #读取摄像头数据
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return img[None, :]

    def get_joint_angles(self, f_models, device):
        raw = read_qpos()
        roll = []
        pitch = []
        for i in range(4):
            x = raw[:, 2*i:2*(i+1)].to(device)
            with torch.no_grad():
                y = f_models[i](x)
            roll.append(y[:, 0])
            pitch.append(y[:, 1])
        roll = torch.cat(roll, dim=-1).unsqueeze(0)
        pitch = torch.cat(pitch, dim=-1).unsqueeze(0)
        return roll.cpu().numpy(), pitch.cpu().numpy()  # degrees

    def step(self, action):
        action = clip_and_scale_action(action, self.low, self.high)
        action += self.prev_arm_action
        action = np.clip(action, -2, 2)
        self.prev_arm_action = action.copy()
        print(self.prev_arm_action)

        set_action(action)

# from dataclasses import dataclass
# from typing import Annotated, Optional
# import tyro
# import gymnasium as gym
# import os
# import sys
# sys.path.append(os.getcwd())
# import envs.agents.dofbot_se
# @dataclass
# class Args:
#     robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "dofbot_se"
#     sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "cpu"
#     control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
#     keyframe: Annotated[Optional[str], tyro.conf.arg(aliases=["-k"])] = None
#     shader: str = "default"
#     keyframe_actions: bool = False
#     random_actions: bool = False
#     none_actions: bool = False
#     zero_actions: bool = True
#     sim_freq: int = 100
#     control_freq: int = 20
#     seed: Annotated[Optional[int], tyro.conf.arg(aliases=["-s"])] = None

# class TestInterface:
#     def __init__(self, env):
#         self.env = env
#         self.image = None
#         self.qpos = None

#     def step(self, action):
#         if self.image is None:
#             obs, _ = self.env.reset()
#         else:
#             obs, _, _, _, _ = self.env.step(action)
#         self.image = obs['sensor_data']['hand_camera']['rgb']
#         self.qpos = obs['agent']['qpos']
        
    
#     def get_image(self):
#         return self.image.detach().cpu().numpy()
    
#     def get_joint_angles(self):
#         return self.qpos[..., :6].detach().cpu().numpy()

# def test(args: Args):
#     URDF_PATH = str(Path(__file__).parent.resolve() / "urdf/dofbot.urdf")
#     EE_LINK = "gripper_tcp"  # !!! replace with your actual end-effector link name
#     MODEL_PATH = str(Path(__file__).parent.resolve() / "model/best_agent.pt")
#     # =========================
#     # 1. 创建原始环境
#     # =========================
#     env_settings = dict(
#         object_type="cube",                                     # "cube", "sphere", "cylinder", "box", 或者 None 表示随机选择
#         object_size_info={                                      # 一个字典，预设物体尺寸参数 (m)，设置为 {} 表示随机选择
#             'cube': {'half_size': 0.015},                        # cube 的边长为 0.06m
#         },                                      
#         object_mass=0.1,                                        # 物体质量 (kg)，None 表示随机选择
#         object_color=[1, 0, 0, 1],                              # 物体颜色，输入一个 RGBA 列表，None 表示随机选择，各维度 0 ~ 1 之间
#         randomize_camera=False,                                 # 是否随机摄像头位置, None 表示部分随机
#     )
#     base_env = gym.make(
#         # "MyEmpty-v1",
#         "Empty-v1",
#         # "PickObjectRandomDofbot-v1",
#         obs_mode="rgb+state_dict",
#         reward_mode="none",
#         enable_shadow=True,
#         control_mode=args.control_mode,
#         robot_uids=args.robot_uid,
#         sensor_configs=dict(shader_pack=args.shader),
#         human_render_camera_configs=dict(shader_pack=args.shader),
#         viewer_camera_configs=dict(shader_pack=args.shader),
#         render_mode="rgb_array",  # ⚠️ 录视频必须是 rgb_array
#         sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
#         sim_backend=args.sim_backend,
#         max_episode_steps=100,
#         # **env_settings
#     )
#     # qpos = base_env.agent.robot.get_qpos()[..., :-6]
#     interface = TestInterface(base_env)
#     goal_pos = np.array([0.1, 0.1, 0.1])
#     runner = Runner(URDF_PATH, EE_LINK, MODEL_PATH, interface, goal_pos, device='cuda')
#     runner.run()
    

# ==============================
# Entry
# ==============================
# if __name__ == "__main__":
    # test(tyro.cli(Args))

    