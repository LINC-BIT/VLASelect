import os
import sys
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datetime import datetime
import argparse
import time
from tqdm import tqdm
import torch
from accelerate import Accelerator
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from mani_skill.utils.io_utils import load_json, dump_json
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import gymnasium as gym
import random

from train.octo.model import Agent, make_mlp_with_orth_init
from train.reinforcement_learning.utils import collect_rollout, compute_gae, ppo_update_on_policy, get_step_infos, get_env_settings_by_ddl
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.evaluate import evaluate
import envs.pick_obj_random

# =====================================================
# Preprocess
# =====================================================
model_name = 'octo'
robot_name = 'panda_wristcam'
sensor_configs = {
    "shader_pack": "default",
    "width": 128,
    "height": 128
}
env_settings = dict(
    object_type="cube",                                     # "cube", "sphere", "cylinder", "box", 或者 None 表示随机选择
    object_size_info={                                      # 一个字典，预设物体尺寸参数 (m)，设置为 {} 表示随机选择
        'cube': {'half_size': 0.02},                        # cube 的边长为 0.06m
        'sphere': {'radius': 0.03},                         # sphere 的半径为 0.03m
        'cylinder': {'radius': 0.03, 'half_length': 0.03},  # cylinder 的半径为 0.03m，长度为 0.06m
        'box': {'half_sizes': [0.03, 0.04, 0.05]},          # box 的长宽高分别为 0.06m, 0.08m, 0.1m
    },                                      
    object_mass=0.1,                                        # 物体质量 (kg)，None 表示随机选择
    object_color=[1, 0, 0, 1],                              # 物体颜色，输入一个 RGBA 列表，None 表示随机选择，各维度 0 ~ 1 之间
    randomize_camera=False,                                 # 是否随机摄像头位置, None 表示部分随机
    ambient_light_temperature=2000,                         # 环境光色温，None 表示随机选择，建议范围 2000K~6500K
    ambient_light_intensity=1.0,                            # 环境光强度，None 表示随机选择，建议范围 0.1 ~ 1.0
    directional_light_temperature=2000,                     # 方向光色温，None 表示随机选择，建议范围 2000K~6500K
    directional_light_direction=[1, 1, -1],                 # 方向光方向，输入一个 3 维列表，None 表示随机选择, 建议各维度 0 ~ 0.5 之间
    shadow_scale=5,                                         # 阴影长度，None 表示随机选择, 建议 5 ~ 10
)
cameras=("base_camera",)

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
        "actor_path": actor_path
    }

def make_collate_fn(args, cameras, device):
    # device 指将obs移动到什么device, 不是obs是什么device
    def collate_fn(obs):
        if len(cameras) == 2:
            idx = (0, 2)
        elif 'base_camera' in cameras:
            idx = (0, 1)
        elif 'hand_camera' in cameras:
            idx = (1, 2)
        else:
            raise NotImplementedError(f'不支持摄像头：{cameras}')

        if isinstance(obs['rgb'], np.ndarray):
            rgb = torch.from_numpy(obs['rgb'] / 255.0).permute(0, 3, 1, 2)[:, idx[0]*3: idx[1]*3].float()
            depth = torch.from_numpy(obs['depth'] / 1024.0).permute(0, 3, 1, 2)[:, idx[0]: idx[1]].float()
            state = torch.from_numpy(obs["state"])
        else:
            rgb = obs["rgb"].permute(0, 3, 1, 2)[:, idx[0]*3: idx[1]*3].float() / 255.0  # (env, H, W, 3)
            depth = obs["depth"].permute(0, 3, 1, 2)[:, idx[0]: idx[1]].float() / 1024.0  # (env, H, W, 1)
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
        depth = _resize(depth).to(device)
        state = state.to(device)

        return {
            'rgb': rgb,
            'depth': depth,
            'state': state,
        }

    return collate_fn

def make_sample_fn(args, agent_model, cameras, accelerator):
    device = accelerator.device
    # device 指将obs移动到什么device, 不是obs是什么device
    def sample_fn(obs):
        if len(cameras) == 2:
            idx = (0, 2)
        elif 'base_camera' in cameras:
            idx = (0, 1)
        elif 'hand_camera' in cameras:
            idx = (1, 2)
        else:
            raise NotImplementedError(f'不支持摄像头：{cameras}')

        if isinstance(obs['rgb'], np.ndarray):
            rgb = torch.from_numpy(obs['rgb'] / 255.0).permute(0, 3, 1, 2)[:, idx[0]*3: idx[1]*3].float()
            depth = torch.from_numpy(obs['depth'] / 1024.0).permute(0, 3, 1, 2)[:, idx[0]: idx[1]].float()
            state = torch.from_numpy(obs["state"])
        else:
            rgb = obs["rgb"].permute(0, 3, 1, 2)[:, idx[0]*3: idx[1]*3].float() / 255.0  # (env, H, W, 3)
            depth = obs["depth"].permute(0, 3, 1, 2)[:, idx[0]: idx[1]].float() / 1024.0  # (env, H, W, 1)
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
        depth = _resize(depth).to(device)
        state = state.to(device)

        batch = {
            'rgb': rgb,
            'depth': depth,
            'state': state,
        }

        action = agent_model.get_action(batch)
        return action

    return sample_fn

def get_agent_info(args, env_kwargs, collate_fn):
    test_env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend='gpu',
        env_kwargs=env_kwargs,
        wrappers=[FlattenRGBDObservationWrapper]
    )
    obs, _ = test_env.reset()
    batch = collate_fn(obs)
    infos = dict(state_dim=batch['state'].shape[-1], action_dim=test_env.single_action_space.shape[0])
    return infos

# =====================================================
# Main
# =====================================================
def on_ddl_change(args, agent, ddl, device='cpu'):
    # entropy schedule
    ent_table = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if args.ent_coef == 0:
        args.ent_coef = ent_table[ddl]
    
    with torch.no_grad():
        # ---- restore exploration radius (Actor std) ----
        if ddl >= 3:
            agent.reset_logstd(-0.5)  # std ≈ exp(-0.5) ≈ 0.6

        # ---- reset critic head only (not shared feature encoder) ----
        if ddl >= 3:
            agent.critic = make_mlp_with_orth_init(
                256 * 3, [512, 1], last_act=False
            ).to(device)

def main(args):
    ckpt = resolve_ckpt_dir(args, model_name)
    step_infos = get_step_infos(args)
    env_kwargs = {
      "obs_mode": "rgb+depth+state_dict",
      "control_mode": "pd_joint_delta_pos",
      "render_mode": "rgb_array",
      "reward_mode": "normalized_dense",
      "shader_dir": "default",
      "sim_backend": "physx_cuda",
      "robot_uids": robot_name,
      "sensor_configs": sensor_configs,
      "max_episode_steps": args.max_episode_steps,
      "global_random": True,
    }
    env_kwargs_for_eval = env_kwargs
    env_kwargs_for_eval.pop('sim_backend')

    # PickObjectRandom-v1 设置
    # 若某项设置为随机，但是只想在开始时随机一次，可以添加 reconfiguration_freq=0 参数（或不添加）
    # 若想每次重置时都随机，可以添加 reconfiguration_freq=1 参数，仅限eval阶段 (Maniskill不支持在train阶段更换环境)
    # 若不添加reconfiguration_freq=1，reset时仅随机位置参数，不随机尺寸、质量、颜色等参数
    # 色温是越低越偏红，越高越偏蓝
    
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
    agent = Agent(**infos, normalize_state=args.normalize_state)
    if os.path.exists(ckpt["latest_agent"]):
        print(f"[Train] Resume agent from {ckpt['latest_agent']}")
        agent_dict = torch.load(ckpt["latest_agent"], map_location="cpu")
        agent.load_state_dict(agent_dict)
    elif os.path.exists(ckpt["actor_path"]):
        print(f"[Train] load actor from {ckpt['actor_path']}")
        agent.load_actor(ckpt['actor_path'])
    elif args.agent_dir is not None: 
        print(f"[Train] load agent from {args.agent_dir}")
        agent.load_state_dict(torch.load(os.path.join(args.agent_dir, "best_agent.pt"), map_location="cpu"))
        agent.reset_logstd(-0.5)
        agent.critic = make_mlp_with_orth_init(
            256 * 3, [512, 1], last_act=False
        ).to(device)

    num_rollouts = env_stage = sta_steps = global_steps = 0
    rollouts_num_aft_env_change = ddl = -1
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    start_time = time.time()
    if os.path.exists(ckpt["latest_opt"]):
        print(f"[Train] Resume optimizer from {ckpt['latest_opt']}")
        resume_opt = torch.load(ckpt["latest_opt"], map_location="cpu")
        optimizer.load_state_dict(resume_opt['opt'])
        sta_steps = global_steps = resume_opt['step']
        ddl = resume_opt['ddl']
        rollouts_num_aft_env_change = resume_opt['rollouts_num_aft_env_change']
        num_rollouts = resume_opt['num_rollouts']
        env_stage = resume_opt['env_stage']

    agent, optimizer = accelerator.prepare(agent, optimizer)

    if args.reset_logstd:
        print(f"[Train] Reset actor log std to -0.5")
        agent.reset_logstd(-0.5)

    best_score = -1
    metrics_log = []
    pbar = tqdm(total=step_infos['total_steps'], initial=global_steps, ascii=True)
    writer = SummaryWriter(ckpt["log_dir"], purge_step=global_steps)
    print(f"[TensorBoard] Logging to {ckpt['log_dir']}")
    print(f"[TensorBoard] Using `tensorboard --logdir {ckpt['log_dir']} --port {tensorboard_port}` to show the logs")

    if args.evaluate_mode:
        if args.agent_dir is None:
            raise ValueError("Please provide --agent-dir for evaluation mode")
        
        eval_envs = make_eval_envs(
            env_id=args.task_name,
            num_envs=args.num_eval_envs,
            sim_backend='gpu',
            env_kwargs=env_kwargs_for_eval,
            video_dir=f'{ckpt["video_dir"]}_train' if not args.evaluate_mode else f'{ckpt["video_dir"]}_eval',
            wrappers=[FlattenRGBDObservationWrapper],
        )
        
        print("[Evaluate] Start evaluation only mode")
        agent.eval()
        eval_metrics = evaluate(
            n=100,
            sample_fn=make_sample_fn(args, agent, cameras, accelerator),
            eval_envs=eval_envs,
        )
        for k, v in eval_metrics.items():
            mean = v.mean()
            print(f"eval_{k}_mean={mean}")
        dump_json(os.path.join(ckpt['root_dir'], 'eval_metrics.json'), {k: float(v.mean()) for k, v in eval_metrics.items()})
        return

    envs = None
    next_done = None
    next_obs = None

    if os.path.exists(ckpt["metrics"]):
        metrics_log = load_json(ckpt["metrics"])
    resume_skip = True if args.resume_dir is not None else False

    if args.minibatch_size == 0:
        args.minibatch_size = step_infos['rollot_steps'] // args.num_minibatch // args.grad_accum_steps
    
    change_env_freq = 30
    rollouts_num_aft_env_change_lim = 10
    while global_steps < step_infos['total_steps']:
        if num_rollouts == change_env_freq or envs is None:
            now_env_settings, new_ddl = get_env_settings_by_ddl(args, global_steps, env_settings)
            if ddl != new_ddl:
                on_ddl_change(args, agent, ddl, device)
                rollouts_num_aft_env_change = 0
            ddl = new_ddl
            if ddl > 0 or envs is None:
                if envs is not None:
                    envs.close()
                    del envs
                now_env_kwargs = env_kwargs.copy()
                now_env_kwargs.update(**now_env_settings)
                envs = gym.make(args.task_name, num_envs=args.num_envs, restart_id=env_stage, **now_env_kwargs)
                envs = FlattenRGBDObservationWrapper(envs)
                envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=args.ignore_partial_reset, record_metrics=True)
                next_obs, _ = envs.reset(seed=args.seed)
                next_done = torch.zeros(args.num_envs, device=device)
                env_stage += 1
            num_rollouts = 0

        if args.change_env_dynamic:
            if ddl <= 1:
                change_env_freq = 10
                rollouts_num_aft_env_change_lim = 10
            elif ddl <= 3:
                change_env_freq = 10
                rollouts_num_aft_env_change_lim = 25
            else:
                change_env_freq = 25
                rollouts_num_aft_env_change_lim = 100

        agent.eval()
        if not resume_skip and global_steps % (step_infos['save_interval_steps']) == 0:
            torch.save(agent.state_dict(), ckpt['latest_agent'])
            torch.save({'opt': optimizer.state_dict(), 'step': global_steps, 'ddl':ddl, 'num_rollouts':num_rollouts, 'env_stage':env_stage}, ckpt["latest_opt"])
            
            eval_envs = make_eval_envs(
                env_id=args.task_name,
                num_envs=args.num_eval_envs,
                sim_backend='gpu',
                env_kwargs=env_kwargs_for_eval,
                video_dir=f'{ckpt["video_dir"]}_{global_steps}',
                wrappers=[FlattenRGBDObservationWrapper],
            )
            _, _ = eval_envs.reset(seed=args.seed)

            # ---------------- evaluate ----------------
            eval_metrics = evaluate(
                n=25 * args.num_eval_envs,
                sample_fn=make_sample_fn(args, agent, cameras, accelerator),
                eval_envs=eval_envs,
            )
            
            eval_envs.close()
            del eval_envs

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

            if score > best_score:
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

        num_rollouts += 1

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
        stats = ppo_update_on_policy(args, agent, optimizer, data, collate_fn, accelerator, f'Updating Env {env_stage} (Gap level={ddl})', writer, rollouts_num_aft_env_change)
        
        if rollouts_num_aft_env_change != -1:
            rollouts_num_aft_env_change += 1
        if rollouts_num_aft_env_change == rollouts_num_aft_env_change_lim:
            rollouts_num_aft_env_change = -1

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
    envs.close()
    eval_envs.close()
    torch.save(agent.state_dict(), ckpt['latest_agent'])
    torch.save({'opt': optimizer.state_dict(), 'step': global_steps, 'ddl':ddl, 'num_rollouts':num_rollouts, 'env_stage':env_stage}, ckpt["latest_opt"])

# batchsize=num_envs * rollout_steps
# minibatch_size = batchsize // num_minibatch
# 若envs中存在好样本，minibatch_size应该足够大以覆盖好样本，但过大会稀释好样本的影响
# 若资源不够，可以采用grad_accum_steps来累积梯度，相当于增大了minibatch_size
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--actor-ckpt-path", type=str, default='ckpt/PickCube-v1/ours/octo/pretrain_large_model/20260121-092802/checkpoints')
    parser.add_argument("--actor-ckpt-path", type=str, default=None)
    parser.add_argument("--task-name", type=str, default="PickObjectRandom-v1")
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    parser.add_argument("--critic-warmup-rollouts", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=384)
    parser.add_argument("--num-eval-envs", type=int, default=8)
    parser.add_argument("--ignore-partial-reset", action="store_true")
    parser.add_argument("--ignore-torch-deterministic", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--num_minibatch", type=int, default=16)
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
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--gae-lambda", type=float, default=0.9)
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.25)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--save-dir", type=str, default="ckpt")
    parser.add_argument("--resume-dir", type=str, default=None)
    
    parser.add_argument("--save-interval-per-rollout", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--evaluate-mode", action="store_true")
    parser.add_argument("--agent-dir", type=str, default=None)

    parser.add_argument("--change-env-dynamic", action="store_true") # rollouts number between changing env settings
    
    parser.add_argument("--robot-name", type=str, default="panda")
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main(parse_args())
