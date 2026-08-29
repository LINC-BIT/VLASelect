import os
import gymnasium as gym
from datetime import datetime
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim

# Domain Shift Adjustment
def get_env_settings_by_ddl(args, global_steps, env_settings):
    progress = global_steps / args.total_steps

    ddl = 0

    if progress > 0.02:
        ddl = 1
        env_settings["ambient_light_temperature"] = None
        env_settings["ambient_light_intensity"] = None
        env_settings["directional_light_temperature"] = None
        env_settings["shadow_scale"] = None

    if progress > 0.04:
        ddl = 2
        env_settings["object_color"] = None

    if progress > 0.06:
        ddl = 3
        env_settings["object_size_info"] = {}
        env_settings["object_mass"] = None

    if progress > 0.08:
        ddl = 4
        env_settings["randomize_camera"] = True

    if progress > 0.20:
        ddl = 5
        env_settings["object_type"] = None

    return env_settings, ddl


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
    latest_actor_lora = os.path.join(resume_dir, "latest_actor.pt")
    latest_critic_lora = os.path.join(resume_dir, "latest_critic.pt")
    latest_critic_head = os.path.join(resume_dir, "latest_critic_head.pt")
    latest_lora = os.path.join(resume_dir, "latest_lora.pt")
    latest_opt = os.path.join(resume_dir, "latest_opt.pt")
    load_lora = os.path.join(args.ft_lora, "latest_lora.pt")

    norm_stats = os.path.join(args.ft_lora if args.resume_dir is None else resume_dir, "norm_stats.json") 

    return {
        "task_dir": task_dir,
        "log_dir": log_dir,
        "root_dir": root_dir,
        "video_dir": video_dir,
        "latest_actor_lora": latest_actor_lora,
        "latest_critic_lora": latest_critic_lora,
        "latest_critic_head": latest_critic_head,
        "latest_lora": latest_lora,
        "latest_opt": latest_opt,
        "load_lora": load_lora,
        "best_actor_lora": os.path.join(resume_dir, "best_actor_lora.pt"),
        "best_critic_lora": os.path.join(resume_dir, "best_critic_lora.pt"),
        "best_critic_head": os.path.join(resume_dir, "best_critic_head.pt"),
        "best_lora": os.path.join(resume_dir, "best_lora.pt"),
        "metrics": os.path.join(resume_dir, "metrics.json"),
        "norm_stats": norm_stats
    }
# =====================================================
# Learning Rate
# =====================================================
def get_init_lr(lr):
    init_lr = {
        "value_head": lr,
        "lm_head": lr / 3,
        "lora": lr / 10
    }
    return init_lr

def get_step_infos(args):
    rollot_steps = args.num_envs * args.rollout_steps

    critic_warmup_steps = rollot_steps * args.critic_warmup_rollouts
    stage_2 = stage_1 = (args.total_steps - critic_warmup_steps) // 2
    save_interval_steps = args.save_interval_per_rollout * rollot_steps

    return {
        "rollot_steps": rollot_steps,
        "total_steps": args.total_steps,
        "critic_warmup_steps": critic_warmup_steps,
        "stage_1": stage_1,
        "stage_2": stage_2,
        "save_interval_steps": save_interval_steps,
    }

def adjust_lr(optimizer, global_step, step_infos, init_lr, kl=None, mode="linear"):
    """
    动态调整学习率
    - mode="linear": 线性衰减
    - mode="kl": KL 自适应衰减 lm_head/lora lr
    """
    if global_step < step_infos["critic_warmup_steps"]:
        # 阶段1：critic预训练
        fraction = 1.0
        for group in optimizer.param_groups:
            if "value_head" in group["name"]:
                group["lr"] = init_lr["value_head"]
            else:
                group["lr"] = 0.0  # lm_head / LoRA 不动

    elif global_step < step_infos["critic_warmup_steps"] + step_infos["stage_1"]:
        # 阶段2：PPO初期
        step_in_phase2 = global_step - step_infos["critic_warmup_steps"]
        fraction = 1.0 - step_in_phase2 / step_infos["stage_1"]
        for group in optimizer.param_groups:
            if "value_head" in group["name"]:
                group["lr"] = init_lr["value_head"]  # 可固定
            elif "lm_head" in group["name"]:
                group["lr"] = init_lr["lm_head"] * fraction
            elif "lora" in group["name"]:
                group["lr"] = init_lr["lora"] * fraction

    else:
        # 阶段3：PPO后期
        step_in_phase3 = global_step - (step_infos["critic_warmup_steps"] + step_infos["stage_1"])
        fraction = 1.0 - step_in_phase3 / step_infos["stage_2"]
        for group in optimizer.param_groups:
            if "value_head" in group["name"]:
                group["lr"] = init_lr["value_head"] * fraction
            elif "lm_head" in group["name"]:
                group["lr"] = init_lr["lm_head"] * fraction
            elif "lora" in group["name"]:
                group["lr"] = init_lr["lora"] * fraction

    # 可选：KL 自适应调整 lm_head / LoRA
    if mode == "kl" and kl is not None:
        kl_target = 0.01
        scale = max(0.1, min(1.0, kl_target / (kl + 1e-8)))
        for group in optimizer.param_groups:
            if "lm_head" in group["name"] or "lora" in group["name"]:
                group["lr"] *= scale

def get_stage(global_step, step_infos):
    """
    动态调整学习率
    - mode="linear": 线性衰减
    - mode="kl": KL 自适应衰减 lm_head/lora lr
    """
    if global_step < step_infos["critic_warmup_steps"]:
        # 阶段1：critic预训练
        return "Critic Training Stage"
    elif global_step < step_infos["critic_warmup_steps"] + step_infos["stage_1"]:
        # 阶段2：PPO初期
        return "PPO Stage 1"
    else:
        # 阶段3：PPO后期
        return "PPO Stage 2"

def get_optimizer(agent, init_lr):
    lora_params = [p for n, p in agent.base_vla.named_parameters()
               if "lora" in n.lower() and p.requires_grad]
    optimizer = optim.AdamW([
        {"params": agent.critic.value_head.parameters(), "lr": init_lr['value_head'], "name": "value_head"},
        {"params": agent.base_vla.get_output_embeddings().parameters(), "lr": init_lr['lm_head'], "name": "lm_head"},
        {"params": lora_params, "lr": init_lr['lora'], "name": "lora"}
    ])
    return optimizer

# =====================================================
# GAE
# =====================================================
def compute_gae(
    rewards,              # (T, N)
    dones,                # (T, N)  -> next_done (float 0/1)
    values,               # (T, N)
    final_values,         # (T, N)  -> 每一步“final value”，即 done 时的 bootstrap value
    next_obs,          # (T, N)  -> 每一步“真实 next value”
    next_done,        # (N,)    -> 最后一步的 next_done
    agent,
    collate_fn,
    args,
    accelerator,
):
    """
    rewards[t]        = r_t
    dones[t]          = next_done_t
    values[t]         = V(s_t)
    next_values[t]    = V(s_{t+1}) or V(final_obs) if reset
    """

    with torch.no_grad():
        batch = collate_fn(next_obs)
        with accelerator.autocast():
            next_value = agent.get_value(batch).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(accelerator.device)
        lastgaelam = 0
        for t in reversed(range(args.rollout_steps)):
            if t == args.rollout_steps - 1:
                next_not_done = 1.0 - next_done
                nextvalues = next_value
            else:
                next_not_done = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
            # next_not_done means nextvalues is computed from the correct next_obs
            # if next_not_done is 1, final_values is always 0
            # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
            if args.finite_horizon_gae:
                """
                See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                """
                if t == args.rollout_steps - 1: # initialize
                    lam_coef_sum = 0.
                    reward_term_sum = 0. # the sum of the second term
                    value_term_sum = 0. # the sum of the third term
                lam_coef_sum = lam_coef_sum * next_not_done
                reward_term_sum = reward_term_sum * next_not_done
                value_term_sum = value_term_sum * next_not_done

                lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
            else:
                delta = rewards[t] + args.gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
        returns = advantages + values

    return advantages, returns

# =====================================================
# Rollout
# =====================================================
class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.bool if v.dtype == np.bool_ else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            torch.int64 if v.dtype == np.int64 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)
    
def agent_forward_microbatch_for_rollout(args, agent, batch, accelerator, only_value=False, return_action_bins=False):
    B = next(iter(batch.values())).shape[0]

    actions = []
    logps = []
    values = []
    action_bins = []
    max_bs = args.rollout_minibatch_size
    device = accelerator.device

    for i in range(0, B, max_bs):
        sub_batch = {
            k: v[i:i + max_bs].to(device)
            for k, v in batch.items()
        }

        if only_value:
            with torch.no_grad():
                with accelerator.autocast():
                    value = agent.get_value(sub_batch)
            actions.append(None)
            logps.append(None)
            values.append(value.detach().cpu())
            action_bins.append(None)
            continue

        with torch.no_grad():  # rollout 阶段一般不开梯度
            with accelerator.autocast():
                if hasattr(agent, "get_action_and_value"):
                    if return_action_bins:
                        action, logp, _, value, bins = agent.get_action_and_value(
                            sub_batch, return_action_bins=True
                        )
                    else:
                        action, logp, _, value = agent.get_action_and_value(sub_batch)
                        bins = None
                else:
                    action, logp, _, value = agent(sub_batch)
                    bins = None
        actions.append(action)
        if isinstance(logp, dict):
            logps.append({k: v.detach().cpu() for k, v in logp.items()})
        else:
            logps.append(logp.detach().cpu())
        values.append(value.detach().cpu())
        action_bins.append(bins)

    if isinstance(actions[0], dict):
        merged_actions = {k: np.concatenate([chunk[k] for chunk in actions], axis=0) for k in actions[0].keys()}
    else:
        merged_actions = np.concatenate(actions, axis=0)

    if isinstance(logps[0], dict):
        merged_logps = {k: torch.cat([chunk[k] for chunk in logps], dim=0) for k in logps[0].keys()}
    else:
        merged_logps = torch.cat(logps, dim=0)

    if return_action_bins:
        merged_bins = {k: torch.cat([chunk[k].detach().cpu() for chunk in action_bins], dim=0) for k in action_bins[0].keys()}
        return (
            merged_actions,
            merged_logps,
            torch.cat(values, dim=0),
            merged_bins,
        )

    return (
        merged_actions,
        merged_logps,
        torch.cat(values, dim=0),
    )

def collect_rollout(args, agent, collate_fn, envs, next_obs, next_done, accelerator, writer, global_step):
    obs = DictArray((args.rollout_steps, args.num_envs), envs.single_observation_space, device=accelerator.device)
    actions = np.zeros((args.rollout_steps, args.num_envs) + envs.single_action_space.shape)
    logprobs = torch.zeros((args.rollout_steps, args.num_envs)).to(accelerator.device)
    rewards = torch.zeros((args.rollout_steps, args.num_envs)).to(accelerator.device)
    dones = torch.zeros((args.rollout_steps, args.num_envs)).to(accelerator.device)
    values = torch.zeros((args.rollout_steps, args.num_envs)).to(accelerator.device)  # 👈 关键：每一步的 real next value
    final_values = torch.zeros((args.rollout_steps, args.num_envs), device=accelerator.device)

    pbar = trange(args.rollout_steps, desc=f'Rollout {args.task_name}', ascii=True)
    success = torch.zeros(args.num_envs, dtype=torch.bool, device=accelerator.device)
    now_step = global_step
    
    for step in pbar:
        now_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            agent.update_state_stats(next_obs)
            batch = collate_fn(next_obs)
            if args.rollout_minibatch_size == 0:
                with accelerator.autocast():
                    if hasattr(agent, "get_action_and_value"):
                        action, logprob, _, value = agent.get_action_and_value(batch)
                    else:
                        action, logprob, _, value = agent(batch)
            else:
                action, logprob, _, value = agent_forward_microbatch_for_rollout(args, agent, batch, accelerator)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        if action.shape[0] != envs.num_envs:
            print("⚠ ENV DIM CHANGED!")
            print("step =", step)
            print("next_obs.shape =", next_obs.shape)
            print("dones.sum() =", dones.sum())
            breakpoint()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.step(action)
        next_done = torch.logical_or(terminations, truncations).to(torch.float32)
        rewards[step] = reward.view(-1) * args.reward_scale
        success = success | infos["success"].to(torch.bool)

        pbar.set_postfix({
            "grasp_rate": infos["is_grasped"].float().mean().item(),
            # "placed_rate": infos["is_obj_placed"].float().mean().item(),
            # "static_rate": infos["is_robot_static"].float().mean().item(),
            # "tcp_to_obj_pos": next_obs['state'][:, -6:-3].norm(dim=-1).mean().item(),
            # "obj_to_goal_pos": next_obs['state'][:, -3:].norm(dim=-1).mean().item(),
            "success_rate": success.float().mean().item(),
        })

        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                writer.add_scalar(f"rollout/{k}", v[done_mask].float().mean(), now_step)

            for k in infos["final_observation"]:
                infos["final_observation"][k] = infos["final_observation"][k][done_mask]
            with torch.no_grad():
                with accelerator.autocast():
                    agent.update_state_stats(infos["final_observation"])
                    batch = collate_fn(infos["final_observation"])
                    if args.rollout_minibatch_size != 0:
                        _, _, _, final_value = agent_forward_microbatch_for_rollout(args, agent, batch, accelerator, only_value=True)
                        final_values[step, torch.arange(args.num_envs, device=accelerator.device)[done_mask]] = final_value.view(-1)
                    else:
                        final_values[step, torch.arange(args.num_envs, device=accelerator.device)[done_mask]] = agent.get_value(batch).view(-1)

    return (
        obs,                 # obs
        actions,             # actions
        logprobs,            # logp
        rewards,             # rewards
        dones,               # dones
        values,              # values
        final_values,        # final values
        next_obs,            # next_obs
        next_done            # next_done
    )

# =====================================================
# PPO Update
# =====================================================
def ppo_update_on_policy(args, agent, optimizer, data, collate_fn, accelerator, stage, writer, rollouts_num_aft_env_change=None):

    b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = data

    B = b_logprobs.size(0)

    b_inds = np.arange(B)
    pbar = tqdm(total=args.update_epochs * (B // args.minibatch_size), desc=f'{stage}', ascii=True)
    
    n_updates = 0
    stats = {
        "policy_loss": 0,
        "value_loss": 0,
        "entropy": 0,
        "approx_kl": 0,
        "clip_frac": 0,
        "old_approx_kl": 0,
    }

    if not args.not_normalize_adv:
        adv_std = b_advantages.std(unbiased=False)
        if torch.isfinite(adv_std) and adv_std > 0:
            b_advantages = (b_advantages - b_advantages.mean()) / (adv_std + 1e-8)
        else:
            b_advantages = b_advantages - b_advantages.mean()

    accum_count = 0
    early_stop = False
    optimizer.zero_grad()
    for _ in range(args.update_epochs):
        np.random.shuffle(b_inds)

        for start in range(0, B, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            batch = collate_fn(b_obs[mb_inds])

            with accelerator.autocast():
                if hasattr(agent, "get_action_and_value"):
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(batch, action=b_actions[mb_inds])
                else:
                    _, newlogprob, entropy, newvalue = agent(batch, action=b_actions[mb_inds])

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            # ===== logical minibatch running stats init =====
            if accum_count % args.grad_accum_steps == 0:
                run_old_kl = 0.0
                run_kl = 0.0
                run_clipfrac = 0.0
                run_count = 0

            # ===== micro → logical minibatch running mean =====
            with torch.no_grad():
                old_kl_mb = (-logratio).mean().item()
                new_kl_mb = ((ratio - 1) - logratio).mean().item()
                clip_mb = ((ratio - 1.0).abs() > args.clip_eps).float().mean().item()

                micro_size = ratio.numel()
                new_total = run_count + micro_size

                run_old_kl += (old_kl_mb - run_old_kl) * (micro_size / new_total)
                run_kl += (new_kl_mb - run_kl) * (micro_size / new_total)
                run_clipfrac += (clip_mb - run_clipfrac) * (micro_size / new_total)

                run_count = new_total

            mb_advantages = b_advantages[mb_inds]

            # ===== Policy loss =====
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # ===== Value loss =====
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_eps,
                    args.clip_eps,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            loss = loss / args.grad_accum_steps

            accelerator.backward(loss)

            accum_count += 1

            # ===== logical minibatch boundary =====
            if accum_count % args.grad_accum_steps == 0:

                approx_kl = run_kl
                old_approx_kl = run_old_kl
                clipfrac = run_clipfrac

                # ===== EARLY STOP (必须在 step 前) =====
                if rollouts_num_aft_env_change is not None and rollouts_num_aft_env_change == -1 and args.target_kl is not None and approx_kl > args.target_kl:
                    early_stop = True
                    break

                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # ===== logging (logical minibatch level) =====
                writer.add_scalar("training/loss", loss.item(), n_updates)
                writer.add_scalar("training/policy_loss", pg_loss.item(), n_updates)
                writer.add_scalar("training/value_loss", v_loss.item(), n_updates)
                writer.add_scalar("training/entropy", entropy_loss.item(), n_updates)
                writer.add_scalar("training/approx_kl", approx_kl, n_updates)
                writer.add_scalar("training/old_approx_kl", old_approx_kl, n_updates)
                writer.add_scalar("training/clip_frac", clipfrac, n_updates)

                pbar.set_postfix(entropy=entropy_loss.item(), approx_kl=approx_kl, clip_frac=clipfrac, rollouts_num=rollouts_num_aft_env_change)

                stats["policy_loss"] += pg_loss.item()
                stats["value_loss"] += v_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clipfrac
                stats["old_approx_kl"] += old_approx_kl

                n_updates += 1

            pbar.update(1)

        if early_stop:
            break
    
    for k in stats:
        stats[k] /= (n_updates + 1e-8)

    return stats

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



if __name__ == "__main__":
    # m = ManiSkillDataset('datasets/PickCube-v1/motionplanning/trajectory.rgb+state_dict.pd_ee_delta_pose.physx_cpu.h5', task_name="PickCube-v1", normalize_states=True)
    # print(m.export_data_stat())
    pass
