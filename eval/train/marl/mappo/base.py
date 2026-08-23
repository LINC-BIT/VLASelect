import inspect
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange, tqdm
from train.reinforcement_learning.utils import DictArray, agent_forward_microbatch_for_rollout


def _huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    abs_error = error.abs()
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic.square() + delta * linear


def _compute_value_loss(args, new_value: torch.Tensor, old_value: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    returns = returns.to(device=new_value.device, dtype=new_value.dtype)
    old_value = old_value.to(device=new_value.device, dtype=new_value.dtype)
    if getattr(args, "clip_vloss", False):
        value_pred_clipped = old_value + (new_value - old_value).clamp(-args.clip_eps, args.clip_eps)
        error_clipped = returns - value_pred_clipped
        error_original = returns - new_value
        value_loss_clipped = _huber_loss(error_clipped, args.value_huber_delta)
        value_loss_original = _huber_loss(error_original, args.value_huber_delta)
        return torch.maximum(value_loss_original, value_loss_clipped).mean()
    return 0.5 * (new_value - returns).square().mean()


def _record_mappo_update_metrics(
    writer,
    stats,
    *,
    n_updates: int,
    loss_value: float,
    policy_loss_value: float,
    value_loss_value: float,
    entropy_value: float,
    approx_kl: float,
    old_approx_kl: float,
    clipfrac: float,
    full_kl_value: float,
    argmax_change_frac_value: float,
) -> None:
    writer.add_scalar("training/loss", loss_value, n_updates)
    writer.add_scalar("training/policy_loss", policy_loss_value, n_updates)
    writer.add_scalar("training/value_loss", value_loss_value, n_updates)
    writer.add_scalar("training/entropy", entropy_value, n_updates)
    writer.add_scalar("training/approx_kl", approx_kl, n_updates)
    writer.add_scalar("training/old_approx_kl", old_approx_kl, n_updates)
    writer.add_scalar("training/clip_frac", clipfrac, n_updates)
    writer.add_scalar("training/full_kl", full_kl_value, n_updates)
    writer.add_scalar("training/argmax_change_frac", argmax_change_frac_value, n_updates)

    stats["policy_loss"] += policy_loss_value
    stats["value_loss"] += value_loss_value
    stats["entropy"] += entropy_value
    stats["approx_kl"] += approx_kl
    stats["clip_frac"] += clipfrac
    stats["old_approx_kl"] += old_approx_kl
    stats["full_kl"] += full_kl_value
    stats["argmax_change_frac"] += argmax_change_frac_value

def get_trainable_optimizer_parameters(optimizer):
    params = []
    seen = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param is None or not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(param)
    return params


def _agent_supports_kwarg(agent, kwarg_name):
    try:
        signature = inspect.signature(agent.get_action_and_value)
    except (TypeError, ValueError, AttributeError):
        return False
    return kwarg_name in signature.parameters


def _get_active_agent_names(args, candidate_names):
    active_names = getattr(args, "active_agent_names", None)
    if not active_names:
        return list(candidate_names)
    active_name_set = set(active_names)
    return [name for name in candidate_names if name in active_name_set]


def _zero_action_like(action):
    if torch.is_tensor(action):
        return torch.zeros_like(action)
    return np.zeros_like(action)


def _collect_policy_output(args, agent, batch):
    supports_action_bins = _agent_supports_kwarg(agent, "return_action_bins")
    supports_token_logits = _agent_supports_kwarg(agent, "return_token_logits")
    if args.rollout_minibatch_size != 0:
        raise NotImplementedError(
            "MAPPO rollout capture currently requires rollout_minibatch_size == 0"
        )

    if supports_action_bins and supports_token_logits:
        actions_dict, logp_dict, ent_dict, value, action_bins_dict, token_logits_dict = agent.get_action_and_value(
            batch, return_action_bins=True, return_token_logits=True
        )
        return actions_dict, logp_dict, ent_dict, value, action_bins_dict, True, token_logits_dict
    if supports_action_bins:
        actions_dict, logp_dict, ent_dict, value, action_bins_dict = agent.get_action_and_value(
            batch, return_action_bins=True
        )
        return actions_dict, logp_dict, ent_dict, value, action_bins_dict, True, None

    actions_dict, logp_dict, ent_dict, value = agent.get_action_and_value(batch)
    return actions_dict, logp_dict, ent_dict, value, actions_dict, False, None


def _policy_update_forward(agent, batch, stored_actions, return_token_logits=False):
    if _agent_supports_kwarg(agent, "action_bins_input"):
        return agent.get_action_and_value(
            batch,
            action_bins_input={k: v for k, v in stored_actions.items()},
            return_token_logits=return_token_logits,
        )
    return agent.get_action_and_value(
        batch,
        actions_input={k: v for k, v in stored_actions.items()},
        return_token_logits=return_token_logits,
    )


def _snapshot_client_remote_messages(clients):
    if clients is None:
        return None
    snapshots = {}
    for client_name, client in clients.items():
        if hasattr(client, "snapshot_remote_messages"):
            snapshots[client_name] = client.snapshot_remote_messages()
    return snapshots


def _restore_client_remote_messages(clients, snapshots):
    if clients is None or snapshots is None:
        return
    for client_name, client_snapshots in snapshots.items():
        client = clients.get(client_name)
        if client is None:
            continue
        if hasattr(client, "restore_remote_messages"):
            client.restore_remote_messages(client_snapshots)


def _get_agent_names(agent):
    if hasattr(agent, "actor_heads"):
        return list(agent.actor_heads.keys())
    if hasattr(agent, "agent_names"):
        return list(agent.agent_names)
    raise AttributeError(f"Unsupported multi-agent policy type: {type(agent).__name__}")


def _extract_action_match_from_obs(obs, device, agent_names):
    if not isinstance(obs, dict):
        return None
    extra = obs.get("extra")
    if not isinstance(extra, dict):
        return None
    per_agent_match = {}
    for name in agent_names:
        key = f"action_match_{name}"
        if key not in extra:
            continue
        value = extra[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=device)
        else:
            value = value.to(device=device)
        per_agent_match[name] = value.view(-1).to(torch.bool)
    if per_agent_match:
        return per_agent_match
    if "action_match" not in extra:
        return None
    action_match = extra["action_match"]
    if not torch.is_tensor(action_match):
        action_match = torch.as_tensor(action_match, device=device)
    else:
        action_match = action_match.to(device=device)
    return action_match.view(-1).to(torch.bool)

def collect_rollout(args, agent, collate_fn, envs, next_obs, next_done,
                    accelerator, writer, global_step, clients=None):
    def apply_done_mask(x, mask):

        if torch.is_tensor(x):

            # only apply if first dim == mask size
            if x.shape[0] == mask.shape[0]:
                return x[mask]
            else:
                return x

        if isinstance(x, dict):
            return {k: apply_done_mask(v, mask) for k, v in x.items()}

        if isinstance(x, (list, tuple)):
            return type(x)(apply_done_mask(v, mask) for v in x)

        return x
    
    device = accelerator.device
    T = args.rollout_steps
    N = args.num_envs
    agent_names = _get_agent_names(agent)

    obs = DictArray((T, N), envs.single_observation_space, device=device)

    actions = {
        name: np.zeros((T, N) + envs.single_action_space[name].shape)
        for name in agent_names
    }
    action_bins = {
        name: torch.zeros((T, N) + envs.single_action_space[name].shape, dtype=torch.long, device=device)
        for name in agent_names
    }

    logprobs = {
        name: torch.zeros((T, N), device=device)
        for name in agent_names
    }

    entropies = {
        name: torch.zeros((T, N), device=device)
        for name in agent_names
    }
    old_token_logits = None
    if getattr(args, "full_kl_coef", 0.0) > 0 or getattr(args, "log_full_kl", False):
        actor_module = getattr(agent, "actor", None)
        num_action_bins = getattr(actor_module, "num_action_bins", 1)
        old_token_logits = {
            name: torch.zeros(
                (T, N) + envs.single_action_space[name].shape + (num_action_bins,),
                dtype=torch.float32,
                device=device,
            )
            for name in agent_names
        }

    actions = DictArray((T, N), None, data_dict=actions, device=device)
    action_bins = DictArray((T, N), None, data_dict=action_bins, device=device)
    logprobs = DictArray((T, N), None, data_dict=logprobs, device=device)
    entropies = DictArray((T, N), None, data_dict=entropies, device=device)
    if old_token_logits is not None:
        old_token_logits = DictArray((T, N), None, data_dict=old_token_logits, device=device)

    rewards = torch.zeros((T, N), device=device)
    dones = torch.zeros((T, N), device=device)
    values = torch.zeros((T, N), device=device)
    final_values = torch.zeros((T, N), device=device)
    communication_snapshots = [] if (clients is not None and getattr(args, "communication_replay", False)) else None

    success = torch.zeros(N, dtype=torch.bool, device=device)
    using_action_bins = False

    pbar = trange(T, ascii=True, desc='Collecting Rollout')

    now_step = global_step
    critic_warmup_steps = args.num_envs * args.rollout_steps * args.critic_warmup_rollouts
    critic_only = global_step < critic_warmup_steps

    for step in pbar:

        now_step += N

        obs[step] = next_obs
        dones[step] = next_done
        action_match = _extract_action_match_from_obs(next_obs, device, agent_names)

        with torch.no_grad():
            batch = collate_fn(next_obs)

            (
                actions_dict,
                logp_dict,
                ent_dict,
                value,
                rollout_action_dict,
                using_action_bins,
                token_logits_dict,
            ) = _collect_policy_output(
                args, agent, batch
            )

        if getattr(args, "zero_inactive_agents_during_rollout", False):
            active_rollout_names = set(_get_active_agent_names(args, actions_dict.keys()))
            for name in list(actions_dict.keys()):
                if name in active_rollout_names:
                    continue
                actions_dict[name] = _zero_action_like(actions_dict[name])
                rollout_action_dict[name] = _zero_action_like(rollout_action_dict[name])

        values[step] = value.view(-1)

        # -------- per-agent store --------
        for name in actions_dict.keys():
            if using_action_bins:
                action_bins[name][step] = rollout_action_dict[name]
            else:
                actions[name][step] = rollout_action_dict[name]
            logprobs[name][step] = logp_dict[name].detach().cpu()
            entropies[name][step] = ent_dict[name].detach().cpu()
            if old_token_logits is not None and token_logits_dict is not None:
                old_token_logits[name][step] = token_logits_dict[name].detach().to(dtype=torch.float32)

        next_obs, reward, terminations, truncations, infos = envs.step(actions_dict)

        next_done = torch.logical_or(terminations, truncations).float()

        rewards[step] = reward.view(-1) * args.reward_scale
        if communication_snapshots is not None:
            communication_snapshots.append(_snapshot_client_remote_messages(clients))

        episode_success = torch.zeros(N, dtype=torch.bool, device=device)
        if "final_info" in infos and "_final_info" in infos:
            done_mask = infos["_final_info"]
            final_info = infos["final_info"]
            if torch.is_tensor(done_mask) and isinstance(final_info, dict) and "success" in final_info:
                done_mask = done_mask.to(device=device, dtype=torch.bool).view(-1)
                final_success = final_info["success"]
                if not torch.is_tensor(final_success):
                    final_success = torch.as_tensor(final_success, device=device)
                else:
                    final_success = final_success.to(device=device)
                final_success = final_success.to(torch.bool).view(-1)
                episode_success[done_mask] = final_success[done_mask]

        success = success | infos["success"].to(torch.bool)

        if clients is not None:
            for client_name, client in clients.items():
                client_action_match = action_match
                if isinstance(action_match, dict):
                    client_action_match = action_match.get(client_name)
                client.after_each_forward_during_rollout(
                    reward.view(-1).detach(),
                    next_done.detach(),
                    action_mean=torch.tensor(actions_dict[client_name], device=device).detach() if client_name in actions_dict else None,
                    success=episode_success.detach(),
                    action_match=client_action_match.detach() if client_action_match is not None else None,
                )

        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                writer.add_scalar(f"rollout/{k}", v[done_mask].float().mean(), now_step)

            if isinstance(final_info, dict):
                fail_bool_keys = [
                    "fail_drop",
                    "fail_nonfinite",
                    "push_contact_timeout",
                    "table_goal_stalled",
                ]
                for key in fail_bool_keys:
                    if key not in final_info:
                        continue
                    value = final_info[key]
                    if not torch.is_tensor(value):
                        value = torch.as_tensor(value, device=device)
                    else:
                        value = value.to(device=device)
                    writer.add_scalar(
                        f"rollout/{key}",
                        value[done_mask].float().mean(),
                        now_step,
                    )

                if "fail_reason" in final_info:
                    fail_reason = final_info["fail_reason"]
                    if not torch.is_tensor(fail_reason):
                        fail_reason = torch.as_tensor(fail_reason, device=device)
                    else:
                        fail_reason = fail_reason.to(device=device)
                    fail_reason = fail_reason[done_mask].to(torch.long)
                    max_reason = int(fail_reason.max().item()) if fail_reason.numel() > 0 else 0
                    for reason_id in range(max_reason + 1):
                        writer.add_scalar(
                            f"rollout/fail_reason_{reason_id}",
                            (fail_reason == reason_id).float().mean(),
                            now_step,
                        )

            infos["final_observation"] = apply_done_mask(infos["final_observation"], done_mask)
            with torch.no_grad():
                with accelerator.autocast():
                    batch = collate_fn(infos["final_observation"])
                    if args.rollout_minibatch_size != 0:
                        _, _, _, final_value = agent_forward_microbatch_for_rollout(args, agent, batch, accelerator, only_value=True)
                        final_values[step, torch.arange(args.num_envs, device=accelerator.device)[done_mask]] = (
                            final_value.view(-1).to(dtype=final_values.dtype)
                        )
                    else:
                        final_values[step, torch.arange(args.num_envs, device=accelerator.device)[done_mask]] = (
                            agent.get_value(batch).view(-1).to(dtype=final_values.dtype)
                        )

    rollout_actions = action_bins if using_action_bins else actions

    return_values = (
        obs,
        rollout_actions,
        logprobs,
        # entropies,
        rewards,
        dones,
        values,
        final_values,
        next_obs,
        next_done
    )
    if old_token_logits is not None:
        if communication_snapshots is not None:
            return return_values + (old_token_logits, communication_snapshots)
        return return_values + (old_token_logits,)
    if communication_snapshots is not None:
        return return_values + (communication_snapshots,)
    return return_values

def mappo_update_on_policy(
    args,
    agent,
    optimizer,
    data,
    collate_fn,
    accelerator,
    stage,
    writer,
    rollouts_num_aft_env_change=None
):

    (
        b_obs,
        b_actions,
        b_logprobs,
        b_advantages,
        b_returns,
        b_values,
        *extra_data,
    ) = data
    b_old_token_logits = extra_data[0] if extra_data else None

    device = accelerator.device

    B = b_advantages.shape[0]
    b_inds = np.arange(B)

    pbar = tqdm(total=args.update_epochs * (B // args.minibatch_size), desc=stage, ascii=True)

    stats = {
        "policy_loss": 0,
        "value_loss": 0,
        "entropy": 0,
        "approx_kl": 0,
        "clip_frac": 0,
        "old_approx_kl": 0,
        "full_kl": 0,
        "argmax_change_frac": 0,
    }

    n_updates = 0
    accum_count = 0
    early_stop = False
    critic_only_update = getattr(args, "critic_only_update", False)

    if not args.not_normalize_adv:
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    optimizer.zero_grad()

    for _ in range(args.update_epochs):

        np.random.shuffle(b_inds)

        for start in range(0, B, args.minibatch_size):

            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            batch = collate_fn(b_obs[mb_inds])

            # ================= MAPPO forward =================
            forward_out = _policy_update_forward(
                agent,
                batch,
                {k: v[mb_inds] for k, v in b_actions.data.items()},
                return_token_logits=b_old_token_logits is not None,
            )
            if b_old_token_logits is not None:
                _, new_logp, entropy, new_value, new_token_logits = forward_out
            else:
                _, new_logp, entropy, new_value = forward_out
                new_token_logits = None

            new_value = new_value.view(-1)
            if (
                not torch.isfinite(new_value).all()
                or any(not torch.isfinite(v).all() for v in new_logp.values())
                or any(not torch.isfinite(v).all() for v in entropy.values())
            ):
                print(f"[{stage}] skip non-finite forward output")
                pbar.update(1)
                continue

            # ================= VALUE LOSS =================
            v_loss = _compute_value_loss(
                args,
                new_value,
                b_values[mb_inds].view(-1),
                b_returns[mb_inds].view(-1),
            )
            if not torch.isfinite(v_loss):
                print(f"[{stage}] skip non-finite value loss")
                pbar.update(1)
                continue

            # ================= RUNNING KL STATS (PPO完全保留) =================
            if accum_count % args.grad_accum_steps == 0:
                run_old_kl = 0.0
                run_kl = 0.0
                run_clipfrac = 0.0
                run_count = 0

            policy_loss = 0
            entropy_loss = 0
            approx_kl = 0
            clip_frac = 0
            full_kl = torch.tensor(0.0, device=device)
            argmax_change_frac = torch.tensor(0.0, device=device)

            # ================= per-agent PPO =================
            active_agent_names = _get_active_agent_names(args, new_logp.keys())
            for name in active_agent_names:

                logratio = new_logp[name] - b_logprobs[name][mb_inds]
                ratio = logratio.exp()

                adv = b_advantages[mb_inds]

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(
                    ratio,
                    1 - args.clip_eps,
                    1 + args.clip_eps
                )

                policy_loss += torch.max(pg1, pg2).mean()

                entropy_loss += entropy[name].mean()

                # ===== PPO KL tracking（完全保留）=====
                with torch.no_grad():
                    old_kl_mb = (-logratio).mean().item()
                    new_kl_mb = ((ratio - 1) - logratio).mean().item()
                    clip_mb = ((ratio - 1).abs() > args.clip_eps).float().mean().item()

                    micro_size = ratio.numel()
                    new_total = run_count + micro_size

                    run_old_kl += (old_kl_mb - run_old_kl) * (micro_size / new_total)
                    run_kl += (new_kl_mb - run_kl) * (micro_size / new_total)
                    run_clipfrac += (clip_mb - run_clipfrac) * (micro_size / new_total)

                    run_count = new_total

                approx_kl += logratio.mean().item()
                clip_frac += clip_mb

                if b_old_token_logits is not None and new_token_logits is not None:
                    old_logits = b_old_token_logits[name][mb_inds].to(device=device, dtype=torch.float32)
                    new_logits = new_token_logits[name].to(device=device, dtype=torch.float32)
                    old_log_probs_full = torch.log_softmax(old_logits, dim=-1)
                    new_log_probs_full = torch.log_softmax(new_logits, dim=-1)
                    old_probs_full = old_log_probs_full.exp()
                    full_kl += (old_probs_full * (old_log_probs_full - new_log_probs_full)).sum(dim=-1).mean()
                    argmax_change_frac += (
                        old_logits.argmax(dim=-1) != new_logits.argmax(dim=-1)
                    ).to(torch.float32).mean()

            num_agents = max(len(active_agent_names), 1)
            policy_loss = policy_loss / num_agents
            entropy_loss = entropy_loss / num_agents
            approx_kl = approx_kl / num_agents
            full_kl = full_kl / num_agents
            argmax_change_frac = argmax_change_frac / num_agents
            if critic_only_update:
                loss = args.vf_coef * v_loss
            else:
                loss = (
                    policy_loss
                    - args.ent_coef * entropy_loss
                    + args.vf_coef * v_loss
                )
            if (
                not critic_only_update
                and getattr(args, "full_kl_coef", 0.0) > 0
                and b_old_token_logits is not None
            ):
                loss = loss + args.full_kl_coef * full_kl

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                print(
                    f"[{stage}] skip non-finite loss: "
                    f"loss={loss.detach().item() if loss.numel() == 1 else loss}, "
                    f"policy={policy_loss.detach().item() if torch.is_tensor(policy_loss) else policy_loss}, "
                    f"value={v_loss.detach().item() if torch.is_tensor(v_loss) else v_loss}, "
                    f"entropy={entropy_loss.detach().item() if torch.is_tensor(entropy_loss) else entropy_loss}"
                )
                pbar.update(1)
                continue

            loss = loss / args.grad_accum_steps

            accelerator.backward(loss)

            accum_count += 1

            # ================= optimizer step =================
            if accum_count % args.grad_accum_steps == 0:

                approx_kl = run_kl
                old_approx_kl = run_old_kl
                clipfrac = run_clipfrac

                # ===== EARLY STOP（完全保留）=====
                if (
                    not critic_only_update
                    and
                    rollouts_num_aft_env_change is not None
                    and rollouts_num_aft_env_change == -1
                    and args.target_kl is not None
                    and approx_kl > args.target_kl
                ):
                    _record_mappo_update_metrics(
                        writer,
                        stats,
                        n_updates=n_updates,
                        loss_value=loss.item(),
                        policy_loss_value=policy_loss.item(),
                        value_loss_value=v_loss.item(),
                        entropy_value=entropy_loss.item(),
                        approx_kl=approx_kl,
                        old_approx_kl=old_approx_kl,
                        clipfrac=clipfrac,
                        full_kl_value=full_kl.item(),
                        argmax_change_frac_value=argmax_change_frac.item(),
                    )
                    n_updates += 1
                    print(
                        f"[{stage}] early stop on KL: "
                        f"approx_kl={approx_kl:.6f}, old_approx_kl={old_approx_kl:.6f}, "
                        f"target={args.target_kl:.6f}, clip_frac={clipfrac:.6f}"
                    )
                    early_stop = True
                    break

                grad_norm = nn.utils.clip_grad_norm_(
                    get_trainable_optimizer_parameters(optimizer),
                    args.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad()
                    print(f"[{stage}] skip non-finite grad_norm={grad_norm}")
                    pbar.update(1)
                    continue
                optimizer.step()
                optimizer.zero_grad()

                # ================= logging（对齐PPO）=================
                _record_mappo_update_metrics(
                    writer,
                    stats,
                    n_updates=n_updates,
                    loss_value=loss.item(),
                    policy_loss_value=policy_loss.item(),
                    value_loss_value=v_loss.item(),
                    entropy_value=entropy_loss.item(),
                    approx_kl=approx_kl,
                    old_approx_kl=old_approx_kl,
                    clipfrac=clipfrac,
                    full_kl_value=full_kl.item(),
                    argmax_change_frac_value=argmax_change_frac.item(),
                )

                n_updates += 1

            pbar.update(1)

        if early_stop:
            break

    for k in stats:
        stats[k] /= (n_updates + 1e-8)

    return stats

def mappo_update_on_policy_ag(
    args,
    agent,
    optimizer,
    data,
    collate_fn,
    accelerator,
    stage,
    clients,
    writer,
    rollouts_num_aft_env_change=None,
    target_kl_override=None,
):

    (
        b_obs,
        b_actions,
        b_logprobs,
        b_advantages,
        b_returns,
        b_values,
        *extra_data,
    ) = data
    b_old_token_logits = None
    b_communication_snapshots = None
    if len(extra_data) == 1:
        if isinstance(extra_data[0], list):
            b_communication_snapshots = extra_data[0]
        else:
            b_old_token_logits = extra_data[0]
    elif len(extra_data) >= 2:
        b_old_token_logits = extra_data[0]
        b_communication_snapshots = extra_data[1]

    device = accelerator.device

    B = b_advantages.shape[0]
    b_inds = np.arange(B)

    pbar = tqdm(total=args.update_epochs * (B // args.minibatch_size), desc=stage, ascii=True)

    stats = {
        "policy_loss": 0,
        "value_loss": 0,
        "entropy": 0,
        "approx_kl": 0,
        "clip_frac": 0,
        "old_approx_kl": 0,
        "full_kl": 0,
        "argmax_change_frac": 0,
    }

    n_updates = 0
    accum_count = 0
    early_stop = False

    if not args.not_normalize_adv:
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    optimizer.zero_grad()
    gate_reg_coef = getattr(args, "gate_reg_coef", 0.0)
    gate_target_mean = getattr(args, "gate_target_mean", 0.6)
    gate_std_coef = getattr(args, "gate_std_coef", 0.0)
    feature_gate_reg_coef = getattr(args, "feature_gate_reg_coef", None)
    feature_gate_target_mean = getattr(args, "feature_gate_target_mean", None)
    feature_gate_std_coef = getattr(args, "feature_gate_std_coef", None)
    action_gate_reg_coef = getattr(args, "action_gate_reg_coef", None)
    action_gate_target_mean = getattr(args, "action_gate_target_mean", None)
    action_gate_std_coef = getattr(args, "action_gate_std_coef", None)
    feature_gate_quality_coef = getattr(args, "feature_gate_quality_coef", 0.0)
    action_gate_quality_coef = getattr(args, "action_gate_quality_coef", 0.0)
    feature_gate_reg_coef = gate_reg_coef if feature_gate_reg_coef is None else feature_gate_reg_coef
    feature_gate_target_mean = gate_target_mean if feature_gate_target_mean is None else feature_gate_target_mean
    feature_gate_std_coef = gate_std_coef if feature_gate_std_coef is None else feature_gate_std_coef
    action_gate_reg_coef = gate_reg_coef if action_gate_reg_coef is None else action_gate_reg_coef
    action_gate_target_mean = gate_target_mean if action_gate_target_mean is None else action_gate_target_mean
    action_gate_std_coef = gate_std_coef if action_gate_std_coef is None else action_gate_std_coef
    feature_consistency_coef = getattr(args, "feature_consistency_coef", 0.0)
    action_consistency_coef = getattr(args, "action_consistency_coef", 0.0)
    feature_attn_entropy_coef = getattr(args, "feature_attn_entropy_coef", 0.0)
    action_attn_entropy_coef = getattr(args, "action_attn_entropy_coef", 0.0)
    feature_attn_diversity_coef = getattr(args, "feature_attn_diversity_coef", 0.0)
    action_attn_diversity_coef = getattr(args, "action_attn_diversity_coef", 0.0)

    for _ in range(args.update_epochs):

        np.random.shuffle(b_inds)

        for start in range(0, B, args.minibatch_size):

            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            batch = collate_fn(b_obs[mb_inds])
            if b_communication_snapshots is not None and len(mb_inds) > 0:
                unique_inds, counts = np.unique(mb_inds, return_counts=True)
                dominant_snapshot_ind = int(unique_inds[np.argmax(counts)])
                _restore_client_remote_messages(clients, b_communication_snapshots[dominant_snapshot_ind])

            # ================= MAPPO forward =================
            forward_out = _policy_update_forward(
                agent,
                batch,
                {k: v[mb_inds] for k, v in b_actions.data.items()},
                return_token_logits=b_old_token_logits is not None,
            )
            if b_old_token_logits is not None:
                _, new_logp, entropy, new_value, new_token_logits = forward_out
            else:
                _, new_logp, entropy, new_value = forward_out
                new_token_logits = None

            new_value = new_value.view(-1)

            # ================= VALUE LOSS =================
            v_loss = _compute_value_loss(
                args,
                new_value,
                b_values[mb_inds].view(-1),
                b_returns[mb_inds].view(-1),
            )

            # ================= RUNNING KL STATS (PPO完全保留) =================
            if accum_count % args.grad_accum_steps == 0:
                run_old_kl = 0.0
                run_kl = 0.0
                run_clipfrac = 0.0
                run_count = 0

            policy_loss = 0
            entropy_loss = 0
            approx_kl = 0
            clip_frac = 0
            full_kl = torch.tensor(0.0, device=device)
            argmax_change_frac = torch.tensor(0.0, device=device)

            # ================= per-agent PPO =================
            for name in new_logp.keys():

                logratio = new_logp[name] - b_logprobs[name][mb_inds]
                ratio = logratio.exp()

                adv = b_advantages[mb_inds]

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(
                    ratio,
                    1 - args.clip_eps,
                    1 + args.clip_eps
                )

                policy_loss += torch.max(pg1, pg2).mean()

                entropy_loss += entropy[name].mean()

                # ===== PPO KL tracking（完全保留）=====
                with torch.no_grad():
                    old_kl_mb = (-logratio).mean().item()
                    new_kl_mb = ((ratio - 1) - logratio).mean().item()
                    clip_mb = ((ratio - 1).abs() > args.clip_eps).float().mean().item()

                    micro_size = ratio.numel()
                    new_total = run_count + micro_size

                    run_old_kl += (old_kl_mb - run_old_kl) * (micro_size / new_total)
                    run_kl += (new_kl_mb - run_kl) * (micro_size / new_total)
                    run_clipfrac += (clip_mb - run_clipfrac) * (micro_size / new_total)

                    run_count = new_total

                approx_kl += logratio.mean().item()
                clip_frac += clip_mb

                if b_old_token_logits is not None and new_token_logits is not None:
                    old_logits = b_old_token_logits[name][mb_inds].to(device=device, dtype=torch.float32)
                    new_logits = new_token_logits[name].to(device=device, dtype=torch.float32)
                    old_log_probs_full = torch.log_softmax(old_logits, dim=-1)
                    new_log_probs_full = torch.log_softmax(new_logits, dim=-1)
                    old_probs_full = old_log_probs_full.exp()
                    full_kl += (old_probs_full * (old_log_probs_full - new_log_probs_full)).sum(dim=-1).mean()
                    argmax_change_frac += (
                        old_logits.argmax(dim=-1) != new_logits.argmax(dim=-1)
                    ).to(torch.float32).mean()

            num_agents = max(len(new_logp), 1)
            policy_loss = policy_loss / num_agents
            entropy_loss = entropy_loss / num_agents
            approx_kl = approx_kl / num_agents
            full_kl = full_kl / num_agents
            argmax_change_frac = argmax_change_frac / num_agents
            loss = (
                policy_loss
                - args.ent_coef * entropy_loss
                + args.vf_coef * v_loss
            )
            if getattr(args, "full_kl_coef", 0.0) > 0 and b_old_token_logits is not None:
                loss = loss + args.full_kl_coef * full_kl

            gate_reg_loss = torch.tensor(0.0, device=device)
            gate_std_bonus = torch.tensor(0.0, device=device)
            gate_quality_loss = torch.tensor(0.0, device=device)
            consistency_loss = torch.tensor(0.0, device=device)
            attention_im_loss = torch.tensor(0.0, device=device)
            feature_gate_mean_avg = None
            feature_gate_std_avg = None
            action_gate_mean_avg = None
            action_gate_std_avg = None
            feature_gate_quality_avg = None
            action_gate_quality_avg = None
            feature_q_avg = None
            action_q_avg = None
            feature_q_ret_avg = None
            action_q_ret_avg = None
            feature_q_attn_avg = None
            action_q_attn_avg = None
            feature_q_attn_token_avg = None
            action_q_attn_token_avg = None
            feature_q_attn_traj_avg = None
            action_q_attn_traj_avg = None
            feature_consistency_avg = None
            action_consistency_avg = None
            feature_attn_entropy_avg = None
            action_attn_entropy_avg = None
            feature_attn_diversity_avg = None
            action_attn_diversity_avg = None
            feature_attn_im_avg = None
            action_attn_im_avg = None

            ppo_loss = loss / args.grad_accum_steps

            if (
                feature_gate_reg_coef > 0
                or feature_gate_std_coef > 0
                or action_gate_reg_coef > 0
                or action_gate_std_coef > 0
                or feature_gate_quality_coef > 0
                or action_gate_quality_coef > 0
                or feature_consistency_coef > 0
                or action_consistency_coef > 0
                or feature_attn_entropy_coef > 0
                or action_attn_entropy_coef > 0
                or feature_attn_diversity_coef > 0
                or action_attn_diversity_coef > 0
            ):
                feature_gate_means = []
                feature_gate_stds = []
                action_gate_means = []
                action_gate_stds = []
                feature_gate_quality_losses = []
                action_gate_quality_losses = []
                feature_q_values = []
                action_q_values = []
                feature_q_ret_values = []
                action_q_ret_values = []
                feature_q_attn_values = []
                action_q_attn_values = []
                feature_q_attn_token_values = []
                action_q_attn_token_values = []
                feature_q_attn_traj_values = []
                action_q_attn_traj_values = []
                feature_consistency_losses = []
                action_consistency_losses = []
                feature_attn_entropy_losses = []
                action_attn_entropy_losses = []
                feature_attn_diversity_losses = []
                action_attn_diversity_losses = []
                feature_attn_im_losses = []
                action_attn_im_losses = []
                for client in clients.values():
                    for fa_group in client.feature_aggregators.values():
                        aggregators = fa_group if isinstance(fa_group, (list, tuple)) else [fa_group]
                        for fa in aggregators:
                            if fa.module.feature_gate_mean is not None:
                                feature_gate_means.append(fa.module.feature_gate_mean)
                            if fa.module.feature_gate_std is not None:
                                feature_gate_stds.append(fa.module.feature_gate_std)
                            if fa.module.action_gate_mean is not None:
                                action_gate_means.append(fa.module.action_gate_mean)
                            if fa.module.action_gate_std is not None:
                                action_gate_stds.append(fa.module.action_gate_std)
                            if fa.module.feature_gate_quality_loss is not None:
                                feature_gate_quality_losses.append(fa.module.feature_gate_quality_loss)
                            if fa.module.action_gate_quality_loss is not None:
                                action_gate_quality_losses.append(fa.module.action_gate_quality_loss)
                            if fa.module.feature_q_mean is not None:
                                feature_q_values.append(fa.module.feature_q_mean)
                            if fa.module.action_q_mean is not None:
                                action_q_values.append(fa.module.action_q_mean)
                            if fa.module.feature_q_ret_mean is not None:
                                feature_q_ret_values.append(fa.module.feature_q_ret_mean)
                            if fa.module.action_q_ret_mean is not None:
                                action_q_ret_values.append(fa.module.action_q_ret_mean)
                            if fa.module.feature_q_attn_mean is not None:
                                feature_q_attn_values.append(fa.module.feature_q_attn_mean)
                            if fa.module.action_q_attn_mean is not None:
                                action_q_attn_values.append(fa.module.action_q_attn_mean)
                            if getattr(fa.module, "feature_q_attn_token_mean", None) is not None:
                                feature_q_attn_token_values.append(fa.module.feature_q_attn_token_mean)
                            if getattr(fa.module, "action_q_attn_token_mean", None) is not None:
                                action_q_attn_token_values.append(fa.module.action_q_attn_token_mean)
                            if getattr(fa.module, "feature_q_attn_traj_mean", None) is not None:
                                feature_q_attn_traj_values.append(fa.module.feature_q_attn_traj_mean)
                            if getattr(fa.module, "action_q_attn_traj_mean", None) is not None:
                                action_q_attn_traj_values.append(fa.module.action_q_attn_traj_mean)
                            if fa.module.feature_consistency_loss is not None:
                                feature_consistency_losses.append(fa.module.feature_consistency_loss)
                            if fa.module.action_consistency_loss is not None:
                                action_consistency_losses.append(fa.module.action_consistency_loss)
                            if getattr(fa.module, "feature_attn_entropy_loss", None) is not None:
                                feature_attn_entropy_losses.append(fa.module.feature_attn_entropy_loss)
                            if getattr(fa.module, "action_attn_entropy_loss", None) is not None:
                                action_attn_entropy_losses.append(fa.module.action_attn_entropy_loss)
                            if getattr(fa.module, "feature_attn_diversity_loss", None) is not None:
                                feature_attn_diversity_losses.append(fa.module.feature_attn_diversity_loss)
                            if getattr(fa.module, "action_attn_diversity_loss", None) is not None:
                                action_attn_diversity_losses.append(fa.module.action_attn_diversity_loss)
                            if getattr(fa.module, "feature_attn_im_loss", None) is not None:
                                feature_attn_im_losses.append(fa.module.feature_attn_im_loss)
                            if getattr(fa.module, "action_attn_im_loss", None) is not None:
                                action_attn_im_losses.append(fa.module.action_attn_im_loss)

                if feature_gate_means:
                    feature_gate_mean_avg = torch.stack(feature_gate_means).mean()
                    gate_reg_loss = gate_reg_loss + feature_gate_reg_coef * torch.relu(
                        feature_gate_mean_avg - feature_gate_target_mean
                    )
                if action_gate_means:
                    action_gate_mean_avg = torch.stack(action_gate_means).mean()
                    gate_reg_loss = gate_reg_loss + action_gate_reg_coef * torch.relu(
                        action_gate_mean_avg - action_gate_target_mean
                    )
                if feature_gate_stds:
                    feature_gate_std_avg = torch.stack(feature_gate_stds).mean()
                    gate_std_bonus = gate_std_bonus - feature_gate_std_coef * feature_gate_std_avg
                if action_gate_stds:
                    action_gate_std_avg = torch.stack(action_gate_stds).mean()
                    gate_std_bonus = gate_std_bonus - action_gate_std_coef * action_gate_std_avg
                if feature_gate_quality_losses:
                    feature_gate_quality_avg = torch.stack(feature_gate_quality_losses).mean()
                    gate_quality_loss = gate_quality_loss + feature_gate_quality_coef * feature_gate_quality_avg
                if action_gate_quality_losses:
                    action_gate_quality_avg = torch.stack(action_gate_quality_losses).mean()
                    gate_quality_loss = gate_quality_loss + action_gate_quality_coef * action_gate_quality_avg
                if feature_q_values:
                    feature_q_avg = torch.stack(feature_q_values).mean()
                if action_q_values:
                    action_q_avg = torch.stack(action_q_values).mean()
                if feature_q_ret_values:
                    feature_q_ret_avg = torch.stack(feature_q_ret_values).mean()
                if action_q_ret_values:
                    action_q_ret_avg = torch.stack(action_q_ret_values).mean()
                if feature_q_attn_values:
                    feature_q_attn_avg = torch.stack(feature_q_attn_values).mean()
                if action_q_attn_values:
                    action_q_attn_avg = torch.stack(action_q_attn_values).mean()
                if feature_q_attn_token_values:
                    feature_q_attn_token_avg = torch.stack(feature_q_attn_token_values).mean()
                if action_q_attn_token_values:
                    action_q_attn_token_avg = torch.stack(action_q_attn_token_values).mean()
                if feature_q_attn_traj_values:
                    feature_q_attn_traj_avg = torch.stack(feature_q_attn_traj_values).mean()
                if action_q_attn_traj_values:
                    action_q_attn_traj_avg = torch.stack(action_q_attn_traj_values).mean()
                if feature_consistency_losses:
                    feature_consistency_avg = torch.stack(feature_consistency_losses).mean()
                    consistency_loss = consistency_loss + feature_consistency_coef * feature_consistency_avg
                if action_consistency_losses:
                    action_consistency_avg = torch.stack(action_consistency_losses).mean()
                    consistency_loss = consistency_loss + action_consistency_coef * action_consistency_avg
                if feature_attn_entropy_losses:
                    feature_attn_entropy_avg = torch.stack(feature_attn_entropy_losses).mean()
                    attention_im_loss = attention_im_loss + feature_attn_entropy_coef * feature_attn_entropy_avg
                if action_attn_entropy_losses:
                    action_attn_entropy_avg = torch.stack(action_attn_entropy_losses).mean()
                    attention_im_loss = attention_im_loss + action_attn_entropy_coef * action_attn_entropy_avg
                if feature_attn_diversity_losses:
                    feature_attn_diversity_avg = torch.stack(feature_attn_diversity_losses).mean()
                    attention_im_loss = attention_im_loss + feature_attn_diversity_coef * feature_attn_diversity_avg
                if action_attn_diversity_losses:
                    action_attn_diversity_avg = torch.stack(action_attn_diversity_losses).mean()
                    attention_im_loss = attention_im_loss + action_attn_diversity_coef * action_attn_diversity_avg
                if feature_attn_im_losses:
                    feature_attn_im_avg = torch.stack(feature_attn_im_losses).mean()
                if action_attn_im_losses:
                    action_attn_im_avg = torch.stack(action_attn_im_losses).mean()

            auxiliary_loss = gate_reg_loss + gate_std_bonus + gate_quality_loss + consistency_loss + attention_im_loss
            loss = ppo_loss + auxiliary_loss

            accelerator.backward(loss)

            accum_count += 1

            # ================= optimizer step =================
            if accum_count % args.grad_accum_steps == 0:

                approx_kl = run_kl
                old_approx_kl = run_old_kl
                clipfrac = run_clipfrac
                effective_target_kl = (
                    args.aggregator_target_kl
                    if target_kl_override is None else target_kl_override
                )

                # ===== EARLY STOP（完全保留）=====
                if (
                    rollouts_num_aft_env_change is not None
                    and rollouts_num_aft_env_change == -1
                    and effective_target_kl is not None
                    and approx_kl > effective_target_kl
                ):
                    _record_mappo_update_metrics(
                        writer,
                        stats,
                        n_updates=n_updates,
                        loss_value=loss.item(),
                        policy_loss_value=policy_loss.item(),
                        value_loss_value=v_loss.item(),
                        entropy_value=entropy_loss.item(),
                        approx_kl=approx_kl,
                        old_approx_kl=old_approx_kl,
                        clipfrac=clipfrac,
                        full_kl_value=full_kl.item(),
                        argmax_change_frac_value=argmax_change_frac.item(),
                    )
                    n_updates += 1
                    print(
                        f"[{stage}] early stop on KL: "
                        f"approx_kl={approx_kl:.6f}, target={effective_target_kl:.6f}"
                    )
                    early_stop = True
                    break

                nn.utils.clip_grad_norm_(
                    get_trainable_optimizer_parameters(optimizer),
                    args.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()

                # ================= logging（对齐PPO）=================
                writer.add_scalar("training/aggregator/loss", loss.item(), n_updates)
                writer.add_scalar("training/aggregator/policy_loss", policy_loss.item(), n_updates)
                writer.add_scalar("training/aggregator/value_loss", v_loss.item(), n_updates)
                writer.add_scalar("training/aggregator/entropy", entropy_loss.item(), n_updates)
                writer.add_scalar("training/aggregator/approx_kl", approx_kl, n_updates)
                writer.add_scalar("training/aggregator/old_approx_kl", old_approx_kl, n_updates)
                writer.add_scalar("training/aggregator/clip_frac", clipfrac, n_updates)
                writer.add_scalar("training/aggregator/full_kl", full_kl.item(), n_updates)
                writer.add_scalar("training/aggregator/argmax_change_frac", argmax_change_frac.item(), n_updates)
                writer.add_scalar("training/aggregator/gate_reg_loss", gate_reg_loss.item(), n_updates)
                writer.add_scalar("training/aggregator/gate_std_bonus", gate_std_bonus.item(), n_updates)
                writer.add_scalar("training/aggregator/gate_quality_loss", gate_quality_loss.item(), n_updates)
                writer.add_scalar("training/aggregator/consistency_loss", consistency_loss.item(), n_updates)
                writer.add_scalar("training/aggregator/attention_im_loss", attention_im_loss.item(), n_updates)
                if feature_gate_mean_avg is not None:
                    writer.add_scalar("monitoring/aggregation/gate/feature/mean", feature_gate_mean_avg.item(), n_updates)
                if feature_gate_std_avg is not None:
                    writer.add_scalar("monitoring/aggregation/gate/feature/std", feature_gate_std_avg.item(), n_updates)
                if action_gate_mean_avg is not None:
                    writer.add_scalar("monitoring/aggregation/gate/action/mean", action_gate_mean_avg.item(), n_updates)
                if action_gate_std_avg is not None:
                    writer.add_scalar("monitoring/aggregation/gate/action/std", action_gate_std_avg.item(), n_updates)
                if feature_gate_quality_avg is not None:
                    writer.add_scalar("monitoring/aggregation/gate/feature/quality", feature_gate_quality_avg.item(), n_updates)
                if action_gate_quality_avg is not None:
                    writer.add_scalar("monitoring/aggregation/gate/action/quality", action_gate_quality_avg.item(), n_updates)
                if feature_q_avg is not None:
                    writer.add_scalar("monitoring/aggregation/q/feature/value", feature_q_avg.item(), n_updates)
                if action_q_avg is not None:
                    writer.add_scalar("monitoring/aggregation/q/action/value", action_q_avg.item(), n_updates)
                if feature_q_ret_avg is not None:
                    writer.add_scalar("monitoring/aggregation/q/feature/ret", feature_q_ret_avg.item(), n_updates)
                if action_q_ret_avg is not None:
                    writer.add_scalar("monitoring/aggregation/q/action/ret", action_q_ret_avg.item(), n_updates)
                if feature_q_attn_avg is not None:
                    writer.add_scalar("monitoring/aggregation/q/feature/attn", feature_q_attn_avg.item(), n_updates)
                if action_q_attn_avg is not None:
                    writer.add_scalar("monitoring/aggregation/q/action/attn", action_q_attn_avg.item(), n_updates)
                if feature_q_attn_token_avg is not None:
                    writer.add_scalar("monitoring/aggregation/q/feature/attn_token", feature_q_attn_token_avg.item(), n_updates)
                if action_q_attn_token_avg is not None:
                    writer.add_scalar("monitoring/aggregation/q/action/attn_token", action_q_attn_token_avg.item(), n_updates)
                if feature_q_attn_traj_avg is not None:
                    writer.add_scalar("monitoring/aggregation/q/feature/attn_traj", feature_q_attn_traj_avg.item(), n_updates)
                if action_q_attn_traj_avg is not None:
                    writer.add_scalar("monitoring/aggregation/q/action/attn_traj", action_q_attn_traj_avg.item(), n_updates)
                if feature_consistency_avg is not None:
                    writer.add_scalar("monitoring/aggregation/consistency/feature", feature_consistency_avg.item(), n_updates)
                if action_consistency_avg is not None:
                    writer.add_scalar("monitoring/aggregation/consistency/action", action_consistency_avg.item(), n_updates)
                if feature_attn_entropy_avg is not None:
                    writer.add_scalar("monitoring/aggregation/attention/feature/entropy_loss", feature_attn_entropy_avg.item(), n_updates)
                if action_attn_entropy_avg is not None:
                    writer.add_scalar("monitoring/aggregation/attention/action/entropy_loss", action_attn_entropy_avg.item(), n_updates)
                if feature_attn_diversity_avg is not None:
                    writer.add_scalar("monitoring/aggregation/attention/feature/diversity_loss", feature_attn_diversity_avg.item(), n_updates)
                if action_attn_diversity_avg is not None:
                    writer.add_scalar("monitoring/aggregation/attention/action/diversity_loss", action_attn_diversity_avg.item(), n_updates)
                if feature_attn_im_avg is not None:
                    writer.add_scalar("monitoring/aggregation/attention/feature/im_loss", feature_attn_im_avg.item(), n_updates)
                if action_attn_im_avg is not None:
                    writer.add_scalar("monitoring/aggregation/attention/action/im_loss", action_attn_im_avg.item(), n_updates)

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += v_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clipfrac
                stats["old_approx_kl"] += old_approx_kl
                stats["full_kl"] += full_kl.item()
                stats["argmax_change_frac"] += argmax_change_frac.item()

                n_updates += 1

            pbar.update(1)

        if early_stop:
            break

    for k in stats:
        stats[k] /= (n_updates + 1e-8)

    return stats
