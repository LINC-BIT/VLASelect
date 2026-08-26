import os
import sys
import time
import bisect
import random
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train.toy_cnn.multi_agents.two_robot_pick.gpu_auto_select import configure_cuda_visible_devices

configure_cuda_visible_devices()

import argparse

import gymnasium as gym
import numpy as np
import torch
from train.common.env_cleanup import clear_torch_cuda_cache, close_envs

os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")
import torch.multiprocessing as mp
import torch.optim as optim
from accelerate import Accelerator
from mani_skill.utils.io_utils import dump_json, load_json
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import envs.two_robot_pick_cube_v2  # noqa: F401
from train.marl.mappo.base import collect_rollout, mappo_update_on_policy, mappo_update_on_policy_ag
from train.reinforcement_learning.evaluate import evaluate
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.utils import compute_gae, get_step_infos
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.mappo_feature_aggregator_pretrain import (
    build_clients,
    get_agent_info,
    log_effective_transmission_stats,
    infer_aggregator_ckpt_prefix,
    infer_best_step_from_metrics,
    iter_client_feature_aggregators,
    log_agent_feature_cosine_similarity,
    log_feature_aggregator_agent_cosine_similarity,
    log_uploaded_q_task_stats,
    make_collate_fn,
    make_sample_fn,
    maybe_load_initial_agent,
    set_client_feature_aggregator_requires_grad,
    set_optimizer_phase_lrs,
    set_requires_grad,
)
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.mixed_sft_agent import (
    MixedTinyVLAAdapterSmolVLASFTAgent,
    build_mixed_mappo_optimizer,
)
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.online_utils import build_continual_env_schedule

mp.set_start_method("spawn", force=True)


MODEL_NAME = "vla_adapter_smolvla_mappo"
ALGO_NAME = "ppo"
AGENT_CLS = MixedTinyVLAAdapterSmolVLASFTAgent
OPTIMIZER_FN = build_mixed_mappo_optimizer
UPDATE_FN = mappo_update_on_policy
ROLLOUT_FN = collect_rollout
ENV_KWARGS_LIST = [
    {"name": "ChangeShapeToSphere0p9", "object_type": "sphere", "object_scale": 0.9},
    {"name": "ChangeShapeToCube0p7", "object_scale": 0.7},
    {"name": "ChangeShapeToCylinder0p9", "object_type": "cylinder", "object_scale": 0.9},
    {"name": "ChangeShapeToSphere0p8", "object_type": "sphere", "object_scale": 0.8},
    {"name": "ChangeShapeToCylinder0p8", "object_type": "cylinder", "object_scale": 0.8},
    {"name": "ChangeShapeToSphere0p9", "object_type": "sphere", "object_scale": 0.9},
    {"name": "ChangeShapeToCube0p7", "object_scale": 0.7},
    {"name": "ChangeShapeToCylinder0p9", "object_type": "cylinder", "object_scale": 0.9},
    {"name": "ChangeShapeToSphere0p8", "object_type": "sphere", "object_scale": 0.8},
    {"name": "ChangeShapeToCylinder0p8", "object_type": "cylinder", "object_scale": 0.8},
]


def build_task_dir_name(args):
    base_task_name = args.ckpt_task_name if args.ckpt_task_name is not None else args.task_name
    if args.baseline:
        return f"{base_task_name}_online_baseline"
    if args.not_train_aggregator:
        return f"{base_task_name}_online_wo_ag"
    return f"{base_task_name}_online"


def resolve_ckpt_dir(args):
    task_dir = os.path.join(
        args.save_dir,
        build_task_dir_name(args),
        ALGO_NAME,
        args.robot_name,
        MODEL_NAME,
    )
    os.makedirs(task_dir, exist_ok=True)

    fresh_run_dir = os.path.join(task_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    resume_dir = args.resume_dir if args.resume_dir is not None else fresh_run_dir
    root_dir = resume_dir
    os.makedirs(root_dir, exist_ok=True)
    return {
        "task_dir": task_dir,
        "root_dir": root_dir,
        "log_dir": os.path.join(resume_dir, "tb"),
        "video_dir": os.path.join(resume_dir, "videos"),
        "latest_agent": os.path.join(resume_dir, "latest_agent.pt"),
        "latest_opt": os.path.join(resume_dir, "latest_opt.pt"),
        "best_agent": os.path.join(resume_dir, "best_agent.pt"),
        "metrics": os.path.join(resume_dir, "metrics.json"),
    }


def save_checkpoint(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def build_base_env_kwargs(args):
    env_kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "rgb_array",
        "max_episode_steps": args.max_episode_steps,
        "sim_backend": "physx_cuda",
    }
    if getattr(args, "enable_mixed_train_envs", False):
        env_kwargs.update(
            {
                "enable_mixed_domain_randomization": True,
                "mixed_domain_randomization_clear_ratio": args.mixed_env_clean_ratio,
                "mixed_domain_randomization_mild_ratio": args.mixed_env_mild_ratio,
                "mixed_domain_randomization_hard_ratio": args.mixed_env_hard_ratio,
            }
        )
    return env_kwargs


def make_envs_for_env_kwargs(args, ckpt, env_kwargs_update_id, base_env_kwargs):
    tmp_env_kwargs = base_env_kwargs.copy()
    env_kwargs_update = ENV_KWARGS_LIST[env_kwargs_update_id].copy()
    env_name = env_kwargs_update.pop("name")
    tmp_env_kwargs.update(**env_kwargs_update)

    env_kwargs_for_eval = tmp_env_kwargs.copy()
    env_kwargs_for_eval.pop("sim_backend")
    env_kwargs_for_eval["render_mode"] = "rgb_array"
    tmp_env_kwargs["render_mode"] = "none"

    eval_video_root = f"{ckpt['video_dir']}_eval" if args.evaluate_mode else f"{ckpt['video_dir']}_train"
    eval_envs = make_eval_envs(
        env_id=args.task_name,
        num_envs=args.num_eval_envs,
        sim_backend="gpu",
        env_kwargs=env_kwargs_for_eval,
        video_dir=os.path.join(eval_video_root, env_name),
    )
    eval_envs.reset(seed=args.seed)

    envs = gym.make(args.task_name, num_envs=args.num_envs, **tmp_env_kwargs)
    envs = ManiSkillVectorEnv(
        envs,
        args.num_envs,
        ignore_terminations=args.ignore_partial_reset,
        record_metrics=True,
    )
    return envs, eval_envs, env_name


def initialize_client_graph(clients, agent, load_dir=None, ckpt_prefix="best_ag_"):
    agent_names = list(clients.keys())
    if load_dir is not None:
        for name in agent_names:
            ag_path = os.path.join(load_dir, f"{ckpt_prefix}{name}.pt")
            if os.path.exists(ag_path):
                clients[name].load_feature_aggregators(ag_path)
                print(f"[Checkpoint] loaded feature aggregators for {name}: {ag_path}")
            else:
                print(f"[Checkpoint] missing feature aggregators for {name}: {ag_path}")

    client_infos = {name: clients[name].before_training_start(agent) for name in agent_names}
    for recv_name, client_recv in clients.items():
        for send_name, client_send_info in client_infos.items():
            if recv_name != send_name:
                client_recv.add_feature_aggregator(send_name, client_send_info)


def add_aggregator_param_groups(args, clients, optimizer, training_feature_aggregator_modules):
    for client_name, client in clients.items():
        feature_aggregators_parameters = client.get_feature_aggregators_parameters()
        for remote_name, fap in feature_aggregators_parameters.items():
            group_name = f"{client_name}_from_{remote_name}"
            if group_name in training_feature_aggregator_modules:
                continue
            fap_list = list(fap)
            if len(fap_list) == 0:
                continue
            for parameter in fap_list:
                parameter.requires_grad = True
            optimizer.add_param_group(
                {
                    "params": fap_list,
                    "lr": args.feature_aggregator_learning_rate,
                    "eps": 1e-5,
                    "group_name": "feature_aggregator",
                }
            )
            training_feature_aggregator_modules.add(group_name)


def build_agent(args, infos):
    return AGENT_CLS(
        **infos,
        model_dir=args.model_dir,
        normalize_state=args.normalize_state,
        freeze_vla_backbone=args.freeze_vla_backbone,
        critic_hidden_dim=args.critic_hidden_dim,
        attention_implementation=args.attn_implementation,
        image_size=args.image_size,
        use_vla_lora=args.use_vla_lora,
        use_vision_lora=args.use_vision_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_vision_backbone=args.train_vision_backbone,
        vision_token_pool_size=args.vision_token_pool_size,
        policy_mode=args.policy_mode,
        tiny_hidden_dim=args.tiny_hidden_dim,
        tiny_vision_layers=args.tiny_vision_layers,
        tiny_decoder_layers=args.tiny_decoder_layers,
        tiny_attention_heads=args.tiny_attention_heads,
        tiny_patch_size=args.tiny_patch_size,
        tiny_ffn_mult=args.tiny_ffn_mult,
        tiny_num_action_bins=args.tiny_num_action_bins,
        tiny_prompt_length=args.tiny_prompt_length,
    )


def main(args):
    torch.backends.cudnn.deterministic = not args.ignore_torch_deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator(mixed_precision="bf16" if args.use_amp else "no")
    device = accelerator.device

    ckpt = resolve_ckpt_dir(args)
    step_infos = get_step_infos(args)
    infos = get_agent_info(args)
    agent_names = infos["agent_names"]
    collate_fn = make_collate_fn(agent_names)

    writer = None
    agent = build_agent(args, infos)
    if getattr(args, "full_kl_coef", 0.0) > 0 or getattr(args, "log_full_kl", False):
        if accelerator.is_main_process:
            print(
                "[Config] mixed_tiny_vla_smolvla disables full token-distribution KL; "
                "force full_kl_coef=0 and log_full_kl=False"
            )
        args.full_kl_coef = 0.0
        args.log_full_kl = False

    base_param_groups = [
        {
            "params": [p for p in agent.vla_actor.vla.parameters() if p.requires_grad],
            "lr": args.backbone_learning_rate,
            "group_name": "vla_actor_vla",
        },
        {
            "params": [p for p in agent.vla_actor.state_projector.parameters() if p.requires_grad],
            "lr": args.state_learning_rate,
            "group_name": "vla_actor_state_projector",
        },
        {
            "params": [p for p in agent.vla_actor.context_projector.parameters() if p.requires_grad]
            + [p for p in agent.vla_actor.actor_head.parameters() if p.requires_grad],
            "lr": args.head_learning_rate,
            "group_name": "vla_actor_heads",
        },
        {
            "params": [p for p in agent.smolvla_actor.vla.parameters() if p.requires_grad],
            "lr": args.backbone_learning_rate,
            "group_name": "smolvla_actor_vla",
        },
        {
            "params": [p for p in agent.smolvla_actor.state_projector.parameters() if p.requires_grad],
            "lr": args.state_learning_rate,
            "group_name": "smolvla_actor_state_projector",
        },
        {
            "params": [p for p in agent.smolvla_actor.context_projector.parameters() if p.requires_grad]
            + [p for p in agent.smolvla_actor.action_mean_head.parameters() if p.requires_grad]
            + [p for p in agent.smolvla_actor.log_std_head.parameters() if p.requires_grad],
            "lr": args.head_learning_rate,
            "group_name": "smolvla_actor_heads",
        },
        {
            "params": [p for p in agent.critic_state_encoder.parameters() if p.requires_grad],
            "lr": args.value_head_learning_rate,
            "group_name": "critic_state_encoder",
        },
        {
            "params": [p for p in agent.critic_visual_encoder.parameters() if p.requires_grad],
            "lr": args.value_head_learning_rate,
            "group_name": "critic_visual_encoder",
        },
        {
            "params": [p for p in agent.critic.parameters() if p.requires_grad],
            "lr": args.value_head_learning_rate,
            "group_name": "critic",
        },
    ]
    base_param_groups = [group for group in base_param_groups if group["params"]]

    head_trainable_parameters = []
    encoder_trainable_parameters = []
    critic_trainable_parameters = []
    for group in base_param_groups:
        group_name = group["group_name"]
        params = list(group["params"])
        if group_name in {"vla_actor_heads", "smolvla_actor_heads"}:
            head_trainable_parameters.extend(params)
        elif group_name in {
            "vla_actor_vla",
            "vla_actor_state_projector",
            "smolvla_actor_vla",
            "smolvla_actor_state_projector",
        }:
            encoder_trainable_parameters.extend(params)
        elif group_name in {"critic_state_encoder", "critic_visual_encoder", "critic"}:
            critic_trainable_parameters.extend(params)

    resume_info = maybe_load_initial_agent(args, ckpt, agent)
    global_steps = 0
    latest_opt_payload = None
    if os.path.exists(ckpt["latest_opt"]):
        latest_opt_payload = torch.load(ckpt["latest_opt"], map_location="cpu")
        if resume_info["resumed"] and resume_info["used_best_agent"]:
            inferred_best_step = resume_info["best_step"]
            if inferred_best_step is not None:
                global_steps = int(inferred_best_step)
            else:
                global_steps = int(latest_opt_payload["step"])
                print(
                    f"[Resume] best-agent step could not be inferred; "
                    f"fall back to latest optimizer step={global_steps}"
                )

    clients = None if args.baseline else build_clients(args, agent, infos, agent_names, device)
    if clients is not None and not args.evaluate_mode:
        if args.resume_dir is not None:
            ckpt_prefix = "best_ag_" if args.resume_use_best_agent else "latest_ag_"
            initialize_client_graph(clients, agent, load_dir=ckpt["root_dir"], ckpt_prefix=ckpt_prefix)
        elif args.init_agent_path:
            initialize_client_graph(
                clients,
                agent,
                load_dir=os.path.dirname(args.init_agent_path),
                ckpt_prefix=infer_aggregator_ckpt_prefix(args.init_agent_path),
            )
        else:
            initialize_client_graph(clients, agent)

    optimizer = OPTIMIZER_FN(args, agent)
    training_feature_aggregator_modules = set()
    if clients is not None:
        add_aggregator_param_groups(args, clients, optimizer, training_feature_aggregator_modules)

    if latest_opt_payload is not None and not (resume_info["resumed"] and resume_info["used_best_agent"]):
        optimizer.load_state_dict(latest_opt_payload["opt"])
        global_steps = int(latest_opt_payload["step"])

    if accelerator.is_main_process:
        writer = SummaryWriter(ckpt["log_dir"], purge_step=global_steps)
        print(f"[TensorBoard] Logging to {ckpt['log_dir']}")

    agent, optimizer = accelerator.prepare(agent, optimizer)

    base_env_kwargs = build_base_env_kwargs(args)
    continual_env_schedule = build_continual_env_schedule(args, ENV_KWARGS_LIST)
    current_env_index = 0
    training_start_time = time.monotonic()
    training_compute_seconds = 0.0
    envs, eval_envs, current_env_name = make_envs_for_env_kwargs(args, ckpt, current_env_index, base_env_kwargs)
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    def maybe_switch_envs():
        nonlocal envs, eval_envs, next_obs, next_done, current_env_index, current_env_name
        if continual_env_schedule is None:
            return False, False, None
        elapsed_minutes = (time.monotonic() - training_start_time) / 60.0
        scheduled_env_index = bisect.bisect_right(
            continual_env_schedule.change_time_points,
            elapsed_minutes,
        )
        if scheduled_env_index >= len(continual_env_schedule.env_kwarg_list):
            if args.max_time is not None:
                return False, False, elapsed_minutes
            return False, True, elapsed_minutes
        if scheduled_env_index == current_env_index:
            return False, False, elapsed_minutes

        previous_env_name = current_env_name
        current_env_index = scheduled_env_index
        close_envs(envs, eval_envs)
        envs = None
        eval_envs = None
        clear_torch_cuda_cache()
        envs, eval_envs, current_env_name = make_envs_for_env_kwargs(
            args,
            ckpt,
            current_env_index,
            base_env_kwargs,
        )
        next_obs, _ = envs.reset(seed=args.seed)
        next_done = torch.zeros(args.num_envs, device=device)
        print(
            f"[OnlineRL] switch env from {previous_env_name} to {current_env_name} "
            f"at elapsed={elapsed_minutes:.2f} minutes"
        )
        return True, False, elapsed_minutes

    if args.evaluate_mode:
        if args.eval_agent_dir is None:
            raise ValueError("Please provide --eval-agent-dir for evaluation mode")
        if clients is not None:
            initialize_client_graph(clients, accelerator.unwrap_model(agent), load_dir=args.eval_agent_dir, ckpt_prefix="best_ag_")
            set_client_feature_aggregator_requires_grad(clients, False)
            for client in clients.values():
                client.use_eval_feature_selector_strategy()

        unwrapped_agent = accelerator.unwrap_model(agent)
        unwrapped_agent.load_checkpoint_state_dict(
            torch.load(os.path.join(args.eval_agent_dir, "best_agent.pt"), map_location="cpu")
        )
        unwrapped_agent.eval()
        eval_metrics = evaluate(
            n=args.eval_episodes,
            sample_fn=make_sample_fn(agent_names, unwrapped_agent, deterministic=True),
            eval_envs=eval_envs,
        )
        payload = {k: float(v.mean()) for k, v in eval_metrics.items()}
        close_envs(envs, eval_envs)
        envs = None
        eval_envs = None
        clear_torch_cuda_cache()
        if accelerator.is_main_process:
            dump_json(os.path.join(ckpt["root_dir"], "eval_metrics.json"), payload)
        return

    metrics_log = load_json(ckpt["metrics"]) if os.path.exists(ckpt["metrics"]) else []
    _, inferred_best_score = infer_best_step_from_metrics(ckpt["metrics"])
    best_score = -1.0 if inferred_best_score is None else float(inferred_best_score)
    start_time = time.time()
    last_save_skip = args.resume_dir is not None

    if args.minibatch_size == 0:
        args.minibatch_size = step_infos["rollot_steps"] // args.num_minibatch // args.grad_accum_steps

    pbar = tqdm(total=step_infos["total_steps"], initial=global_steps, ascii=True)
    agent.freeze_state_stats()

    while global_steps < step_infos["total_steps"]:
        switched_env, should_stop_for_schedule, elapsed_minutes = maybe_switch_envs()
        if writer is not None and elapsed_minutes is not None:
            writer.add_scalar("time/elapsed_minutes", elapsed_minutes, global_steps)
            writer.add_scalar("continual/current_env_index", current_env_index, global_steps)
        if should_stop_for_schedule:
            print(f"[OnlineRL] continual schedule finished at elapsed={elapsed_minutes:.2f} minutes")
            break
        if args.max_time is not None:
            elapsed_minutes = training_compute_seconds / 60.0
            if elapsed_minutes >= args.max_time:
                print(
                    f"[OnlineRL] reached max_time={args.max_time} minutes "
                    "of rollout/update time"
                )
                break
        if switched_env:
            print(f"[OnlineRL] active env={current_env_name}")

        critic_only = global_steps < step_infos["critic_warmup_steps"]
        set_optimizer_phase_lrs(args, optimizer, critic_only=critic_only, train_actor_base=True)
        set_requires_grad(head_trainable_parameters, not critic_only)
        set_requires_grad(encoder_trainable_parameters, not critic_only)
        set_requires_grad(critic_trainable_parameters, True)
        if clients is not None and args.not_train_aggregator:
            rollout_aggregator_params = []
            for client in clients.values():
                for aggregator in iter_client_feature_aggregators(client):
                    rollout_aggregator_params.extend(list(aggregator.module.parameters()))
            set_requires_grad(rollout_aggregator_params, False)

        unwrapped_agent = accelerator.unwrap_model(agent)
        unwrapped_agent.eval()
        if clients is not None:
            for client in clients.values():
                client.eval()

        if not last_save_skip and global_steps % step_infos["save_interval_steps"] == 0 and accelerator.is_main_process:
            if clients is not None:
                set_client_feature_aggregator_requires_grad(clients, False)
                for client in clients.values():
                    client.use_eval_feature_selector_strategy()

            eval_metrics = evaluate(
                n=args.eval_episodes,
                sample_fn=make_sample_fn(agent_names, unwrapped_agent, deterministic=True),
                eval_envs=eval_envs,
            )
            if clients is not None:
                for client in clients.values():
                    client.use_train_feature_selector_strategy()
            for key, value in eval_metrics.items():
                mean = value.mean()
                writer.add_scalar(f"eval/{key}", mean, global_steps)
                print(f"eval_{key}_mean={mean}")

            score = eval_metrics.get("success_rate", eval_metrics[list(eval_metrics.keys())[0]]).mean()
            pbar.set_postfix(eval_score=score, env=current_env_name)
            metrics_log.append(
                {
                    "step": global_steps,
                    "score": float(score),
                    "env": current_env_name,
                    "elapsed_minutes": training_compute_seconds / 60.0,
                }
            )
            dump_json(ckpt["metrics"], metrics_log)
            if score >= best_score:
                best_score = score
                print(f"[Eval] New best score={score:.3f}, env={current_env_name}")

        last_save_skip = False

        if clients is not None:
            for client in clients.values():
                client.train()
        compute_start_time = time.monotonic()
        rollout = ROLLOUT_FN(
            args=args,
            agent=agent,
            collate_fn=collate_fn,
            envs=envs,
            next_obs=next_obs,
            next_done=next_done,
            accelerator=accelerator,
            writer=writer,
            global_step=global_steps,
            clients=clients,
        )

        communication_snapshots = None
        if len(rollout) == 11:
            (
                obs_buf,
                action_bin_buf,
                logp_buf,
                rew_buf,
                done_buf,
                val_buf,
                final_val_buf,
                next_obs,
                next_done,
                old_token_logits_buf,
                communication_snapshots,
            ) = rollout
        elif len(rollout) == 10:
            (
                obs_buf,
                action_bin_buf,
                logp_buf,
                rew_buf,
                done_buf,
                val_buf,
                final_val_buf,
                next_obs,
                next_done,
                old_token_logits_buf,
            ) = rollout
        else:
            (
                obs_buf,
                action_bin_buf,
                logp_buf,
                rew_buf,
                done_buf,
                val_buf,
                final_val_buf,
                next_obs,
                next_done,
            ) = rollout
            old_token_logits_buf = None

        adv_buf, ret_buf = compute_gae(
            rew_buf,
            done_buf,
            val_buf,
            final_val_buf,
            next_obs,
            next_done,
            agent,
            collate_fn,
            args,
            accelerator,
        )

        data = (
            obs_buf.reshape((-1,)),
            action_bin_buf.reshape((-1,)),
            logp_buf.reshape((-1,)),
            adv_buf.reshape(-1),
            ret_buf.reshape(-1),
            val_buf.reshape(-1),
        )
        if old_token_logits_buf is not None:
            data = data + (old_token_logits_buf.reshape((-1,)),)
        if communication_snapshots is not None:
            flat_communication_snapshots = []
            num_envs = done_buf.shape[1]
            for step_snapshot in communication_snapshots:
                flat_communication_snapshots.extend([step_snapshot] * num_envs)
            data = data + (flat_communication_snapshots,)

        active_base_training = len(base_param_groups) > 0
        all_aggregator_params = []
        has_active_aggregator = False
        if clients is not None:
            for client in clients.values():
                for aggregator in iter_client_feature_aggregators(client):
                    for parameter in aggregator.module.parameters():
                        all_aggregator_params.append(parameter)
                    has_active_aggregator = has_active_aggregator or aggregator.remote_features is not None

        agent.eval()
        if args.baseline:
            stats = UPDATE_FN(
                args,
                agent,
                optimizer,
                data,
                collate_fn,
                accelerator,
                "baseline_mappo",
                writer,
                -1,
            )
        elif args.joint_policy_aggregator_update:
            set_requires_grad(all_aggregator_params, not args.not_train_aggregator)
            if has_active_aggregator and len(all_aggregator_params) > 0 and not args.not_train_aggregator:
                stats = mappo_update_on_policy_ag(
                    args,
                    agent,
                    optimizer,
                    data,
                    collate_fn,
                    accelerator,
                    "joint_mappo_ag",
                    clients,
                    writer,
                    -1,
                    target_kl_override=args.joint_target_kl,
                )
            elif active_base_training:
                print("[Aggregation] no active remote messages, fallback to base PPO update")
                stats = UPDATE_FN(
                    args,
                    agent,
                    optimizer,
                    data,
                    collate_fn,
                    accelerator,
                    "mappo",
                    writer,
                    -1,
                )
            else:
                stats = {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0,
                    "approx_kl": 0.0,
                    "clip_frac": 0.0,
                    "old_approx_kl": 0.0,
                    "full_kl": 0.0,
                    "argmax_change_frac": 0.0,
                }
        else:
            set_requires_grad(all_aggregator_params, False)
            stats = UPDATE_FN(
                args,
                agent,
                optimizer,
                data,
                collate_fn,
                accelerator,
                "mappo",
                writer,
                -1,
            )
            if has_active_aggregator and len(all_aggregator_params) > 0 and not args.not_train_aggregator:
                set_requires_grad(head_trainable_parameters, False)
                set_requires_grad(encoder_trainable_parameters, False)
                set_requires_grad(all_aggregator_params, True)
                stats = mappo_update_on_policy_ag(
                    args,
                    agent,
                    optimizer,
                    data,
                    collate_fn,
                    accelerator,
                    "mappo_ag",
                    clients,
                    writer,
                    -1,
                    target_kl_override=args.aggregator_target_kl,
                )
                set_requires_grad(head_trainable_parameters, not critic_only)
                set_requires_grad(encoder_trainable_parameters, not critic_only)

        training_compute_seconds += time.monotonic() - compute_start_time

        if clients is not None:
            ag_data_infos = {}
            for client_name, client in clients.items():
                ag_data_infos[client_name] = client.export_feature_and_action()
            log_uploaded_q_task_stats(
                writer,
                ag_data_infos,
                global_steps,
                enable_histograms=not args.disable_ag_debug_histograms,
            )
            log_effective_transmission_stats(
                writer,
                clients,
                global_steps,
            )
            log_agent_feature_cosine_similarity(writer, ag_data_infos, global_steps)
            for recv_name, client_recv in clients.items():
                for send_name, ag_data in ag_data_infos.items():
                    if recv_name != send_name:
                        client_recv.receive_feature_and_action(send_name, ag_data)

            if not args.not_train_aggregator:
                add_aggregator_param_groups(args, clients, optimizer, training_feature_aggregator_modules)
            for client_name, client in clients.items():
                feature_aggregators_gate_g = client.debug_feature_aggregators()
                gate_g_strs = []
                for remote_name, gate_info in feature_aggregators_gate_g.items():
                    for stream_name, gate_g in gate_info.items():
                        if gate_g is None:
                            continue
                        metric_prefix = f"monitoring/aggregation/gate_g/{client_name}/from_{remote_name}/{stream_name}"
                        if not args.disable_ag_debug_histograms:
                            writer.add_histogram(metric_prefix, gate_g, global_steps)
                        writer.add_scalar(f"{metric_prefix}_mean", gate_g.mean(), global_steps)
                        writer.add_scalar(f"{metric_prefix}_std", gate_g.std(), global_steps)
                        gate_g_strs.append(f"{remote_name}.{stream_name}={gate_g.mean().item():.4f}")
                if gate_g_strs:
                    print(f'Client {client_name} gate_g_mean: {", ".join(gate_g_strs)}')
            log_feature_aggregator_agent_cosine_similarity(writer, clients, global_steps)

        agent.unfreeze_state_stats()
        agent.update_state_stats(obs_buf.reshape((-1,)), update_actor=False, update_critic=True)
        agent.update_state_stats(next_obs, update_actor=False, update_critic=True)
        agent.freeze_state_stats()

        global_steps += step_infos["rollot_steps"]
        pbar.update(step_infos["rollot_steps"])

        if accelerator.is_main_process:
            y_pred = val_buf.flatten(0, 1).cpu().numpy()
            y_true = ret_buf.flatten(0, 1).cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            sps = global_steps / max(time.time() - start_time, 1e-6)
            writer.add_scalar("charts/SPS", sps, global_steps)
            writer.add_scalar("loss/policy", stats["policy_loss"], global_steps)
            writer.add_scalar("loss/value", stats["value_loss"], global_steps)
            writer.add_scalar("loss/entropy", stats["entropy"], global_steps)
            writer.add_scalar("loss/approx_kl", stats["approx_kl"], global_steps)
            writer.add_scalar("loss/old_approx_kl", stats["old_approx_kl"], global_steps)
            writer.add_scalar("loss/clip_frac", stats["clip_frac"], global_steps)
            writer.add_scalar("loss/full_kl", stats.get("full_kl", 0.0), global_steps)
            writer.add_scalar("loss/argmax_change_frac", stats.get("argmax_change_frac", 0.0), global_steps)
            writer.add_scalar("loss/explained_var", explained_var, global_steps)

    close_envs(envs, eval_envs)
    envs = None
    eval_envs = None
    clear_torch_cuda_cache()
    if accelerator.is_main_process:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ckpt-task-name", type=str, default=None)
    parser.add_argument(
        "--env-change-time-points",
        type=str,
        default="[31,61,91,121,151,181,211,241,271,301]",
    )
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--actor-hook-learning-rate", type=float, default=3e-6)
    parser.add_argument("--feature-aggregator-learning-rate", type=float, default=3e-5)
    parser.add_argument("--aggregator-target-kl", type=float, default=2.0)
    parser.add_argument("--joint-target-kl", type=float, default=0.5)
    parser.add_argument("--feature-selector-alpha", type=float, default=0.2)
    parser.add_argument("--feature-selector-topk-trajectories", type=int, default=None)
    parser.add_argument("--feature-selector-temporal-pool-steps", type=int, default=None)
    parser.add_argument(
        "--feature-selector-strategy",
        type=str,
        default="topk_return",
        choices=["topk_return", "random", "return_span"],
    )
    parser.add_argument(
        "--eval-feature-selector-strategy",
        type=str,
        default=None,
        choices=["topk_return", "random", "return_span"],
    )
    parser.add_argument("--feature-aggregator-attention-num-heads", type=int, default=4)
    parser.add_argument(
        "--feature-aggregator-gate-type",
        type=str,
        default="two-layers",
        choices=["single-layer", "two-layers"],
    )
    parser.add_argument(
        "--feature-aggregator-gate-activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "silu", "tanh"],
    )
    parser.add_argument(
        "--feature-aggregator-norm-type",
        type=str,
        default="none",
        choices=["none", "layernorm"],
    )
    parser.add_argument("--feature-aggregator-feature-gate-open-max", type=float, default=0.25)
    parser.add_argument("--feature-aggregator-action-gate-open-max", type=float, default=0.10)
    parser.add_argument("--feature-aggregator-q-ret-weight", type=float, default=0.85)
    parser.add_argument("--feature-aggregator-q-attn-weight", type=float, default=0.15)
    parser.add_argument("--feature-aggregator-remote-dropout-prob", type=float, default=0.0)
    parser.add_argument("--feature-aggregator-remote-noise-std", type=float, default=0.0)
    parser.add_argument("--feature-aggregator-remote-stale-shift-max", type=int, default=0)
    parser.add_argument("--gate-reg-coef", type=float, default=0.0)
    parser.add_argument("--gate-target-mean", type=float, default=0.6)
    parser.add_argument("--gate-std-coef", type=float, default=0.0)
    parser.add_argument("--feature-gate-reg-coef", type=float, default=None)
    parser.add_argument("--feature-gate-target-mean", type=float, default=None)
    parser.add_argument("--feature-gate-std-coef", type=float, default=None)
    parser.add_argument("--action-gate-reg-coef", type=float, default=None)
    parser.add_argument("--action-gate-target-mean", type=float, default=None)
    parser.add_argument("--action-gate-std-coef", type=float, default=None)
    parser.add_argument("--feature-gate-quality-coef", type=float, default=0.0)
    parser.add_argument("--action-gate-quality-coef", type=float, default=0.0)
    parser.add_argument("--feature-consistency-coef", type=float, default=0.0)
    parser.add_argument("--action-consistency-coef", type=float, default=0.0)
    parser.add_argument("--feature-attn-entropy-coef", type=float, default=0.0)
    parser.add_argument("--action-attn-entropy-coef", type=float, default=0.0)
    parser.add_argument("--feature-attn-diversity-coef", type=float, default=0.0)
    parser.add_argument("--action-attn-diversity-coef", type=float, default=0.0)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--not-train-aggregator", action="store_true")
    parser.set_defaults(joint_policy_aggregator_update=True)
    parser.add_argument(
        "--joint-policy-aggregator-update",
        dest="joint_policy_aggregator_update",
        action="store_true",
    )
    parser.add_argument(
        "--separate-policy-aggregator-update",
        dest="joint_policy_aggregator_update",
        action="store_false",
    )
    parser.set_defaults(communication_replay=False)
    parser.add_argument("--communication-replay", dest="communication_replay", action="store_true")
    parser.add_argument("--no-communication-replay", dest="communication_replay", action="store_false")
    parser.add_argument("--enable-mixed-train-envs", action="store_true")
    parser.add_argument("--mixed-env-clean-ratio", type=float, default=0.5)
    parser.add_argument("--mixed-env-mild-ratio", type=float, default=0.3)
    parser.add_argument("--mixed-env-hard-ratio", type=float, default=0.2)
    parser.add_argument("--mixed-env-shard-size", type=int, default=1)
    parser.add_argument("--disable-ag-debug-histograms", action="store_true")
    extras, _ = parser.parse_known_args(sys.argv[1:])

    stripped_flags = {
        "--ckpt-task-name",
        "--env-change-time-points",
        "--max-time",
        "--eval-episodes",
        "--actor-hook-learning-rate",
        "--feature-aggregator-learning-rate",
        "--aggregator-target-kl",
        "--joint-target-kl",
        "--feature-selector-alpha",
        "--feature-selector-topk-trajectories",
        "--feature-selector-temporal-pool-steps",
        "--feature-selector-strategy",
        "--eval-feature-selector-strategy",
        "--feature-aggregator-attention-num-heads",
        "--feature-aggregator-gate-type",
        "--feature-aggregator-gate-activation",
        "--feature-aggregator-norm-type",
        "--feature-aggregator-feature-gate-open-max",
        "--feature-aggregator-action-gate-open-max",
        "--feature-aggregator-q-ret-weight",
        "--feature-aggregator-q-attn-weight",
        "--feature-aggregator-remote-dropout-prob",
        "--feature-aggregator-remote-noise-std",
        "--feature-aggregator-remote-stale-shift-max",
        "--gate-reg-coef",
        "--gate-target-mean",
        "--gate-std-coef",
        "--feature-gate-reg-coef",
        "--feature-gate-target-mean",
        "--feature-gate-std-coef",
        "--action-gate-reg-coef",
        "--action-gate-target-mean",
        "--action-gate-std-coef",
        "--feature-gate-quality-coef",
        "--action-gate-quality-coef",
        "--feature-consistency-coef",
        "--action-consistency-coef",
        "--feature-attn-entropy-coef",
        "--action-attn-entropy-coef",
        "--feature-attn-diversity-coef",
        "--action-attn-diversity-coef",
        "--baseline",
        "--not-train-aggregator",
        "--joint-policy-aggregator-update",
        "--separate-policy-aggregator-update",
        "--communication-replay",
        "--no-communication-replay",
        "--enable-mixed-train-envs",
        "--mixed-env-clean-ratio",
        "--mixed-env-mild-ratio",
        "--mixed-env-hard-ratio",
        "--mixed-env-shard-size",
        "--disable-ag-debug-histograms",
    }
    flags_without_values = {
        "--baseline",
        "--not-train-aggregator",
        "--joint-policy-aggregator-update",
        "--separate-policy-aggregator-update",
        "--communication-replay",
        "--no-communication-replay",
        "--enable-mixed-train-envs",
        "--disable-ag-debug-histograms",
    }

    cleaned_argv = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in stripped_flags:
            if arg not in flags_without_values:
                skip_next = True
            continue
        cleaned_argv.append(arg)

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *cleaned_argv]
        import train.vla_adapter_smolvla.multi_agents.two_robot_pick.mappo_pretrain as mappo_pretrain

        args = mappo_pretrain.parse_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(extras).items():
        setattr(args, key, value)
    args.model_backbone = "mixed_tiny_vla_smolvla"
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    return args


if __name__ == "__main__":
    main(parse_args())
