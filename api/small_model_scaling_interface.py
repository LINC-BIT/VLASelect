"""Small-policy scaling hooks for the unified online-RL runner.

The default implementation is the static FBS scaling flow that used to live in
``unified_online_rl.py``.  Model authors can subclass this class and override
``after_small_model_scaling`` to add work such as distillation while retaining the
sampling, pruning, and regeneration semantics of the reference implementation.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent
VENDOR_ROOT = THIS_DIR / "vendor"
for candidate in (THIS_DIR, VENDOR_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


class SmallModelScalingInterface:
    """Default small-model generator and extension point for custom generators.

    The interface deliberately receives the runner's adapter and reference API as
    dependencies.  This keeps it independent from global state in the runner and lets
    one implementation work with both VLA-Adapter and TinyVLA.
    """

    def collect_sample_for_small_model_scaling(
        self,
        args: Any,
        *,
        large_agent: nn.Module,
        small_agent: Optional[nn.Module],
        eval_envs: Any,
        device: torch.device,
        adapter: Any,
        reference_api: Any,
    ) -> Dict[str, Any]:
        strategy = args.small_model_scaling_strategy
        if strategy in {"target-batch", "target-single"}:
            obs, _ = eval_envs.reset()
            rgbs = adapter.extract_rgb_batch_from_obs(obs)
            states = adapter.extract_state_batch_from_obs(obs)
            generation_agent, _ = self.resolve_generation_policy_agent(
                args, large_agent=large_agent, small_agent=small_agent
            )
            _, _, _, _, action_bins = reference_api.batched_get_action_and_value_no_grad(
                generation_agent,
                rgbs,
                states,
                micro_batch_size=args.eval_micro_batch_size,
                deterministic=True,
            )
            if strategy == "target-single":
                return {
                    "rgbs": rgbs[:1].clone(),
                    "states": states[:1].copy(),
                    "action_bins": action_bins[:1].to(device=device, dtype=torch.long),
                }
            return {
                "rgbs": rgbs.clone(),
                "states": states.copy(),
                "action_bins": action_bins.to(device=device, dtype=torch.long),
            }

        if strategy == "target-single-traj":
            generation_agent, generation_policy = self.resolve_generation_policy_agent(
                args, large_agent=large_agent, small_agent=small_agent
            )
            if generation_policy == "better":
                comparison_seed = np.random.randint(0, 2**31 - 1)
                large_sample, large_return = self.collect_best_return_trajectory_sample(
                    large_agent,
                    eval_envs,
                    args.num_steps,
                    args.eval_micro_batch_size,
                    device,
                    adapter=adapter,
                    reference_api=reference_api,
                    reset_seed=int(comparison_seed),
                )
                small_sample, small_return = self.collect_best_return_trajectory_sample(
                    small_agent,
                    eval_envs,
                    args.num_steps,
                    args.eval_micro_batch_size,
                    device,
                    adapter=adapter,
                    reference_api=reference_api,
                    reset_seed=int(comparison_seed),
                )
                return small_sample if small_return >= large_return else large_sample
            sample, _ = self.collect_best_return_trajectory_sample(
                generation_agent,
                eval_envs,
                args.num_steps,
                args.eval_micro_batch_size,
                device,
                adapter=adapter,
                reference_api=reference_api,
            )
            return sample

        raise NotImplementedError(f"Unknown small_model_scaling_strategy: {strategy}")

    @staticmethod
    def resolve_generation_policy_agent(args: Any, *, large_agent: nn.Module, small_agent: Optional[nn.Module]):
        if args.small_model_scaling_policy == "small":
            return (small_agent if small_agent is not None else large_agent), ("small" if small_agent is not None else "large")
        if args.small_model_scaling_policy == "large":
            return large_agent, "large"
        if args.small_model_scaling_policy == "better":
            if small_agent is None:
                return large_agent, "large"
            return None, "better"
        raise NotImplementedError(f"Unknown small_model_scaling_policy: {args.small_model_scaling_policy}")

    def collect_best_return_trajectory_sample(
        self,
        policy: nn.Module,
        eval_envs: Any,
        num_steps: int,
        micro_batch_size: int,
        device: torch.device,
        *,
        adapter: Any,
        reference_api: Any,
        reset_seed: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], float]:
        was_training = policy.training
        policy.eval()
        obs, _ = eval_envs.reset() if reset_seed is None else eval_envs.reset(seed=reset_seed)
        num_envs = len(adapter.extract_rgb_batch_from_obs(obs))
        running_trajectories: List[List[Dict[str, Any]]] = [[] for _ in range(num_envs)]
        running_returns = [0.0 for _ in range(num_envs)]
        finished_trajectories: List[Dict[str, Any]] = []

        with torch.no_grad():
            for _ in range(num_steps):
                rgbs = adapter.extract_rgb_batch_from_obs(obs)
                states = adapter.extract_state_batch_from_obs(obs)
                _, _, _, _, action_bins = reference_api.batched_get_action_and_value_no_grad(
                    policy,
                    rgbs,
                    states,
                    micro_batch_size=micro_batch_size,
                    deterministic=True,
                )
                actions = policy.bin_indices_to_env_actions(action_bins)
                for env_idx in range(num_envs):
                    running_trajectories[env_idx].append(
                        {
                            "rgbs": rgbs[env_idx].clone(),
                            "states": states[env_idx].copy(),
                            "action_bins": action_bins[env_idx].detach().cpu().clone(),
                        }
                    )
                obs, rewards, terminations, truncations, _ = eval_envs.step(actions)
                reward_values = torch.as_tensor(rewards).detach().cpu().view(-1)
                done_mask = torch.logical_or(
                    torch.as_tensor(terminations), torch.as_tensor(truncations)
                ).cpu().view(-1).bool()
                for env_idx in range(num_envs):
                    running_returns[env_idx] += float(reward_values[env_idx].item())
                    if done_mask[env_idx]:
                        finished_trajectories.append(
                            {"return": running_returns[env_idx], "steps": running_trajectories[env_idx]}
                        )
                        running_trajectories[env_idx] = []
                        running_returns[env_idx] = 0.0

        for env_idx in range(num_envs):
            if running_trajectories[env_idx]:
                finished_trajectories.append(
                    {"return": running_returns[env_idx], "steps": running_trajectories[env_idx]}
                )
        if was_training:
            policy.train()
        if not finished_trajectories:
            raise RuntimeError("Failed to collect any trajectory for small model scaling")
        best = max(finished_trajectories, key=lambda item: item["return"])
        steps = best["steps"]
        return {
            "rgbs": torch.stack([step["rgbs"] for step in steps], dim=0).to(dtype=torch.uint8),
            "states": np.stack([step["states"] for step in steps], axis=0).astype(np.float32),
            "action_bins": torch.stack([step["action_bins"] for step in steps], dim=0).to(
                device=device, dtype=torch.long
            ),
        }, float(best["return"])

    def should_regenerate_small_model_before_rollout(
        self,
        schedule: str,
        update: int,
        start_update: int,
        current_success_end: Optional[float],
        success_end_at_last_regeneration: Optional[float],
        update_at_last_regeneration: Optional[int],
    ) -> bool:
        if schedule == "once":
            return False
        if schedule == "before_per_rollout":
            return update > start_update
        threshold_prefix = "before_per_rollout_if_success_improv_is_larger_than_"
        if schedule.startswith(threshold_prefix):
            threshold = float(schedule[len(threshold_prefix) :])
            return (
                update > start_update
                and current_success_end is not None
                and success_end_at_last_regeneration is not None
                and current_success_end - success_end_at_last_regeneration > threshold
            )
        threshold_match = re.fullmatch(
            r"before_per_rollout_if_success_improv_less_than_([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)_for_(\d+)_iters",
            schedule,
        )
        if threshold_match is not None:
            threshold = float(threshold_match.group(1))
            num_iters = int(threshold_match.group(2))
            return (
                update > start_update
                and current_success_end is not None
                and success_end_at_last_regeneration is not None
                and update_at_last_regeneration is not None
                and update - update_at_last_regeneration >= num_iters
                and current_success_end - success_end_at_last_regeneration < threshold
            )
        raise NotImplementedError(f"Unknown small_model_regeneration_schedule: {schedule}")

    def maybe_regenerate_small_model_before_rollout(
        self,
        *,
        args: Any,
        update: int,
        start_update: int,
        current_success_end: Optional[float],
        success_end_at_last_regeneration: Optional[float],
        update_at_last_regeneration: Optional[int],
        large_agent: nn.Module,
        small_agent: nn.Module,
        current_pruning_info: Optional[dict],
        optimizer: optim.Optimizer,
        eval_envs: Any,
        device: torch.device,
        adapter: Any,
        reference_api: Any,
    ) -> Tuple[bool, Optional[dict]]:
        should_regenerate = self.should_regenerate_small_model_before_rollout(
            args.small_model_regeneration_schedule,
            update,
            start_update,
            current_success_end=current_success_end,
            success_end_at_last_regeneration=success_end_at_last_regeneration,
            update_at_last_regeneration=update_at_last_regeneration,
        )
        if not should_regenerate:
            return False, current_pruning_info
        print("[ours] regenerate small model before rollout")
        new_pruning_info = self.regenerate_small_model_in_place(
            large_agent=large_agent,
            small_agent=small_agent,
            current_pruning_info=current_pruning_info,
            optimizer=optimizer,
            args=args,
            eval_envs=eval_envs,
            device=device,
            adapter=adapter,
            reference_api=reference_api,
        )
        return True, new_pruning_info

    def on_environment_switch(
        self,
        *,
        args: Any,
        large_agent: nn.Module,
        small_agent: nn.Module,
        current_pruning_info: Optional[dict],
        optimizer: optim.Optimizer,
        eval_envs: Any,
        device: torch.device,
        adapter: Any,
        reference_api: Any,
    ) -> Tuple[bool, Optional[dict]]:
        """Optionally refresh the small model after a continual env switch."""
        del args, large_agent, small_agent, optimizer, eval_envs, device, adapter, reference_api
        return False, current_pruning_info

    def generate_initial_small_model(
        self,
        *,
        large_agent: nn.Module,
        args: Any,
        eval_envs: Any,
        device: torch.device,
        adapter: Any,
        reference_api: Any,
    ) -> Tuple[nn.Module, dict]:
        from train.vla_adapter_new.ours.generate_static_small_model import (
            generate_static_small_model_with_returning_pruning_info,
        )

        sample = self.collect_sample_for_small_model_scaling(
            args,
            large_agent=large_agent,
            small_agent=None,
            eval_envs=eval_envs,
            device=device,
            adapter=adapter,
            reference_api=reference_api,
        )
        small_agent, pruning_info = generate_static_small_model_with_returning_pruning_info(
            large_agent,
            sample_batch=sample,
            device=device,
            dtype=torch.bfloat16,
            verify=True,
        )
        self._prepare_generated_small_model(
            small_agent, args=args, device=device, adapter=adapter, initial_generation=True
        )
        self.after_small_model_scaling(
            large_agent=large_agent,
            small_agent=small_agent,
            sample_batch=sample,
            pruning_info=pruning_info,
            optimizer=None,
            args=args,
            device=device,
            adapter=adapter,
            reference_api=reference_api,
        )
        return small_agent, pruning_info

    def regenerate_small_model_in_place(
        self,
        *,
        large_agent: nn.Module,
        small_agent: nn.Module,
        current_pruning_info: Optional[dict],
        optimizer: optim.Optimizer,
        args: Any,
        eval_envs: Any,
        device: torch.device,
        adapter: Any,
        reference_api: Any,
    ) -> dict:
        from train.vla_adapter_new.ours.generate_static_small_model import (
            generate_static_small_model_with_returning_pruning_info,
            inherit_static_small_model_retained_channels,
        )

        sample = self.collect_sample_for_small_model_scaling(
            args,
            large_agent=large_agent,
            small_agent=small_agent,
            eval_envs=eval_envs,
            device=device,
            adapter=adapter,
            reference_api=reference_api,
        )
        regenerated_small_agent, new_pruning_info = generate_static_small_model_with_returning_pruning_info(
            large_agent,
            sample_batch=sample,
            device=device,
            dtype=torch.bfloat16,
            previous_pruning_info=current_pruning_info,
            regeneration_increment_ratio=args.small_model_regeneration_increment_ratio,
            verify=True,
        )
        self._prepare_generated_small_model(
            regenerated_small_agent, args=args, device=device, adapter=adapter, initial_generation=False
        )
        if args.small_model_regeneration_increment_ratio < 1.0 and current_pruning_info is not None:
            inherit_static_small_model_retained_channels(
                regenerated_small_agent,
                small_agent,
                new_pruning_info,
                current_pruning_info,
            )
        small_agent.load_state_dict(regenerated_small_agent.state_dict(), strict=True)
        if args.reset_optimizer_after_regeneration:
            optimizer.zero_grad(set_to_none=True)
            for parameter in small_agent.parameters():
                optimizer.state.pop(parameter, None)
        merge_stats = new_pruning_info.get("merge_stats", {})
        if merge_stats:
            replaced_ratios = []
            for layer_stats in merge_stats.values():
                merged_count = max(int(layer_stats.get("merged_count", 0)), 1)
                replaced_ratios.append(float(layer_stats.get("replaced_count", 0)) / merged_count)
            if replaced_ratios:
                print(
                    f"[regen] replaced_ratio avg={sum(replaced_ratios) / len(replaced_ratios):.4f} "
                    f"min={min(replaced_ratios):.4f} max={max(replaced_ratios):.4f}"
                )
        self.after_small_model_scaling(
            large_agent=large_agent,
            small_agent=small_agent,
            sample_batch=sample,
            pruning_info=new_pruning_info,
            optimizer=optimizer,
            args=args,
            device=device,
            adapter=adapter,
            reference_api=reference_api,
        )
        return new_pruning_info

    @staticmethod
    def _prepare_generated_small_model(
        policy: nn.Module,
        *,
        args: Any,
        device: torch.device,
        adapter: Any,
        initial_generation: bool,
    ) -> None:
        adapter.restore_policy_after_fbs(policy, device=device)
        adapter.configure_trainable_modules(
            policy,
            train_backbone=not args.freeze_vla_backbone
            and (args.backbone_warmup_updates <= 0 or not initial_generation),
        )
        policy.eval_micro_batch_size = args.eval_micro_batch_size

    def after_small_model_scaling(
        self,
        *,
        large_agent: nn.Module,
        small_agent: nn.Module,
        sample_batch: Dict[str, Any],
        pruning_info: dict,
        optimizer: Optional[optim.Optimizer],
        args: Any,
        device: torch.device,
        adapter: Any,
        reference_api: Any,
    ) -> None:
        """Customize a generated policy before the runner starts using it.

        ``optimizer`` is ``None`` for initial generation because the runner builds its
        training optimizer after this hook.  During regeneration it is the active PPO
        optimizer.  Custom generation hyperparameters should be stored on ``self`` at
        construction time rather than added to this method's runtime contract.
        """
        del large_agent, small_agent, sample_batch, pruning_info
        del optimizer, args, device, adapter, reference_api
