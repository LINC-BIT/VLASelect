from collections import defaultdict
from typing import Callable
import numpy as np
import torch
from tqdm import tqdm

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
        first_success_step = None
        eps_count = 0
        pbar = tqdm(desc=f'Episode {eps_count}')
        while eps_count < n:
            action = sample_fn(obs)
            obs, _, _, truncated, info = eval_envs.step(action)
            pbar.update(1)
            if "episode" in info and "success_once" in info["episode"] and "episode_len" in info["episode"]:
                success_once = info["episode"]["success_once"]
                episode_len = info["episode"]["episode_len"]
                if first_success_step is None:
                    first_success_step = torch.zeros_like(episode_len, dtype=torch.float32)
                new_success = (first_success_step <= 0) & success_once
                first_success_step = torch.where(
                    new_success,
                    episode_len.float(),
                    first_success_step,
                )
            # note as there are no partial resets, truncated is True for all environments at the same time
            if truncated.any():
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                    if first_success_step is not None:
                        eval_metrics["success_first_step"].append(first_success_step.cpu().numpy())
                else:
                    for env_i, final_info in enumerate(info["final_info"]):
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                        if first_success_step is not None:
                            eval_metrics["success_first_step"].append(
                                np.asarray(first_success_step[env_i].cpu())
                            )
                eps_count += eval_envs.num_envs
                first_success_step = None
                pbar.reset()
                pbar.set_description(desc=f'Episode {eps_count}')
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
