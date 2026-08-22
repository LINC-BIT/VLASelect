from dataclasses import dataclass
from typing import Dict, List, Callable, Any

import torch


# =========================================================
# SubTask
# =========================================================

@dataclass
class SubTask:

    id: int
    name: str

    # participating agents
    agents: List[str]

    # termination condition
    termination_fn: Callable

    # cooperative or not
    cooperative: bool

    # max steps
    max_episode_steps: int

    # num envs
    num_envs: int

    # device
    device: str

    # control mode
    control_mode: str = "pd_joint_delta_pos"

    def __post_init__(self):

        self.active_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device
        )

        self.step_count = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device
        )

    # -----------------------------------------------------

    def activate(self, env_ids: torch.Tensor):

        self.active_mask[env_ids] = True
        self.step_count[env_ids] = 0

    # -----------------------------------------------------

    def deactivate(self, env_ids: torch.Tensor):

        self.active_mask[env_ids] = False
        self.step_count[env_ids] = 0

    # -----------------------------------------------------

    def update(self):

        self.step_count[self.active_mask] += 1

    # -----------------------------------------------------

    def reach_step_limit(self):

        return (
            self.step_count >= self.max_episode_steps
        )

    # -----------------------------------------------------

    def is_terminated(self):

        env_done = self.termination_fn()

        return torch.logical_or(
            env_done,
            self.reach_step_limit()
        )


# =========================================================
# process_actions.py
# =========================================================

def process_actions(
    env,
    low_level_actions
):
    """
    Convert subtask-level actions into
    full ManiSkill multi-agent actions.

    Parameters
    ----------
    low_level_actions:

    {
        subtask_id:
            tensor[num_active_envs, act_dim]

    OR

        subtask_id:
        {
            agent_name:
                tensor[num_active_envs, act_dim]
        }
    }
    """

    global_actions = {}

    # -----------------------------------------------------
    # initialize zero actions
    # -----------------------------------------------------

    for agent in env.agent.agents:

        uid = agent.uid

        act_dim = agent.action_space.shape[0]

        global_actions[uid] = torch.zeros(
            (
                env.num_envs,
                act_dim
            ),
            dtype=torch.float32,
            device=env.device
        )

    # -----------------------------------------------------
    # scatter subtask actions
    # -----------------------------------------------------

    for subtask_id, action in \
            low_level_actions.items():

        # envs executing this subtask
        env_mask = (
            env.current_subtask
            == subtask_id
        )

        if env_mask.sum() == 0:
            continue

        subtask = env.subtasks[
            subtask_id
        ]

        # cooperative subtask
        if subtask.cooperative:

            for agent_name, act in \
                    action.items():

                global_actions[
                    agent_name
                ][env_mask] = act

        # single-agent subtask
        else:

            agent_name = subtask.agents[0]

            global_actions[
                agent_name
            ][env_mask] = action

    return global_actions