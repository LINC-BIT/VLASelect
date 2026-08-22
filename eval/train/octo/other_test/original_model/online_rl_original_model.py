from __future__ import annotations

from pathlib import Path

import train.octo.ppo_gen.online_rl as reference
from train.octo.other_test.common import (
    build_original_agent,
    count_parameters,
    count_trainable_parameters,
    dump_run_metadata,
)


def build_agent(args, device, env_kwargs):
    agent = build_original_agent(reference, args, device, env_kwargs)
    print(
        f"Loaded original FBS agent: total_params={count_parameters(agent)}, "
        f"trainable_params={count_trainable_parameters(agent)}"
    )
    return agent


def copy_run_metadata(run_name, args):
    dump_run_metadata(
        run_name,
        args,
        Path(__file__),
        extra_files=[
            Path(reference.__file__),
            Path("/home/Maniskill/train/octo/other_test/common.py"),
        ],
    )


reference.build_agent = build_agent
reference.copy_run_metadata = copy_run_metadata


if __name__ == "__main__":
    reference.main()
