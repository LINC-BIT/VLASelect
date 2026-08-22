from __future__ import annotations

import os
from pathlib import Path

import train.octo.ppo_gen.online_rl as reference
from train.octo.other_test.common import build_peft_agent, dump_run_metadata


PEFT_RANK = int(os.environ.get("PEFT_RANK", "8"))
PEFT_ALPHA = float(os.environ.get("PEFT_ALPHA", "16"))
PEFT_DROPOUT = float(os.environ.get("PEFT_DROPOUT", "0.0"))


def build_agent(args, device, env_kwargs):
    print(
        f"Building PEFT agent on top of original FBS model: "
        f"rank={PEFT_RANK}, alpha={PEFT_ALPHA}, dropout={PEFT_DROPOUT}"
    )
    return build_peft_agent(
        reference,
        args,
        device,
        env_kwargs,
        rank=PEFT_RANK,
        alpha=PEFT_ALPHA,
        dropout=PEFT_DROPOUT,
    )


def copy_run_metadata(run_name, args):
    dump_run_metadata(
        run_name,
        args,
        Path(__file__),
        extra_files=[
            Path(reference.__file__),
            Path("/home/Maniskill/train/octo/other_test/common.py"),
            Path("/home/Maniskill/train/octo/other_test/peft_layers.py"),
        ],
    )


reference.build_agent = build_agent
reference.copy_run_metadata = copy_run_metadata


if __name__ == "__main__":
    reference.main()
