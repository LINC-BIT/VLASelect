from __future__ import annotations

import gc
from typing import Any

import torch


def close_envs(*envs: Any) -> None:
    for env in envs:
        if env is None:
            continue
        close = getattr(env, "close", None)
        if callable(close):
            close()


def clear_torch_cuda_cache() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
    if callable(ipc_collect):
        try:
            ipc_collect()
        except RuntimeError:
            pass
