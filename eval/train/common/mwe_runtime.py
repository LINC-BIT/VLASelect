from __future__ import annotations

import os
import time
from dataclasses import dataclass


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ActiveRuntimeTracker:
    wall_clock_start_time: float
    active_only: bool = False
    active_seconds: float = 0.0

    @classmethod
    def from_env(
        cls,
        *,
        wall_clock_start_time: float | None = None,
        env_var: str = "MWE_ACTIVE_RUNTIME_ONLY",
    ) -> "ActiveRuntimeTracker":
        if wall_clock_start_time is None:
            wall_clock_start_time = time.monotonic()
        return cls(
            wall_clock_start_time=wall_clock_start_time,
            active_only=_env_flag(env_var, default=False),
        )

    def add_active_seconds(self, seconds: float) -> None:
        self.active_seconds += max(0.0, float(seconds))

    def current_seconds(self, extra_active_seconds: float = 0.0) -> float:
        if self.active_only:
            return self.active_seconds + max(0.0, float(extra_active_seconds))
        return max(0.0, time.monotonic() - self.wall_clock_start_time)

    def current_minutes(self, extra_active_seconds: float = 0.0) -> float:
        return self.current_seconds(extra_active_seconds=extra_active_seconds) / 60.0

    def current_hours(self, extra_active_seconds: float = 0.0) -> float:
        return self.current_seconds(extra_active_seconds=extra_active_seconds) / 3600.0
