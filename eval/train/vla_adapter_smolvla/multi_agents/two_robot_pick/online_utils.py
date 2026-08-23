import ast
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ContinualEnvSchedule:
    env_kwarg_list: List[dict]
    change_time_points: List[float]


def _parse_cli_sequence(raw_value, arg_name, cast_fn):
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        raw_text = str(raw_value).strip()
        if raw_text == "":
            return []
        try:
            parsed_value = ast.literal_eval(raw_text)
        except (SyntaxError, ValueError):
            parsed_value = [item.strip() for item in raw_text.split(",") if item.strip()]
        if isinstance(parsed_value, (list, tuple)):
            values = list(parsed_value)
        else:
            values = [parsed_value]
    try:
        return [cast_fn(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Failed to parse `{arg_name}` from {raw_value!r}") from exc


def build_continual_env_schedule(args, env_kwarg_list=None) -> Optional[ContinualEnvSchedule]:
    time_points = _parse_cli_sequence(args.env_change_time_points, "env_change_time_points", float)
    if env_kwarg_list is None and time_points is None:
        return None
    if env_kwarg_list is None or time_points is None:
        raise ValueError("`envs_id` and `env_change_time_points` must be provided together")
    if len(env_kwarg_list) == 0:
        raise ValueError("`envs_id` must contain at least one environment")
    if len(env_kwarg_list) != len(time_points):
        raise ValueError(
            f"`envs_id` and `env_change_time_points` must have the same length, "
            f"got {len(env_kwarg_list)} and {len(time_points)}"
        )
    last_time_point = None
    for time_point in time_points:
        if time_point <= 0:
            raise ValueError("All `env_change_time_points` must be positive")
        if last_time_point is not None and time_point <= last_time_point:
            raise ValueError("`env_change_time_points` must be strictly increasing")
        last_time_point = time_point
    return ContinualEnvSchedule(env_kwarg_list=env_kwarg_list, change_time_points=time_points)
