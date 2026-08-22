from collections import defaultdict
import os
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorboard.backend.event_processing import event_accumulator


# DATA_LOGS = {
#     # "ours": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-080331-ours",
#     # "ours": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-125501-ours-target-single",
#     "ours (with feature agg)": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-173128-ours-target-batch-dual-gate",
#     # "ours (update feature agg)": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-174753-ours-target-batch-dual-gate-update-fa",
#     # "ours (no feature agg)": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-082543-ours-ab-no-fa",
#     "ours (without feature agg)": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-173159-target-batch-no-fa",
#     # "baseline (source small model + no feature agg)": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-132513-source-no-fa",
# }
# DATA_LOGS = {
#     # "ours": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-080331-ours",
#     # "ours": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-125501-ours-target-single",
#     "ours (target small model + no feature agg)": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-141057-target-batch-no-fa",
#     # "ours (update feature agg)": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-174753-ours-target-batch-dual-gate-update-fa",
#     # "ours (no feature agg)": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-082543-ours-ab-no-fa",
#     # "ours (without feature agg)": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-173159-target-batch-no-fa",
#     "baseline (source small model + no feature agg)": "ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-132513-source-no-fa",
# }


# DATA_LOGS = {
#     'topk-return': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260409-070716-ours-target-batch-topk_return',
#     'random': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260409-070810-ours-target-batch-random',
#     'return-span': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260409-070849-ours-target-batch-return_span',
#     'no fa': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260408-173159-target-batch-no-fa'
# }

# DATA_LOGS = {
#     # 'ours': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260410-025637-ours-dual_gate_h4_2layergate_none',

#     'ours': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260410-083411-ours-dual_gate_h4_2layergate_none-targetsingletraj',

#     # 'ours (no fa)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260410-025926-ours-no-fa',
    
#     # 'ours (update fa)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260410-033834-ours-dual_gate_h4_2layergate_none-updatefa',
#     'baseline (generating small model on source task +\nno feature aggregation)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260410-044256-baseline-source'
# }

DATA_LOGS = {
    # 'ours (regenerate once)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260410-163328-ours-dual_gate_h4_2layergate_none-targetsingletraj-regen_once',
    # 'ours (regenerate once reproduce)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-022623-ours-targetsingletraj-regen_once',
    # 'ours (4.11newimpl, regenerate before rollout)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260410-161435-ours-dual_gate_h4_2layergate_none-targetsingletraj-regen_before_per_rollout',
    # 'ours (4.11newimpl, regenerate before rollout + alpha 0.2)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260410-165927-ours-dual_gate_h4_2layergate_none-targetsingletraj-regen_before_per_rollout-feedback0.2',
    
    # 'ours (regen before rollout + no reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260411-074503-ours-dual_gate_h4_2layergate_none-targetsingletraj-regen_before_per_rollout-feedback1.0-no_reset_optimizer',
    # 'ours (regen before rollout + reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260411-074635-ours-dual_gate_h4_2layergate_none-targetsingletraj-regen_before_per_rollout-feedback1.0-reset_optimizer',

    # 'ours (alpha 0.1, incre-regen 0.1)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260411-145815-ours-targetsingletraj-regen_per_rollout-feedback0.1-arch_update0.1-no_reset_optimizer',
    # 'ours (alpha 0.1, incre-regen 0.5)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260411-145413-ours-targetsingletraj-regen_per_rollout-feedback0.1-arch_update0.5-no_reset_optimizer',

    # 'ours (alpha 1.0, incre-regen 0.1)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260412-113808-ours-targetsingletraj-regen_per_rollout-feedback1.0-arch_update0.1-no_reset_optimizer',
    # 'ours (alpha 1.0, incre-regen 0.02)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260412-113945-ours-targetsingletraj-regen_per_rollout-feedback1.0-arch_update0.02-no_reset_optimizer'
    
    # 'ours (alpha 1.0, incre-regen 0.1, reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-052359-ours-targetsingletraj-regen_per_rollout-feedback1.0-arch_update0.1-reset_optimizer',
    # 'ours (alpha 0.1, incre-regen 0.1, reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-052232-ours-targetsingletraj-regen_per_rollout-feedback0.1-arch_update0.1-reset_optimizer',
    # 'ours (alpha 0.1, incre-regen 0.05, reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-055829-ours-targetsingletraj-regen_per_rollout-feedback0.1-arch_update0.05-reset_optimizer',

    # 'ours (alpha 0.1, incre-regen 0, reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-074326-ours-targetsingletraj-regen_per_rollout-feedback0.1-arch_update0-reset_optimizer',

    # 'ours (alpha 1.0, incre-regen 0.02, reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-081206-ours-targetsingletraj-regen_per_rollout-feedback1.0-arch_update0.02-reset_optimizer',
    # 'ours (alpha 1.0, incre-regen 0.2, reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-081322-ours-targetsingletraj-regen_per_rollout-feedback1.0-arch_update0.2-reset_optimizer',

    # 'ours (alpha 0.1, incre-regen 0.02, reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-084938-ours-targetsingletraj-regen_per_rollout-feedback0.1-arch_update0.02-reset_optimizer',
    # 'ours (alpha 0.1, incre-regen 0.02, reset opt,\nbetter policy for finding traj)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-095704-ours-targetsingletraj-regen_per_rollout-better_policy-feedback0.1-arch_update0.02-reset_optimizer',
    # 'ours (alpha 0.1, incre-regen 0.02, no reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-085916-ours-targetsingletraj-regen_per_rollout-feedback0.1-arch_update0.02-no_reset_optimizer',
    # 'ours (regen_when_acc_improv>0.2, alpha 0.1, \nincre-regen 0.02, reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-111656-ours-targetsingletraj-regen_per_rollout_when_acc_improv_larger0.20-large_policy-feedback0.1-arch_update0.02-reset_optimizer',
    # 'ours (alpha 0.02, incre-regen 0.02, reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-085522-ours-targetsingletraj-regen_per_rollout-feedback0.02-arch_update0.02-reset_optimizer'
    # 'ours (regen_when_acc_improv>0.2, alpha 0.1, \nincre-regen 0.01, reset opt)': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260413-151511-ours-targetsingletraj-regen_per_rollout_when_acc_improv_larger0.20-large_policy-feedback0.1-arch_update0.01-reset_optimizer',

    '单次生成/反馈': 'ckpt/PickCube-v1-mutable/ours/octo/online_rl/20260414-122407-ours-targetsingletraj-regen_once',

    # -0.02
    # '1': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260414-145839-ours-targetsingletraj-regen_per_rollout0.5-small_policy-feedback0.1-arch_update0.02-reset_optimizer'
    
    # '1': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260414-155729-ours-targetsingletraj-regen_per_rollout0.1-small_policy-feedback0.1-arch_update0.05-reset_optimizer'

    # '1': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260415-011235-ours-targetsingletraj-regen_per_rollout0.1-small_policy-feedback0.1-arch_update0.02-reset_optimizer',

    # +0.02
    # '选择性生成/反馈': 'ckpt/PickCube-v1-mutable/ours/toy_cnn/online_rl/20260415-014729-ours-targetsingletraj-feedif0.2-regen_per_rollout_0.1_5-small_policy-feedback0.1-arch_update0.05-reset_optimizer'

    # SOTA: +0.1 on agent 1, +0.03 on agent 2
    '选择性生成/反馈': 'ckpt/PickCube-v1-mutable/ours/octo/online_rl/20260415-030814-ours-targetsingletraj-feedif0.2-regen_per_rollout_0.1_4-small_policy-feedback0.1-arch_update0.05-reset_optimizer'
}

TARGET_TAG = "eval/success_end"
MAX_RELATIVE_MINUTES = 60
SMOOTHING = 0.
DRAW_AGENT_DETAILS = True
DRAW_REGENERATION_MARKERS = True
OUTPUT_NAME = "online_rl_success_end_vs_time.png"


def configure_matplotlib_fonts():
    candidate_font_paths = [
        os.environ.get("MATPLOTLIB_FONT_PATH"),
        str(Path(__file__).with_name("fonts") / "NotoSansCJK-Regular.ttc"),
        str(Path(__file__).with_name("fonts") / "NotoSansSC-Regular.otf"),
        str(Path(__file__).with_name("fonts") / "SourceHanSansSC-Regular.otf"),
    ]
    chosen_font_name = None

    for font_path in candidate_font_paths:
        if not font_path:
            continue
        if Path(font_path).is_file():
            font_manager.fontManager.addfont(font_path)
            chosen_font_name = font_manager.FontProperties(fname=font_path).get_name()
            break

    if chosen_font_name is None:
        installed_font_names = {
            font_manager.FontProperties(fname=font_path).get_name()
            for font_path in font_manager.findSystemFonts()
        }
        common_cjk_fonts = [
            "Noto Sans CJK SC",
            "Noto Sans SC",
            "Source Han Sans SC",
            "Source Han Sans CN",
            "Microsoft YaHei",
            "SimHei",
            "WenQuanYi Zen Hei",
            "PingFang SC",
            "Heiti SC",
            "STHeiti",
            "Arial Unicode MS",
            "Sarasa Gothic SC",
        ]
        for font_name in common_cjk_fonts:
            if font_name in installed_font_names:
                chosen_font_name = font_name
                break

    font_candidates = [font_name for font_name in [chosen_font_name, "DejaVu Sans"] if font_name]
    plt.rcParams.update({
        "font.size": 22,
        "font.family": "sans-serif",
        "font.sans-serif": font_candidates,
        "axes.unicode_minus": False,
    })

    if chosen_font_name is None:
        print(
            "Warning: no Chinese font found. Chinese text may render as boxes. "
            "Set `MATPLOTLIB_FONT_PATH` to a `.ttf`/`.ttc`/`.otf` font file."
        )
    else:
        print(f"Using matplotlib font: {chosen_font_name}")


configure_matplotlib_fonts()


def load_scalar_events(tb_dir: Path, tag: str):
    accumulator = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    if tag not in scalar_tags:
        return []
    return accumulator.Scalars(tag)


def load_run_args(log_dir: Path):
    args_path = log_dir / "code" / "args.txt"
    if not args_path.is_file():
        return {}

    args = {}
    with open(args_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            args[key.strip()] = value.strip()
    return args


def get_optional_run_arg(run_args, key, default=None):
    value = run_args.get(key, default)
    if value in {None, "None", "null", ""}:
        return default
    return value


def collect_agent_tb_dirs(log_dir: Path):
    def agent_sort_key(path: Path):
        matched = re.search(r"\[agent(\d+)\]", path.name)
        if matched:
            return (0, int(matched.group(1)))
        return (1, path.name)

    agent_dirs = []
    for path in sorted(log_dir.iterdir(), key=agent_sort_key):
        if path.is_dir() and path.name.startswith("[agent"):
            tb_dir = path / "tb"
            if tb_dir.is_dir():
                agent_dirs.append(tb_dir)
    return agent_dirs


def smoothing(scalars, weight):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def infer_schedule_wall_times(events, schedule: str, marker_type: str):
    if not events:
        return []

    if schedule == "once":
        return []

    if schedule == "before_per_rollout":
        return [event.wall_time for event in events[1:]]

    threshold_prefix = "before_per_rollout_if_success_improv_is_larger_than_"
    if schedule.startswith(threshold_prefix):
        threshold = float(schedule[len(threshold_prefix):])
        marker_wall_times = []
        success_end_at_last_marker = events[0].value
        for event in events[1:]:
            if event.value - success_end_at_last_marker > threshold:
                marker_wall_times.append(event.wall_time)
                success_end_at_last_marker = event.value
        return marker_wall_times

    threshold_match = re.fullmatch(
        r"before_per_rollout_if_success_improv_less_than_"
        r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)_for_(\d+)_iters",
        schedule,
    )
    if threshold_match is not None:
        if marker_type != "generation":
            return []
        threshold = float(threshold_match.group(1))
        num_iters = int(threshold_match.group(2))
        marker_wall_times = []
        success_end_at_last_marker = events[0].value
        last_marker_event_idx = 0
        for event_idx, event in enumerate(events[1:], start=1):
            if (
                event_idx - last_marker_event_idx >= num_iters
                and event.value - success_end_at_last_marker < threshold
            ):
                marker_wall_times.append(event.wall_time)
                success_end_at_last_marker = event.value
                last_marker_event_idx = event_idx
        return marker_wall_times

    return []


def resolve_feedback_schedule(run_args):
    feedback_schedule = get_optional_run_arg(run_args, "small_model_feedback_schedule")
    if feedback_schedule is not None:
        return feedback_schedule

    regeneration_schedule = get_optional_run_arg(run_args, "small_model_regeneration_schedule", "once")
    legacy_feedback_compatible_prefix = "before_per_rollout_if_success_improv_is_larger_than_"
    if (
        regeneration_schedule in {"once", "before_per_rollout"}
        or regeneration_schedule.startswith(legacy_feedback_compatible_prefix)
    ):
        return regeneration_schedule
    return "once"


def build_small_model_timing_markers(log_dir: Path, tag: str):
    agent_tb_dirs = collect_agent_tb_dirs(log_dir)
    if not agent_tb_dirs:
        raise FileNotFoundError(f"No agent tensorboard directories found under {log_dir}")

    run_args = load_run_args(log_dir)
    generation_schedule = get_optional_run_arg(run_args, "small_model_regeneration_schedule", "once")
    feedback_schedule = resolve_feedback_schedule(run_args)

    agent_events = []
    start_wall_time = None
    for tb_dir in agent_tb_dirs:
        events = load_scalar_events(tb_dir, tag)
        agent_events.append((tb_dir, events))
        if events:
            agent_start = events[0].wall_time
            if start_wall_time is None or agent_start < start_wall_time:
                start_wall_time = agent_start

    if start_wall_time is None:
        raise ValueError(f"No scalar data with tag '{tag}' found under {log_dir}")

    markers = []
    for tb_dir, events in agent_events:
        generation_xs = [
            (wall_time - start_wall_time) / 60.0
            for wall_time in infer_schedule_wall_times(events, generation_schedule, marker_type="generation")
        ]
        feedback_xs = [
            (wall_time - start_wall_time) / 60.0
            for wall_time in infer_schedule_wall_times(events, feedback_schedule, marker_type="feedback")
        ]
        if MAX_RELATIVE_MINUTES is not None:
            generation_xs = [x for x in generation_xs if x <= MAX_RELATIVE_MINUTES]
            feedback_xs = [x for x in feedback_xs if x <= MAX_RELATIVE_MINUTES]
        markers.append(
            {
                "agent_name": tb_dir.parent.name,
                "generation_xs": generation_xs,
                "feedback_xs": feedback_xs,
            }
        )

    return markers


def add_vertical_markers(axis, xs, color, linestyle, label=None):
    for idx, x in enumerate(xs):
        axis.axvline(
            x,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.5,
            label=label if idx == 0 else None,
        )


def apply_axis_legend(axis, fontsize=16):
    handles, labels = axis.get_legend_handles_labels()
    deduped = {}
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in deduped:
            continue
        deduped[label] = handle
    if deduped:
        axis.legend(deduped.values(), deduped.keys(), fontsize=fontsize)


def build_curve(log_dir: Path, tag: str):
    agent_tb_dirs = collect_agent_tb_dirs(log_dir)
    if not agent_tb_dirs:
        raise FileNotFoundError(f"No agent tensorboard directories found under {log_dir}")

    per_step = defaultdict(lambda: {"times": [], "values": []})
    start_wall_time = None

    for tb_dir in agent_tb_dirs:
        events = load_scalar_events(tb_dir, tag)
        if not events:
            continue
        agent_start = events[0].wall_time
        if start_wall_time is None or agent_start < start_wall_time:
            start_wall_time = agent_start
        for event in events:
            per_step[event.step]["times"].append(event.wall_time)
            per_step[event.step]["values"].append(event.value)

    if not per_step:
        raise ValueError(f"No scalar data with tag '{tag}' found under {log_dir}")

    xs = []
    ys = []
    for step in sorted(per_step):
        wall_times = per_step[step]["times"]
        values = per_step[step]["values"]
        xs.append((sum(wall_times) / len(wall_times) - start_wall_time) / 60.0)
        ys.append(sum(values) / len(values))
    if MAX_RELATIVE_MINUTES is not None:
        filtered_points = [(x, y) for x, y in zip(xs, ys) if x <= MAX_RELATIVE_MINUTES]
        if not filtered_points:
            raise ValueError(
                f"No data points within the first {MAX_RELATIVE_MINUTES} minutes under {log_dir}"
            )
        xs, ys = zip(*filtered_points)
        xs, ys = list(xs), list(ys)

    ys = smoothing(ys, SMOOTHING) if SMOOTHING > 0 else ys

    return xs, ys, len(agent_tb_dirs)


def build_agent_curves(log_dir: Path, tag: str):
    agent_tb_dirs = collect_agent_tb_dirs(log_dir)
    if not agent_tb_dirs:
        raise FileNotFoundError(f"No agent tensorboard directories found under {log_dir}")

    agent_events = []
    start_wall_time = None
    for tb_dir in agent_tb_dirs:
        events = load_scalar_events(tb_dir, tag)
        agent_events.append((tb_dir, events))
        if events:
            agent_start = events[0].wall_time
            if start_wall_time is None or agent_start < start_wall_time:
                start_wall_time = agent_start

    if start_wall_time is None:
        raise ValueError(f"No scalar data with tag '{tag}' found under {log_dir}")

    curves = []
    for tb_dir, events in agent_events:
        xs = [(event.wall_time - start_wall_time) / 60.0 for event in events]
        ys = [event.value for event in events]

        if MAX_RELATIVE_MINUTES is not None:
            filtered_points = [(x, y) for x, y in zip(xs, ys) if x <= MAX_RELATIVE_MINUTES]
            if filtered_points:
                xs, ys = zip(*filtered_points)
                xs, ys = list(xs), list(ys)
            else:
                xs, ys = [], []

        if ys and SMOOTHING > 0:
            ys = smoothing(ys, SMOOTHING)

        curves.append((tb_dir.parent.name, xs, ys))

    return curves


def build_output_name():
    output_path = Path(OUTPUT_NAME)
    if not DRAW_AGENT_DETAILS:
        return output_path.name
    return f"{output_path.stem}_agent_details{output_path.suffix}"


def main():
    if not DATA_LOGS:
        raise ValueError("Please fill DATA_LOGS with {method_name: log_dir} entries first.")

    ordered_items = list(DATA_LOGS.items())
    first_log_dir = Path(ordered_items[0][1])
    output_path = first_log_dir / build_output_name()

    if DRAW_AGENT_DETAILS:
        _, first_log_dir_str = ordered_items[0]
        reference_curves = build_agent_curves(Path(first_log_dir_str), TARGET_TAG)
        num_agents = len(reference_curves)
        figure, axes = plt.subplots(num_agents, 1, figsize=(10, 4 * num_agents), squeeze=False)
        axes = axes.flatten()
        generation_legend_drawn = [False] * num_agents
        feedback_legend_drawn = [False] * num_agents

        max_x = 0
        for method_name, log_dir_str in ordered_items:
            log_dir = Path(log_dir_str)
            agent_curves = build_agent_curves(log_dir, TARGET_TAG)
            timing_markers = build_small_model_timing_markers(log_dir, TARGET_TAG) if DRAW_REGENERATION_MARKERS else None
            if len(agent_curves) != num_agents:
                raise ValueError(
                    f"Method '{method_name}' has {len(agent_curves)} agents, expected {num_agents}."
                )
            if timing_markers is not None and len(timing_markers) != num_agents:
                raise ValueError(
                    f"Method '{method_name}' has {len(timing_markers)} timing marker groups, expected {num_agents}."
                )

            for agent_idx, (_, xs, ys) in enumerate(agent_curves):
                if not xs or not ys:
                    continue
                axes[agent_idx].plot(
                    xs,
                    ys,
                    linewidth=2,
                    label=f"{method_name} (avg.: {(sum(ys) / len(ys)):.3f})",
                )
                max_x = max(max_x, xs[-1])
                if timing_markers is not None:
                    marker_info = timing_markers[agent_idx]
                    add_vertical_markers(
                        axes[agent_idx],
                        marker_info["generation_xs"],
                        color="red",
                        linestyle="--",
                        label="生成" if not generation_legend_drawn[agent_idx] and marker_info["generation_xs"] else None,
                    )
                    add_vertical_markers(
                        axes[agent_idx],
                        marker_info["feedback_xs"],
                        color="purple",
                        linestyle=":",
                        label="反馈" if not feedback_legend_drawn[agent_idx] and marker_info["feedback_xs"] else None,
                    )
                    if marker_info["generation_xs"]:
                        generation_legend_drawn[agent_idx] = True
                    if marker_info["feedback_xs"]:
                        feedback_legend_drawn[agent_idx] = True

        for agent_idx, axis in enumerate(axes):
            axis.set_title(f"Agent {agent_idx}")
            axis.set_ylabel("Task Success Rate")
            axis.set_xlim(left=0, right=max_x)
            if MAX_RELATIVE_MINUTES is not None and MAX_RELATIVE_MINUTES < max_x:
                axis.set_xlim(right=MAX_RELATIVE_MINUTES)
            axis.grid(True, linestyle="--", alpha=0.3)
            apply_axis_legend(axis, fontsize=16)

        axes[-1].set_xlabel("Training Time (minutes)")
        figure.tight_layout()
    else:
        figure, axis = plt.subplots(figsize=(10, 6))
        generation_legend_drawn = False
        feedback_legend_drawn = False

        max_x = 0
        for method_name, log_dir_str in ordered_items:
            log_dir = Path(log_dir_str)
            xs, ys, num_agents = build_curve(log_dir, TARGET_TAG)
            axis.plot(xs, ys, linewidth=2, label=f"{method_name} (avg.: {(sum(ys) / len(ys)):.3f})")
            max_x = max(max_x, xs[-1])
            if DRAW_REGENERATION_MARKERS:
                for marker_info in build_small_model_timing_markers(log_dir, TARGET_TAG):
                    add_vertical_markers(
                        axis,
                        marker_info["generation_xs"],
                        color="red",
                        linestyle="--",
                        label="Small-model generation" if not generation_legend_drawn and marker_info["generation_xs"] else None,
                    )
                    add_vertical_markers(
                        axis,
                        marker_info["feedback_xs"],
                        color="purple",
                        linestyle=":",
                        label="Small-model feedback" if not feedback_legend_drawn and marker_info["feedback_xs"] else None,
                    )
                    if marker_info["generation_xs"]:
                        generation_legend_drawn = True
                    if marker_info["feedback_xs"]:
                        feedback_legend_drawn = True

        axis.set_xlabel("Training Time (minutes)")
        axis.set_ylabel("Task Success Rate")
        axis.set_xlim(left=0, right=max_x)
        if MAX_RELATIVE_MINUTES is not None and MAX_RELATIVE_MINUTES < max_x:
            axis.set_xlim(right=MAX_RELATIVE_MINUTES)
        axis.grid(True, linestyle="--", alpha=0.3)
        apply_axis_legend(axis, fontsize=16)
        figure.tight_layout()

    figure.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")

    with open(output_path.with_suffix(".txt"), "w") as f:
        f.write(f'Metric: {TARGET_TAG}\n')
        f.write(f'Max Relative Minutes: {MAX_RELATIVE_MINUTES}\n\n')

        for method_name, log_dir_str in ordered_items:
            log_dir = Path(log_dir_str)
            xs, ys, num_agents = build_curve(log_dir, TARGET_TAG)
            
            f.write(f"Method: {method_name}\n")
            f.write(f"Average Success Rate: {(sum(ys) / len(ys)):.3f}\n")
            f.write(f"Data Path: {log_dir}\n")
            if DRAW_AGENT_DETAILS:
                for agent_name, agent_xs, agent_ys in build_agent_curves(log_dir, TARGET_TAG):
                    if not agent_ys:
                        f.write(f"{agent_name}: no data\n")
                        continue
                    f.write(f"{agent_name} Average Success Rate: {(sum(agent_ys) / len(agent_ys)):.3f}\n")
                if DRAW_REGENERATION_MARKERS:
                    for marker_info in build_small_model_timing_markers(log_dir, TARGET_TAG):
                        f.write(
                            f"{marker_info['agent_name']} Small-model Generation Times (minutes): "
                            f"{marker_info['generation_xs']}\n"
                        )
                        f.write(
                            f"{marker_info['agent_name']} Small-model Feedback Times (minutes): "
                            f"{marker_info['feedback_xs']}\n"
                        )
            elif DRAW_REGENERATION_MARKERS:
                all_generation_xs = []
                all_feedback_xs = []
                for marker_info in build_small_model_timing_markers(log_dir, TARGET_TAG):
                    all_generation_xs.extend(marker_info["generation_xs"])
                    all_feedback_xs.extend(marker_info["feedback_xs"])
                f.write(f"Small-model Generation Times (minutes): {all_generation_xs}\n")
                f.write(f"Small-model Feedback Times (minutes): {all_feedback_xs}\n")
            f.write("\n")
    print(f"Saved detailed data to {output_path.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
