import argparse
import atexit
import fcntl
import os
import shutil
import signal
import subprocess
from pathlib import Path


_RESERVATION_DIR = Path("/tmp/maniskill_gpu_reservations")
_GLOBAL_LOCK_PATH = _RESERVATION_DIR / ".lock"
_RESERVED_GPU_INDEX = None
_RESERVATION_FILE = None
_CLEANUP_REGISTERED = False
_FREE_GPU_MEMORY_THRESHOLD_MB = 1024
_PREFERRED_GPU_ENV_VARS = ("MANISKILL_GPU_INDEX", "GPU_AUTO_SELECT_INDEX")
_METHOD_GPU_OVERRIDE_ENV_VARS = ("GPU_BY_METHOD_OVERRIDE", "METHOD_GPU_OVERRIDE")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cleanup_reservation(*_args) -> None:
    global _RESERVED_GPU_INDEX, _RESERVATION_FILE

    reservation_file = _RESERVATION_FILE
    _RESERVED_GPU_INDEX = None
    _RESERVATION_FILE = None
    if reservation_file and reservation_file.exists():
        try:
            reservation_file.unlink()
        except FileNotFoundError:
            pass


def _register_cleanup_once() -> None:
    global _CLEANUP_REGISTERED
    if _CLEANUP_REGISTERED:
        return

    atexit.register(_cleanup_reservation)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handler = signal.getsignal(signum)

        def _handler(sig, frame, previous_handler=previous_handler):
            _cleanup_reservation()
            if callable(previous_handler):
                previous_handler(sig, frame)
                return
            if previous_handler == signal.SIG_DFL:
                raise SystemExit(128 + sig)

        signal.signal(signum, _handler)

    _CLEANUP_REGISTERED = True


def _cleanup_stale_reservations() -> set[int]:
    reserved = set()
    for reservation_file in _RESERVATION_DIR.glob("gpu_*.lock"):
        try:
            pid = int(reservation_file.read_text().strip())
            gpu_index = int(reservation_file.stem.split("_", 1)[1])
        except (OSError, ValueError, IndexError):
            try:
                reservation_file.unlink()
            except FileNotFoundError:
                pass
            continue

        if _pid_exists(pid):
            reserved.add(gpu_index)
            continue

        try:
            reservation_file.unlink()
        except FileNotFoundError:
            pass
    return reserved


def _run_nvidia_smi_query(*fields: str) -> list[str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is not available, cannot query GPUs.")

    result = subprocess.run(
        [
            nvidia_smi,
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]


def _query_idle_gpu_indices() -> list[int]:
    idle_gpu_indices = []
    for line in _run_nvidia_smi_query("index", "memory.used"):
        gpu_index_str, memory_used_str = [part.strip() for part in line.split(",", 1)]
        if int(memory_used_str) < _FREE_GPU_MEMORY_THRESHOLD_MB:
            idle_gpu_indices.append(int(gpu_index_str))
    return idle_gpu_indices


def _query_all_gpu_indices() -> list[int]:
    return [int(line) for line in _run_nvidia_smi_query("index")]


def _resolve_requested_gpu_index(preferred_gpu: int | str | None) -> int | None:
    if preferred_gpu is not None and str(preferred_gpu).strip():
        candidate = str(preferred_gpu).strip()
    else:
        candidate = ""
        for env_var in _PREFERRED_GPU_ENV_VARS:
            env_value = os.environ.get(env_var, "").strip()
            if env_value:
                candidate = env_value
                break

    if not candidate:
        return None

    try:
        return int(candidate)
    except ValueError as exc:
        raise RuntimeError(
            f"Requested GPU index must be an integer, got {candidate!r}."
        ) from exc


def parse_method_gpu_map(raw_value: str | None) -> dict[str, int]:
    mapping: dict[str, int] = {}
    if raw_value is None:
        return mapping
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(
                f"Invalid GPU mapping item {item!r}. Expected format like method=0."
            )
        method, gpu_str = item.split("=", 1)
        method = method.strip()
        gpu_str = gpu_str.strip()
        if not method or not gpu_str:
            raise RuntimeError(
                f"Invalid GPU mapping item {item!r}. Expected format like method=0."
            )
        try:
            mapping[method] = int(gpu_str)
        except ValueError as exc:
            raise RuntimeError(
                f"GPU index for method {method!r} must be an integer, got {gpu_str!r}."
            ) from exc
    return mapping


def get_method_gpu_override_map(raw_value: str | None = None) -> dict[str, int]:
    if raw_value is not None and raw_value.strip():
        return parse_method_gpu_map(raw_value)
    for env_var in _METHOD_GPU_OVERRIDE_ENV_VARS:
        env_value = os.environ.get(env_var, "").strip()
        if env_value:
            return parse_method_gpu_map(env_value)
    return {}


def resolve_method_gpu_map(
    method_order: list[str],
    default_map: dict[str, int],
    override_map: dict[str, int] | None = None,
    available_gpu_indices: list[int] | None = None,
) -> dict[str, int]:
    if override_map is None:
        override_map = {}
    if available_gpu_indices is None:
        available_gpu_indices = _query_all_gpu_indices()
    if not available_gpu_indices:
        raise RuntimeError("No GPUs found on this machine.")

    available_gpu_indices = sorted(dict.fromkeys(available_gpu_indices))
    available_gpu_set = set(available_gpu_indices)
    gpu_usage = {gpu: 0 for gpu in available_gpu_indices}
    resolved: dict[str, int] = {}

    unknown_override_methods = sorted(set(override_map) - set(method_order))
    if unknown_override_methods:
        raise RuntimeError(
            f"Unknown methods in GPU_BY_METHOD_OVERRIDE: {unknown_override_methods}."
        )

    for method in method_order:
        explicit_gpu = override_map.get(method)
        if explicit_gpu is not None:
            if explicit_gpu not in available_gpu_set:
                raise RuntimeError(
                    f"Requested GPU {explicit_gpu} for method {method!r} does not exist. "
                    f"Available GPU indices: {available_gpu_indices}."
                )
            resolved_gpu = explicit_gpu
        else:
            default_gpu = default_map[method]
            if default_gpu in available_gpu_set:
                resolved_gpu = default_gpu
            else:
                resolved_gpu = min(
                    available_gpu_indices,
                    key=lambda gpu: (gpu_usage[gpu], gpu),
                )
        resolved[method] = resolved_gpu
        gpu_usage[resolved_gpu] += 1
    return resolved


def configure_cuda_visible_devices(preferred_gpu: int | str | None = None) -> str:
    global _RESERVED_GPU_INDEX, _RESERVATION_FILE

    requested_gpu_index = _resolve_requested_gpu_index(preferred_gpu)
    existing_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if existing_value and requested_gpu_index is None:
        print(f"[gpu_auto_select] Reusing CUDA_VISIBLE_DEVICES={existing_value}")
        return existing_value

    _RESERVATION_DIR.mkdir(parents=True, exist_ok=True)
    with _GLOBAL_LOCK_PATH.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        reserved_gpu_indices = _cleanup_stale_reservations()
        if requested_gpu_index is not None:
            available_gpu_indices = set(_query_all_gpu_indices())
            if requested_gpu_index not in available_gpu_indices:
                raise RuntimeError(
                    f"Requested GPU {requested_gpu_index} does not exist. "
                    f"Available GPU indices: {sorted(available_gpu_indices)}."
                )
            if requested_gpu_index in reserved_gpu_indices:
                raise RuntimeError(
                    f"Requested GPU {requested_gpu_index} is already reserved by another process."
                )
            selected_gpu_index = requested_gpu_index
        else:
            idle_gpu_indices = [
                gpu_index
                for gpu_index in _query_idle_gpu_indices()
                if gpu_index not in reserved_gpu_indices
            ]
            if not idle_gpu_indices:
                raise RuntimeError(
                    "No free GPU found. A free GPU is defined here as memory.used < 1024 MiB."
                )
            selected_gpu_index = idle_gpu_indices[0]

        reservation_file = _RESERVATION_DIR / f"gpu_{selected_gpu_index}.lock"
        reservation_file.write_text(str(os.getpid()), encoding="utf-8")
        _RESERVED_GPU_INDEX = selected_gpu_index
        _RESERVATION_FILE = reservation_file
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    _register_cleanup_once()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu_index)
    if requested_gpu_index is not None:
        print(f"[gpu_auto_select] Using requested GPU {selected_gpu_index}")
    else:
        print(f"[gpu_auto_select] Selected free GPU {selected_gpu_index}")
    return str(selected_gpu_index)


def _default_map_from_arg(raw_value: str) -> dict[str, int]:
    mapping = parse_method_gpu_map(raw_value)
    if not mapping:
        raise RuntimeError("--default-map cannot be empty.")
    return mapping


def _print_method_gpu_map_tsv(resolved_map: dict[str, int], method_order: list[str]) -> None:
    for method in method_order:
        print(f"{method}	{resolved_map[method]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    resolve_parser = subparsers.add_parser("resolve-method-map")
    resolve_parser.add_argument("--method-order", required=True)
    resolve_parser.add_argument("--default-map", required=True)
    resolve_parser.add_argument("--override-map", default="")

    args = parser.parse_args()

    if args.command == "resolve-method-map":
        method_order = [item.strip() for item in args.method_order.split(",") if item.strip()]
        if not method_order:
            raise RuntimeError("--method-order cannot be empty.")
        default_map = _default_map_from_arg(args.default_map)
        missing_defaults = [method for method in method_order if method not in default_map]
        if missing_defaults:
            raise RuntimeError(
                f"Default GPU map is missing methods: {missing_defaults}."
            )
        override_map = get_method_gpu_override_map(args.override_map)
        resolved_map = resolve_method_gpu_map(method_order, default_map, override_map)
        _print_method_gpu_map_tsv(resolved_map, method_order)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
