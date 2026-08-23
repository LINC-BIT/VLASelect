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


def _query_idle_gpu_indices() -> list[int]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is not available, cannot auto-select a free GPU.")

    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    idle_gpu_indices = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        gpu_index_str, memory_used_str = [part.strip() for part in line.split(",", 1)]
        if int(memory_used_str) < _FREE_GPU_MEMORY_THRESHOLD_MB:
            idle_gpu_indices.append(int(gpu_index_str))
    return idle_gpu_indices


def configure_cuda_visible_devices() -> str:
    global _RESERVED_GPU_INDEX, _RESERVATION_FILE

    existing_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if existing_value:
        print(f"[gpu_auto_select] Reusing CUDA_VISIBLE_DEVICES={existing_value}")
        return existing_value

    _RESERVATION_DIR.mkdir(parents=True, exist_ok=True)
    with _GLOBAL_LOCK_PATH.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        reserved_gpu_indices = _cleanup_stale_reservations()
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
    print(f"[gpu_auto_select] Selected free GPU {selected_gpu_index}")
    return str(selected_gpu_index)
