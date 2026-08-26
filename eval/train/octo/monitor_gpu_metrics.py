from __future__ import annotations

import signal

from train.common.monitor_gpu_metrics import main


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(0)))
    main()
