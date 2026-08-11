#!/usr/bin/env python
"""GPU free-memory gate for shared-GPU training (game + ML co-use).

Standalone utility — intentionally has NO PH-Neuro imports so it can run
from anywhere (only needs torch, which is a project dependency anyway).

Polls free GPU memory every 15 s and exits 0 as soon as
``free_memory >= threshold_gb``. Used by ``scripts/train.sh`` and the
``research/scripts/run_*.sh`` launchers so a long training run never
crashes with CUDA OOM because the game happens to be holding VRAM.

Usage:
    python scripts/gpu_wait.py                # threshold 7.0 GB, timeout 120 min
    python scripts/gpu_wait.py --threshold 6.0 --timeout 180
    python scripts/gpu_wait.py --interval 10  # poll every 10 s

Exit codes:
    0  GPU free (or no threshold requested) — launch now
    1  timed out waiting
    2  no CUDA available
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

# Total VRAM of the RTX 4060 (GB). Polled at runtime; only a fallback.
DEFAULT_TOTAL_GB = 8.0


def _free_gb_torch() -> float | None:
    """Return free VRAM in GB via torch.cuda.mem_get_info()."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return free_bytes / (1024 ** 3)
    except Exception:
        return None


def _free_gb_nvidia_smi() -> float | None:
    """Return free VRAM in GB by parsing `nvidia-smi --query-gpu=memory.free`."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        line = out.stdout.strip().splitlines()[0]
        free_mib, total_mib = (int(x.strip()) for x in line.split(","))
        return free_mib / 1024.0, total_mib / 1024.0
    except Exception:
        return None


def gpu_free_gb() -> tuple[float | None, float | None]:
    """Return (free_gb, total_gb) from the best available source.

    Prefers torch (authoritative for what the caching allocator sees);
    falls back to nvidia-smi (sees the whole GPU, incl. the game).
    """
    free = _free_gb_torch()
    if free is not None:
        total = DEFAULT_TOTAL_GB
        smi = _free_gb_nvidia_smi()
        if smi is not None:
            # Use the real total from nvidia-smi when available.
            total = smi[1]
        return free, total
    smi = _free_gb_nvidia_smi()
    if smi is not None:
        return smi
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold", type=float, default=7.0,
        help="free VRAM (GB) needed before we launch (default: 7.0)",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="max minutes to wait before giving up (default: 120)",
    )
    parser.add_argument(
        "--interval", type=float, default=15.0,
        help="poll interval in seconds (default: 15)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="only print the final status line",
    )
    args = parser.parse_args()

    if not _free_gb_torch() and not _free_gb_nvidia_smi():
        print("gpu_wait: ERROR — no CUDA GPU found (torch + nvidia-smi both unavailable)")
        return 2

    free, total = gpu_free_gb()
    if free is None or total is None:
        print("gpu_wait: ERROR — could not read GPU memory")
        return 2

    deadline = time.time() + args.timeout * 60.0
    waited = 0.0

    while True:
        free, total = gpu_free_gb()
        if free is None:
            free, total = 0.0, DEFAULT_TOTAL_GB

        if free >= args.threshold:
            print(f"GPU free: {free:.1f}/{total:.1f} GB → launching")
            return 0

        remaining = max(0.0, deadline - time.time())
        if remaining <= 0:
            print(
                f"gpu_wait: TIMEOUT after {args.timeout:.0f} min — "
                f"GPU still only {free:.1f}/{total:.1f} GB free "
                f"(need {args.threshold:.1f} GB). Retry later when the game is closed."
            )
            return 1

        if not args.quiet:
            print(
                f"Waiting: {free:.1f}/{total:.1f} GB free "
                f"(need {args.threshold:.1f} GB) — game detected? "
                f"retry in {remaining/60:.0f} min",
                flush=True,
            )
        time.sleep(min(args.interval, remaining + 0.1))
        waited += args.interval

    # unreachable
    return 1


if __name__ == "__main__":
    sys.exit(main())
