"""Small helpers: logging, atomic writes (a killed run never leaves a half file that looks done), stage skipping."""
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]          # repo root (the folder that contains src/)
DATA = Path(os.environ.get("ER_DATA_DIR", ROOT / "data" / "student_resource" / "dataset"))   # train/ and test/ TSVs
WORK = Path(os.environ.get("ER_WORK_DIR", ROOT / "work"))                                    # intermediate files


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def announce_pid(name: str):
    log(f"{name} running as PID {os.getpid()}  (stop: kill {os.getpid()} | pause: kill -STOP {os.getpid()} | resume: kill -CONT {os.getpid()})")


def write_parquet_atomic(df: pl.DataFrame, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def done(path: Path, force: bool = False) -> bool:
    """True if the stage output exists and we are not forcing a rebuild (checkpoint skip)."""
    return Path(path).exists() and not force


@contextmanager
def timed(label: str):
    t = time.time()
    log(f"start: {label}")
    yield
    log(f"done:  {label} in {time.time() - t:.1f}s")


def mem_gb() -> float:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)
    except Exception:
        return float("nan")


def resource_guard(label: str, min_free_disk_gb: float = 30.0, min_avail_ram_gb: float = 3.0):
    """Refuse to start a heavy step when the machine is short on disk or memory (protects the user's Mac)."""
    import shutil
    import psutil
    free_disk = shutil.disk_usage(ROOT).free / 1024 ** 3
    avail = psutil.virtual_memory().available / 1024 ** 3
    swap = psutil.swap_memory().used / 1024 ** 3
    log(f"guard[{label}]: free disk {free_disk:.0f} GB, available RAM {avail:.1f} GB, swap used {swap:.1f} GB")
    if free_disk < min_free_disk_gb:
        raise SystemExit(f"STOP: only {free_disk:.0f} GB disk free (< {min_free_disk_gb}). Free space first.")
    if avail < min_avail_ram_gb:
        raise SystemExit(f"STOP: only {avail:.1f} GB RAM available (< {min_avail_ram_gb}). Close apps or lower workers.")


def rss_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1024 ** 3
