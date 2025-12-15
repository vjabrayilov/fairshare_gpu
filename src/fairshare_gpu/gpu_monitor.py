# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations


import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


@dataclass
class GPUSamplerHandle:
    thread: threading.Thread
    stop_flag: threading.Event
    out_path: Path


def start_gpu_sampler(
    out_csv: str | Path,
    gpu_index: int = 0,
    sample_ms: int = 200,
    record_processes: bool = False,
) -> Optional[GPUSamplerHandle]:
    """Start an NVML telemetry sampler in a background thread.

    Writes a CSV with (timestamp_s, util_gpu, util_mem, mem_used_mb, mem_total_mb, power_w, temp_c).
    If NVML is unavailable, returns None.

    Parameters
    ----------
    out_csv:
        Destination CSV path.
    gpu_index:
        GPU index for NVML (matches nvidia-smi ordering).
    sample_ms:
        Sampling period in milliseconds.
    record_processes:
        If True, also records the number of running compute processes and total per-process memory.
        (This is best-effort; permissions may restrict visibility.)

    Returns
    -------
    GPUSamplerHandle | None
    """
    try:
        import pynvml  # type: ignore
    except Exception as e:  # pragma: no cover
        console.print(f"[yellow]NVML not available (pynvml import failed): {e}[/yellow]")
        return None

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stop_flag = threading.Event()

    def _worker():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            mem_total = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2)

            with out_path.open("w", newline="") as f:
                writer = csv.writer(f)
                header = [
                    "timestamp_s",
                    "util_gpu",
                    "util_mem",
                    "mem_used_mb",
                    "mem_total_mb",
                    "power_w",
                    "temp_c",
                ]
                if record_processes:
                    header += ["num_procs", "procs_mem_mb"]
                writer.writerow(header)

                period_s = max(sample_ms / 1000.0, 0.05)
                while not stop_flag.is_set():
                    ts = time.time()
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    power = None
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except Exception:
                        power = None
                    temp = None
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        temp = None

                    row = [
                        f"{ts:.3f}",
                        util.gpu,
                        util.memory,
                        f"{mem.used / (1024**2):.2f}",
                        f"{mem_total:.2f}",
                        f"{power:.2f}" if power is not None else "",
                        temp if temp is not None else "",
                    ]

                    if record_processes:
                        num_procs = 0
                        procs_mem = 0.0
                        try:
                            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                            num_procs = len(procs)
                            procs_mem = sum(p.usedGpuMemory for p in procs) / (1024**2)
                        except Exception:
                            pass
                        row += [num_procs, f"{procs_mem:.2f}"]

                    writer.writerow(row)
                    f.flush()
                    time.sleep(period_s)
        except Exception as e:  # pragma: no cover
            console.print(f"[yellow]GPU sampler failed: {e}[/yellow]")
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    t = threading.Thread(target=_worker, name="gpu-sampler", daemon=True)
    t.start()
    return GPUSamplerHandle(thread=t, stop_flag=stop_flag, out_path=out_path)


def stop_gpu_sampler(handle: Optional[GPUSamplerHandle]) -> None:
    if handle is None:
        return
    handle.stop_flag.set()
    handle.thread.join(timeout=5.0)
