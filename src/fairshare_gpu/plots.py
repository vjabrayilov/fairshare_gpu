from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console

from .analysis import load_run

console = Console()


def plot_latency_cdf(run_dir: str | Path, out_png: str | Path) -> None:
    df, meta = load_run(run_dir)
    if meta.get("warmup_end") is not None:
        df = df[df["t_submit"] >= float(meta["warmup_end"])].copy()
    df = df[df["error"].isna()].copy()
    df["latency_s"] = df["t_done"] - df["t_submit"]

    plt.figure()
    for tid, dft in df.groupby("tenant_id"):
        xs = dft["latency_s"].dropna().sort_values().to_numpy()
        if len(xs) == 0:
            continue
        ys = (range(1, len(xs) + 1))
        ys = [y / len(xs) for y in ys]
        plt.plot(xs, ys, label=str(tid))

    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    console.print(f"[green]Saved[/green] {out_png}")


def plot_throughput_bar(run_dir: str | Path, out_png: str | Path) -> None:
    run_dir = Path(run_dir)
    summary_path = run_dir / "analysis" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    per = summary["per_tenant"]

    tenants = list(per.keys())
    vals = [per[t]["throughput_tokens_s"] for t in tenants]

    plt.figure()
    plt.bar(tenants, vals)
    plt.xlabel("Tenant")
    plt.ylabel("Throughput (output tokens/s)")
    plt.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    console.print(f"[green]Saved[/green] {out_png}")


def plot_gpu_util(run_dir: str | Path, out_png: str | Path) -> None:
    run_dir = Path(run_dir)
    gpu_csv = run_dir / "gpu.csv"
    if not gpu_csv.exists():
        console.print(f"[yellow]No gpu.csv found at {gpu_csv} (telemetry disabled or failed).[/yellow]")
        return

    df = pd.read_csv(gpu_csv)
    if df.empty:
        console.print(f"[yellow]gpu.csv is empty.[/yellow]")
        return

    t0 = df["timestamp_s"].min()
    df["t_rel_s"] = df["timestamp_s"] - t0

    plt.figure()
    plt.plot(df["t_rel_s"], df["util_gpu"])
    plt.xlabel("Time (s)")
    plt.ylabel("GPU Utilization (%)")
    plt.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    console.print(f"[green]Saved[/green] {out_png}")


def plot_all(run_dir: str | Path, out_dir: Optional[str | Path] = None) -> Path:
    run_dir = Path(run_dir)
    if out_dir is None:
        out_dir = run_dir / "figures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_latency_cdf(run_dir, out_dir / "latency_cdf.png")
    plot_throughput_bar(run_dir, out_dir / "throughput_tokens_s.png")
    plot_gpu_util(run_dir, out_dir / "gpu_util.png")
    return out_dir
