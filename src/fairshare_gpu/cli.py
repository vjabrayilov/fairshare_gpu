# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations


import argparse
from pathlib import Path

from rich.console import Console

from .analysis import analyze_run
from .benchmark import run_from_cli
from .plots import plot_all

console = Console()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fairshare-gpu", description="FairShare-GPU benchmark harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a benchmark from a YAML config")
    p_run.add_argument("--config", required=True, help="Path to experiment YAML config")
    p_run.add_argument("--out", required=True, help="Output directory (e.g., runs/my_run)")

    p_an = sub.add_parser("analyze", help="Analyze a run directory and write summary")
    p_an.add_argument("--run", required=True, help="Run directory (contains requests.jsonl)")
    p_an.add_argument("--out", default=None, help="Output directory for analysis (default: <run>/analysis)")
    p_an.add_argument("--slo", type=float, default=5.0, help="Latency SLO threshold in seconds")

    p_pl = sub.add_parser("plot", help="Generate plots from a run directory")
    p_pl.add_argument("--run", required=True, help="Run directory")
    p_pl.add_argument("--out", default=None, help="Output directory for figures (default: <run>/figures)")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        run_from_cli(args.config, args.out)
        return

    if args.cmd == "analyze":
        analyze_run(args.run, out_dir=args.out, slo_latency_s=args.slo)
        return

    if args.cmd == "plot":
        plot_all(args.run, out_dir=args.out)
        return

    raise SystemExit(2)
