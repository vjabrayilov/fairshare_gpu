# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations


import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


def _percentile(series: pd.Series, q: float) -> float:
    if series.empty:
        return float("nan")
    return float(np.nanpercentile(series.to_numpy(dtype=float), q))


def jains_fairness(values: Dict[str, float]) -> float:
    xs = np.array([v for v in values.values() if v is not None and not np.isnan(v)], dtype=float)
    n = len(xs)
    if n == 0:
        return float("nan")
    num = (xs.sum()) ** 2
    den = n * (xs**2).sum()
    if den == 0:
        return float("nan")
    return float(num / den)


def load_run(run_dir: str | Path) -> Tuple[pd.DataFrame, Dict]:
    run_dir = Path(run_dir)
    req_path = run_dir / "requests.jsonl"
    meta_path = run_dir / "meta.json"
    if not req_path.exists():
        raise FileNotFoundError(req_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rows = []
    for line in req_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    # Derived metrics
    df["latency_s"] = df["t_done"] - df["t_submit"]
    df["ttft_s"] = df["t_first_token"] - df["t_submit"]

    return df, meta


def summarize(
    df: pd.DataFrame,
    meta: Dict,
    *,
    slo_latency_s: float = 5.0,
    filter_warmup: bool = True,
) -> Dict:
    df = df.copy()

    # Warmup filter (by submit time)
    if filter_warmup and meta.get("warmup_end") is not None:
        df = df[df["t_submit"] >= float(meta["warmup_end"])].copy()

    # Success / error
    df["ok"] = df["error"].isna() & (df["status_code"].isna() | (df["status_code"] < 400))

    # Measurement window (post-warmup)
    if not df.empty:
        t0 = float(df["t_submit"].min())
        t1 = float(df["t_submit"].max())
    else:
        t0 = float(meta.get("warmup_end", meta.get("start_time", 0.0)))
        t1 = float(meta.get("end_time", t0))
    window_s = max(t1 - t0, 1e-6)

    tenants = sorted(df["tenant_id"].unique().tolist()) if "tenant_id" in df.columns else []

    per_tenant = {}
    throughput_by_tenant = {}

    for tid in tenants:
        dft = df[df["tenant_id"] == tid].copy()
        ok = dft[dft["ok"]].copy()
        err = dft[~dft["ok"]].copy()

        n_total = int(len(dft))
        n_ok = int(len(ok))
        n_err = int(len(err))
        err_rate = (n_err / n_total) if n_total else float("nan")

        # Throughput: output tokens / second (successful only)
        toks = ok["output_tokens"].fillna(0).astype(float).sum()
        toks_s = float(toks / window_s)
        throughput_by_tenant[tid] = toks_s

        per_tenant[tid] = {
            "tenant_id": tid,
            "requests_total": n_total,
            "requests_ok": n_ok,
            "requests_err": n_err,
            "error_rate": err_rate,
            "rps": float(n_ok / window_s) if window_s else float("nan"),
            "throughput_tokens_s": toks_s,
            "latency_p50_s": _percentile(ok["latency_s"], 50),
            "latency_p95_s": _percentile(ok["latency_s"], 95),
            "latency_p99_s": _percentile(ok["latency_s"], 99),
            "ttft_p50_s": _percentile(ok["ttft_s"], 50),
            "ttft_p95_s": _percentile(ok["ttft_s"], 95),
            "ttft_p99_s": _percentile(ok["ttft_s"], 99),
            "slo_attainment": float((ok["latency_s"] <= slo_latency_s).mean()) if n_ok else float("nan"),
        }

    fairness = jains_fairness(throughput_by_tenant)

    # Aggregate metrics (successful only)
    ok_all = df[df["ok"]].copy()
    toks_all = ok_all["output_tokens"].fillna(0).astype(float).sum()
    agg = {
        "requests_total": int(len(df)),
        "requests_ok": int(len(ok_all)),
        "requests_err": int((~df["ok"]).sum()),
        "throughput_tokens_s": float(toks_all / window_s),
        "rps": float(len(ok_all) / window_s),
        "latency_p50_s": _percentile(ok_all["latency_s"], 50),
        "latency_p95_s": _percentile(ok_all["latency_s"], 95),
        "latency_p99_s": _percentile(ok_all["latency_s"], 99),
        "ttft_p50_s": _percentile(ok_all["ttft_s"], 50),
        "ttft_p95_s": _percentile(ok_all["ttft_s"], 95),
        "ttft_p99_s": _percentile(ok_all["ttft_s"], 99),
        "fairness_jain_throughput": fairness,
        "window_s": window_s,
        "t0": t0,
        "t1": t1,
    }

    return {
        "aggregate": agg,
        "per_tenant": per_tenant,
    }


def analyze_run(run_dir: str | Path, out_dir: Optional[str | Path] = None, slo_latency_s: float = 5.0) -> Path:
    run_dir = Path(run_dir)
    if out_dir is None:
        out_dir = run_dir / "analysis"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, meta = load_run(run_dir)
    summary = summarize(df, meta, slo_latency_s=slo_latency_s)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Write CSVs for convenience
    per_tenant_df = pd.DataFrame(list(summary["per_tenant"].values()))
    per_tenant_df.to_csv(out_dir / "per_tenant.csv", index=False)

    pd.DataFrame([summary["aggregate"]]).to_csv(out_dir / "aggregate.csv", index=False)

    console.print(f"[green]Wrote[/green] {out_dir/'summary.json'}")
    return out_dir
