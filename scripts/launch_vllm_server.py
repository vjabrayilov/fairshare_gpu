#!/usr/bin/env python3
"""Convenience launcher for the vLLM OpenAI-compatible server.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import List


def build_cmd(
    *,
    host: str,
    port: int,
    model: str,
    dtype: str,
    max_num_batched_tokens: int,
    extra: List[str],
) -> List[str]:
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--dtype",
        dtype,
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
    ]
    cmd += extra
    return cmd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="bfloat16", help="float16 | bfloat16 (depends on GPU)")
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value (e.g., a MIG UUID or GPU index).",
    )
    p.add_argument(
        "--extra",
        default="",
        help="Extra args appended as-is to the vLLM command (quote as a single string).",
    )
    p.add_argument("--log", default=None, help="Optional log file path")
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    extra = shlex.split(args.extra) if args.extra else []
    cmd = build_cmd(
        host=args.host,
        port=args.port,
        model=args.model,
        dtype=args.dtype,
        max_num_batched_tokens=args.max_num_batched_tokens,
        extra=extra,
    )

    print("Launching:")
    print(" ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return

    stdout = None
    stderr = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = log_path.open("w")
        stdout = f
        stderr = subprocess.STDOUT

    subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)


if __name__ == "__main__":
    main()
