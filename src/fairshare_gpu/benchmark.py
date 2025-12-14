from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .config import ExperimentConfig, load_config, dump_resolved_config
from .gpu_monitor import start_gpu_sampler, stop_gpu_sampler
from .records import RequestRecord
from .tokenization import count_tokens, safe_preview
from .workloads import PromptSource

from .clients.openai_client import chat_completion
from .clients.tgi_client import generate as tgi_generate

console = Console()


async def _writer_task(path: Path, queue: "asyncio.Queue[Dict[str, Any]]") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        while True:
            item = await queue.get()
            if item is None:  # type: ignore[comparison-overlap]
                break
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            queue.task_done()


async def _run_one_request(
    *,
    cfg: ExperimentConfig,
    tenant_id: str,
    endpoint: str,
    api_key: Optional[str],
    model: str,
    tokenizer: str,
    request_idx: int,
    prompt_source: PromptSource,
    tenant_mix: str,
    out_q: "asyncio.Queue[Dict[str, Any]]",
) -> None:
    prompt = prompt_source.sample(mix=tenant_mix, mixed_probs=cfg.workload.mixed_probs)
    prompt_text = prompt.text

    t_submit = time.time()

    max_tokens = next(t.max_tokens for t in cfg.tenants if t.id == tenant_id)
    temperature = next(t.temperature for t in cfg.tenants if t.id == tenant_id)
    top_p = next(t.top_p for t in cfg.tenants if t.id == tenant_id)

    if cfg.backend == "vllm_openai":
        res = await chat_completion(
            endpoint_base=endpoint,
            api_key=api_key,
            model=model,
            prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=cfg.benchmark.stream,
            timeout_s=cfg.benchmark.timeout_s,
        )
        out_text = res.text
        t_first = res.t_first_token
        t_done = res.t_done
        status_code = res.status_code
        err = res.error

        prompt_tok: Optional[int] = None
        out_tok: Optional[int] = None
        if res.usage:
            prompt_tok = res.usage.get("prompt_tokens")
            out_tok = res.usage.get("completion_tokens") or res.usage.get("output_tokens")
        if prompt_tok is None:
            prompt_tok = count_tokens(prompt_text, tokenizer)
        if out_tok is None:
            out_tok = count_tokens(out_text, tokenizer) if out_text else 0

        rec = RequestRecord(
            run_id=cfg.benchmark.run_name,
            tenant_id=tenant_id,
            request_idx=request_idx,
            prompt_id=prompt.prompt_id,
            t_submit=t_submit,
            t_first_token=t_first,
            t_done=t_done,
            prompt_tokens=prompt_tok,
            output_tokens=out_tok,
            prompt_preview=safe_preview(prompt_text),
            output_preview=safe_preview(out_text),
            backend=cfg.backend,
            endpoint=endpoint,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            status_code=status_code,
            error=err,
            extra={"bucket": prompt.bucket},
        )
        await out_q.put(rec.to_json())
        return

    # TGI
    res2 = await tgi_generate(
        endpoint_base=endpoint,
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=cfg.benchmark.stream,
        timeout_s=cfg.benchmark.timeout_s,
    )
    out_text = res2.text
    t_first = res2.t_first_token
    t_done = res2.t_done
    status_code = res2.status_code
    err = res2.error

    prompt_tok = count_tokens(prompt_text, tokenizer)
    out_tok = count_tokens(out_text, tokenizer) if out_text else 0

    rec = RequestRecord(
        run_id=cfg.benchmark.run_name,
        tenant_id=tenant_id,
        request_idx=request_idx,
        prompt_id=prompt.prompt_id,
        t_submit=t_submit,
        t_first_token=t_first,
        t_done=t_done,
        prompt_tokens=prompt_tok,
        output_tokens=out_tok,
        prompt_preview=safe_preview(prompt_text),
        output_preview=safe_preview(out_text),
        backend=cfg.backend,
        endpoint=endpoint,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        status_code=status_code,
        error=err,
        extra={"bucket": prompt.bucket},
    )
    await out_q.put(rec.to_json())


async def _tenant_closed_loop(
    *,
    cfg: ExperimentConfig,
    tenant_id: str,
    endpoint: str,
    api_key: Optional[str],
    model: str,
    tokenizer: str,
    prompt_source: PromptSource,
    tenant_mix: str,
    out_q: "asyncio.Queue[Dict[str, Any]]",
    end_time: float,
    worker_id: int,
) -> None:
    i = 0
    while time.time() < end_time:
        await _run_one_request(
            cfg=cfg,
            tenant_id=tenant_id,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            tokenizer=tokenizer,
            request_idx=(worker_id * 1_000_000_000) + i,
            prompt_source=prompt_source,
            tenant_mix=tenant_mix,
            out_q=out_q,
        )
        i += 1


async def _tenant_open_loop(
    *,
    cfg: ExperimentConfig,
    tenant_id: str,
    endpoint: str,
    api_key: Optional[str],
    model: str,
    tokenizer: str,
    prompt_source: PromptSource,
    tenant_mix: str,
    out_q: "asyncio.Queue[Dict[str, Any]]",
    end_time: float,
    rate_rps: float,
    max_in_flight: int,
) -> None:
    """Open-loop driver with a Poisson arrival process.

    We cap in-flight requests with a semaphore to avoid unbounded queue growth on the client side.
    """
    sem = asyncio.Semaphore(max_in_flight)
    rng = __import__("random").Random(cfg.benchmark.seed + hash(tenant_id) % 10_000)

    in_flight = set()

    async def _one(idx: int):
        try:
            await _run_one_request(
                cfg=cfg,
                tenant_id=tenant_id,
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                tokenizer=tokenizer,
                request_idx=idx,
                prompt_source=prompt_source,
                tenant_mix=tenant_mix,
                out_q=out_q,
            )
        finally:
            sem.release()

    idx = 0
    while time.time() < end_time:
        await sem.acquire()
        task = asyncio.create_task(_one(idx))
        in_flight.add(task)
        task.add_done_callback(in_flight.discard)  # type: ignore[arg-type]
        idx += 1

        # Poisson arrivals: exponential inter-arrival time
        gap = rng.expovariate(rate_rps) if rate_rps and rate_rps > 0 else 0.0
        if gap > 0:
            await asyncio.sleep(gap)
        else:
            await asyncio.sleep(0)

    # Drain outstanding requests before returning
    if in_flight:
        await asyncio.gather(*list(in_flight), return_exceptions=True)



async def run_benchmark(cfg: ExperimentConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve run id/name
    run_id = f"{cfg.benchmark.run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    cfg.benchmark.run_name = run_id

    # Persist resolved config
    dump_resolved_config(cfg, out_dir / "config.resolved.yaml")

    # Metadata
    start_time = time.time()
    end_time = start_time + cfg.benchmark.duration_s
    warmup_end = start_time + cfg.benchmark.warmup_s
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "start_time": start_time,
                "end_time": end_time,
                "warmup_end": warmup_end,
            },
            indent=2,
        )
    )

    # Start telemetry sampler
    gpu_handle = None
    if cfg.telemetry.enable_gpu:
        gpu_handle = start_gpu_sampler(
            out_csv=out_dir / "gpu.csv",
            gpu_index=cfg.telemetry.gpu_index,
            sample_ms=cfg.telemetry.sample_ms,
            record_processes=cfg.telemetry.record_processes,
        )

    # Prepare prompt source (shared across tenants to keep token bucketing consistent)
    prompt_source = PromptSource(cfg.workload, tokenizer_name_or_path=cfg.resolved_tokenizer(), seed=cfg.benchmark.seed)

    out_q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=10_000)
    writer = asyncio.create_task(_writer_task(out_dir / "requests.jsonl", out_q))

    # Run tenants
    tasks = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        t = progress.add_task(description=f"Running {run_id}", total=None)

        for tenant in cfg.tenants:
            mix = tenant.workload.mix
            if tenant.rate_rps is not None and tenant.rate_rps > 0:
                tasks.append(
                    asyncio.create_task(
                        _tenant_open_loop(
                            cfg=cfg,
                            tenant_id=tenant.id,
                            endpoint=tenant.endpoint,
                            api_key=tenant.api_key,
                            model=cfg.model,
                            tokenizer=cfg.resolved_tokenizer(),
                            prompt_source=prompt_source,
                            tenant_mix=mix,
                            out_q=out_q,
                            end_time=end_time,
                            rate_rps=tenant.rate_rps,
                            max_in_flight=tenant.max_in_flight,
                        )
                    )
                )
            else:
                for wid in range(tenant.concurrency):
                    tasks.append(
                        asyncio.create_task(
                            _tenant_closed_loop(
                                cfg=cfg,
                                tenant_id=tenant.id,
                                endpoint=tenant.endpoint,
                                api_key=tenant.api_key,
                                model=cfg.model,
                                tokenizer=cfg.resolved_tokenizer(),
                                prompt_source=prompt_source,
                                tenant_mix=mix,
                                out_q=out_q,
                                end_time=end_time,
                                worker_id=wid,
                            )
                        )
                    )

        while time.time() < end_time:
            await asyncio.sleep(0.2)
            progress.advance(t, 0)

        # Cancel tasks (they check end_time, so they should finish shortly)
        await asyncio.gather(*tasks, return_exceptions=True)

    # Flush writer
    await out_q.put(None)  # type: ignore[arg-type]
    await writer

    # Stop sampler
    stop_gpu_sampler(gpu_handle)

    console.print(f"[green]Done[/green]. Outputs in: {out_dir}")


def run_from_cli(config_path: str, out_dir: str) -> None:
    cfg = load_config(config_path)
    asyncio.run(run_benchmark(cfg, Path(out_dir)))
