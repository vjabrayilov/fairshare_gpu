from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Backend = Literal["vllm_openai", "tgi_generate"]


class TenantWorkload(BaseModel):
    """Tenant-specific workload selector.

    mix:
      - short / medium / long: focuses on a token-length bucket
      - mixed: samples across buckets using `WorkloadConfig.mixed_probs`
    """

    mix: Literal["short", "medium", "long", "mixed"] = "mixed"


class TenantConfig(BaseModel):
    id: str = Field(..., description="Tenant identifier (e.g., A, B, C)")
    endpoint: str = Field(..., description="Base URL for the serving endpoint (e.g., http://localhost:8000)")
    api_key: Optional[str] = Field(None, description="Optional bearer token for OpenAI-compatible endpoints")

    # Load shaping (choose one)
    rate_rps: Optional[float] = Field(
        None,
        description="If set (>0), generate requests with a Poisson arrival process at this average rate (open-loop).",
    )
    concurrency: int = Field(
        1,
        description="If rate_rps is not set, run this many workers in a closed-loop (each sends next request after finishing).",
    )
    max_in_flight: int = Field(64, description="Upper bound on in-flight requests for this tenant (open-loop mode).")

    # Decode params (forwarded to backend)
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

    workload: TenantWorkload = Field(default_factory=TenantWorkload)


class WorkloadConfig(BaseModel):
    kind: Literal["synthetic", "jsonl"] = "synthetic"

    # jsonl settings
    jsonl_path: Optional[str] = None  # each line: {"prompt": "..."} or {"text": "..."} or raw string

    # synthetic settings
    short_chars: int = 256
    medium_chars: int = 1024
    long_chars: int = 4096

    # When tenant.mix == "mixed": probability over buckets [short, medium, long]
    mixed_probs: List[float] = Field(default_factory=lambda: [0.6, 0.3, 0.1])


class BenchmarkConfig(BaseModel):
    run_name: str = "run"
    duration_s: int = 120
    warmup_s: int = 10
    timeout_s: int = 120
    stream: bool = True
    seed: int = 0

    # request formatting
    request_style: Literal["chat"] = "chat"  # keep simple; vLLM OpenAI server supports chat


class TelemetryConfig(BaseModel):
    enable_gpu: bool = True
    gpu_index: int = 0
    sample_ms: int = 200
    record_processes: bool = False


class SLOConfig(BaseModel):
    latency_s: float = 5.0  # default SLO threshold for "attainment" computations


class ExperimentConfig(BaseModel):
    backend: Backend = "vllm_openai"
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer: Optional[str] = None

    tenants: List[TenantConfig]

    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    slo: SLOConfig = Field(default_factory=SLOConfig)

    def resolved_tokenizer(self) -> str:
        return self.tokenizer or self.model


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML config and validate it."""
    import yaml

    p = Path(path)
    data = yaml.safe_load(p.read_text())
    return ExperimentConfig.model_validate(data)


def dump_resolved_config(cfg: ExperimentConfig, out_path: str | Path) -> None:
    import yaml

    Path(out_path).write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
