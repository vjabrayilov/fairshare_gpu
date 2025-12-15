# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations


import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from .config import WorkloadConfig
from .tokenization import count_tokens


_BUCKETS = ("short", "medium", "long")


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str
    bucket: Literal["short", "medium", "long"]
    approx_tokens: Optional[int] = None


def _bucket_for_tokens(n: int) -> Literal["short", "medium", "long"]:
    # Heuristic buckets (tune as needed).
    if n <= 128:
        return "short"
    if n <= 512:
        return "medium"
    return "long"


class PromptSource:
    """Provides prompts for benchmarks.

    The prompt source supports:
    - synthetic prompts sized by characters
    - jsonl prompt sets (with optional token-length bucketing)

    The selection behavior is controlled by each tenant's `workload.mix`.
    """

    def __init__(self, cfg: WorkloadConfig, tokenizer_name_or_path: str, seed: int = 0):
        self.cfg = cfg
        self.tokenizer = tokenizer_name_or_path
        self.rng = random.Random(seed)

        self._prompts_by_bucket: Dict[str, List[Prompt]] = {b: [] for b in _BUCKETS}
        if cfg.kind == "jsonl":
            if not cfg.jsonl_path:
                raise ValueError("workload.kind='jsonl' requires workload.jsonl_path")
            self._load_jsonl(Path(cfg.jsonl_path))
        else:
            # Synthetic is generated on the fly (no preloading)
            pass

    def _load_jsonl(self, path: Path) -> None:
        prompts: List[Tuple[str, str]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, str):
                    text = obj
                elif isinstance(obj, dict):
                    text = obj.get("prompt") or obj.get("text") or obj.get("instruction") or ""
                else:
                    text = str(obj)
            except json.JSONDecodeError:
                text = line

            text = text.strip()
            if text:
                prompts.append((str(uuid.uuid4())[:8], text))

        if not prompts:
            raise ValueError(f"No prompts loaded from {path}")

        # Bucket prompts by token length
        for pid, text in prompts:
            n_tok = count_tokens(text, self.tokenizer)
            bucket = _bucket_for_tokens(n_tok)
            self._prompts_by_bucket[bucket].append(Prompt(prompt_id=pid, text=text, bucket=bucket, approx_tokens=n_tok))

        # Ensure each bucket has at least something; if not, fall back gracefully later.

    def _synthetic_prompt(self, bucket: Literal["short", "medium", "long"]) -> Prompt:
        # Generate pseudo-natural language text to approximate length.
        # Character lengths are from config; tokens will vary by tokenizer.
        target_chars = {
            "short": self.cfg.short_chars,
            "medium": self.cfg.medium_chars,
            "long": self.cfg.long_chars,
        }[bucket]

        words = [
            "GPU", "tenant", "fairness", "latency", "throughput", "batching", "scheduler", "prompt",
            "decode", "token", "inference", "memory", "utilization", "isolation", "MIG", "MPS",
            "vLLM", "TGI", "queueing", "SLO", "P99", "P95", "interference", "benchmark",
        ]
        out = []
        while sum(len(w) + 1 for w in out) < target_chars:
            out.append(self.rng.choice(words))
        text = " ".join(out) + "."
        pid = str(uuid.uuid4())[:8]
        # Token counting is optional for synthetic (we can compute later if needed)
        return Prompt(prompt_id=pid, text=text, bucket=bucket, approx_tokens=None)

    def sample(self, mix: Literal["short", "medium", "long", "mixed"], mixed_probs: List[float]) -> Prompt:
        if self.cfg.kind == "synthetic":
            if mix == "mixed":
                bucket = self.rng.choices(list(_BUCKETS), weights=mixed_probs, k=1)[0]
            else:
                bucket = mix
            return self._synthetic_prompt(bucket)  # type: ignore[arg-type]

        # jsonl mode
        if mix == "mixed":
            bucket = self.rng.choices(list(_BUCKETS), weights=mixed_probs, k=1)[0]
            candidates = self._prompts_by_bucket.get(bucket) or []
            if candidates:
                return self.rng.choice(candidates)

        # Non-mixed or bucket had no candidates: try requested bucket, else fall back.
        if mix != "mixed":
            candidates = self._prompts_by_bucket.get(mix) or []
            if candidates:
                return self.rng.choice(candidates)

        # Fall back to any available prompt
        all_prompts: List[Prompt] = []
        for b in _BUCKETS:
            all_prompts.extend(self._prompts_by_bucket.get(b) or [])
        if not all_prompts:
            raise RuntimeError("PromptSource has no prompts loaded.")
        return self.rng.choice(all_prompts)
