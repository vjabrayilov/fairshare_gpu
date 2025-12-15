# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations


from functools import lru_cache
from typing import Optional

from transformers import AutoTokenizer


@lru_cache(maxsize=4)
def _load_tokenizer(name_or_path: str):
    # We intentionally avoid trust_remote_code unless you need it.
    # If your model requires it, set HF_TRUST_REMOTE_CODE=1 and we will respect it.
    import os

    trust = os.environ.get("HF_TRUST_REMOTE_CODE", "0") == "1"
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True, trust_remote_code=trust)
    # Some LLM tokenizers don't define pad token; set to eos for counting robustness.
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def count_tokens(text: str, tokenizer_name_or_path: str) -> int:
    """Return the number of tokens in `text` according to a Hugging Face tokenizer."""
    tok = _load_tokenizer(tokenizer_name_or_path)
    return len(tok.encode(text))


def safe_preview(text: str, max_chars: int = 160) -> str:
    """Shorten long text for logs/results."""
    t = text.replace("\n", " ").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."
