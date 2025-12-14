from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RequestRecord:
    """A single request/response measurement (one prompt -> one completion).

    All timestamps are in **unix seconds** as returned by `time.time()`.

    Notes:
    - When streaming is enabled, `t_first_token` is the first time we observed
      non-empty output content in the stream.
    - If an error occurs, `error` will be populated and token counts may be 0.
    """

    run_id: str
    tenant_id: str

    request_idx: int
    prompt_id: str

    # Timing
    t_submit: float
    t_first_token: Optional[float]
    t_done: Optional[float]

    # Token counts (best effort)
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]

    # Text (optional; for debugging keep small)
    prompt_preview: str = ""
    output_preview: str = ""

    # Serving backend metadata
    backend: str = ""
    endpoint: str = ""
    model: str = ""

    # Decode params
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    # Errors
    status_code: Optional[int] = None
    error: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
