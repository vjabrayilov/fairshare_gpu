from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


@dataclass
class TGIResult:
    text: str
    t_first_token: Optional[float]
    t_done: Optional[float]
    status_code: Optional[int]
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


async def generate(
    *,
    endpoint_base: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
    timeout_s: int,
) -> TGIResult:
    """Call a Hugging Face TGI server.

    Notes
    -----
    - Non-streaming endpoint: POST /generate
    - Streaming endpoint:     POST /generate_stream  (best-effort parsing)
    """
    endpoint_base = endpoint_base.rstrip("/")
    url = endpoint_base + ("/generate_stream" if stream else "/generate")

    # TGI uses max_new_tokens
    parameters: Dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
    }

    payload: Dict[str, Any] = {
        "inputs": prompt,
        "parameters": parameters,
    }

    timeout = httpx.Timeout(connect=10.0, read=None if stream else float(timeout_s), write=10.0, pool=10.0)

    t_first_token: Optional[float] = None
    t_done: Optional[float] = None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if not stream:
                t0 = time.time()
                resp = await client.post(url, json=payload)
                status = resp.status_code
                if status >= 400:
                    return TGIResult(text="", t_first_token=None, t_done=time.time(), status_code=status, error=resp.text[:2000])
                obj = resp.json()
                text = obj.get("generated_text", "")
                details = obj.get("details")
                return TGIResult(text=text, t_first_token=t0, t_done=time.time(), status_code=status, details=details)

            # streaming
            text_chunks = []
            details: Optional[Dict[str, Any]] = None
            async with client.stream("POST", url, json=payload) as resp:
                status = resp.status_code
                if status >= 400:
                    body = await resp.aread()
                    return TGIResult(text="", t_first_token=None, t_done=time.time(), status_code=status, error=body[:2000].decode("utf-8", errors="ignore"))

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    # Some deployments wrap events as SSE: "data: {...}"
                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()
                    if line == "[DONE]":
                        break
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Streaming objects often look like:
                    # {"token": {"text": "...", ...}, "generated_text": null, ...}
                    token = obj.get("token") or {}
                    chunk = token.get("text") or ""
                    if chunk:
                        if t_first_token is None:
                            t_first_token = time.time()
                        text_chunks.append(chunk)

                    # Final event may include generated_text/details
                    if obj.get("generated_text") is not None:
                        details = obj.get("details") or details

            t_done = time.time()
            return TGIResult(text="".join(text_chunks), t_first_token=t_first_token, t_done=t_done, status_code=200, details=details)

    except Exception as e:
        return TGIResult(text="", t_first_token=t_first_token, t_done=time.time(), status_code=None, error=str(e))
