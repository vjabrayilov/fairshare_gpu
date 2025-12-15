# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations


import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


@dataclass
class OpenAIStreamResult:
    text: str
    t_first_token: Optional[float]
    t_done: Optional[float]
    status_code: Optional[int]
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


async def chat_completion(
    *,
    endpoint_base: str,
    api_key: Optional[str],
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
    timeout_s: int,
) -> OpenAIStreamResult:
    """Call an OpenAI-compatible chat completion endpoint (e.g., vLLM OpenAI server).

    Parameters
    ----------
    endpoint_base:
        Like `http://localhost:8000`
    stream:
        If True, uses server-sent events and measures TTFT.
    """
    url = endpoint_base.rstrip("/") + "/v1/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }

    headers = {
        "Content-Type": "application/json",
        **_auth_headers(api_key),
    }

    timeout = httpx.Timeout(connect=10.0, read=None if stream else float(timeout_s), write=10.0, pool=10.0)

    t_first_token: Optional[float] = None
    t_done: Optional[float] = None
    usage: Optional[Dict[str, Any]] = None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if not stream:
                t0 = time.time()
                resp = await client.post(url, headers=headers, json=payload)
                status = resp.status_code
                if status >= 400:
                    return OpenAIStreamResult(
                        text="",
                        t_first_token=None,
                        t_done=time.time(),
                        status_code=status,
                        error=f"HTTP {status}: {resp.text[:2000]}",
                    )
                obj = resp.json()
                text = obj["choices"][0]["message"].get("content", "")
                usage = obj.get("usage")
                return OpenAIStreamResult(text=text, t_first_token=t0, t_done=time.time(), status_code=status, usage=usage)

            # streaming
            text_chunks = []
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                status = resp.status_code
                if status >= 400:
                    body = await resp.aread()
                    return OpenAIStreamResult(
                        text="",
                        t_first_token=None,
                        t_done=time.time(),
                        status_code=status,
                        error=f"HTTP {status}: {body[:2000]!r}",
                    )

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # Some servers attach usage in the final chunk.
                    if isinstance(obj, dict) and "usage" in obj:
                        usage = obj.get("usage")

                    try:
                        choice0 = obj.get("choices", [])[0]
                        delta = choice0.get("delta") or {}
                        chunk = delta.get("content") or ""
                        if chunk:
                            if t_first_token is None:
                                t_first_token = time.time()
                            text_chunks.append(chunk)
                    except Exception:
                        continue

            t_done = time.time()
            return OpenAIStreamResult(
                text="".join(text_chunks),
                t_first_token=t_first_token,
                t_done=t_done,
                status_code=200,
                usage=usage,
            )

    except Exception as e:
        return OpenAIStreamResult(
            text="",
            t_first_token=t_first_token,
            t_done=time.time(),
            status_code=None,
            usage=usage,
            error=str(e),
        )
