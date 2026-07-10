# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP client for OpenAI-compatible LLM judge endpoints (Cloudflare etc.).

One ``JudgeClient`` per reward worker actor — owns its aiohttp session and a
semaphore that caps concurrent in-flight calls. Retries with exponential
backoff on transient errors. Supports gpt-oss ``reasoning_effort`` and
qwen-style ``enable_thinking`` via ``extra_body.chat_template_kwargs``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class JudgeClient:
    """Async OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        endpoint_url: str,
        model: str,
        api_key_env: str = "LLM_JUDGE_API_KEY",
        headers: Optional[dict[str, str]] = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
        max_concurrency: int = 32,
        max_tokens_param: str = "max_tokens",
        sock_connect_s: Optional[float] = None,
        sock_read_s: Optional[float] = None,
    ):
        if not endpoint_url:
            raise ValueError("endpoint_url is required")
        if not model:
            raise ValueError("model is required")

        self.endpoint_url = endpoint_url
        self.model = model
        # OpenAI reasoning models (o-series / gpt-5-*) require "max_completion_tokens"
        # and reject "max_tokens"; vLLM/gpt-oss use "max_tokens". Configurable so one
        # client class serves both. Default keeps the legacy "max_tokens" behaviour.
        self.max_tokens_param = max_tokens_param
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        # Opt-in fail-fast timeouts for a flapping/dropped connection. None => unset (legacy
        # behaviour, e.g. the non-streaming judge). Streaming callers (the E fleet) set
        # sock_read as an INTER-CHUNK idle bound, so a mid-stream tunnel drop aborts in
        # seconds instead of hanging the whole `timeout_s`; a healthy SSE stream is untouched.
        self.sock_connect_s = sock_connect_s
        self.sock_read_s = sock_read_s

        api_key = os.environ.get(api_key_env, "")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        else:
            logger.warning(
                "JudgeClient: env var %s is not set; sending requests without auth header",
                api_key_env,
            )
        if headers:
            self._headers.update(dict(headers))

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        # Lazy init: aiohttp requires a running event loop at construction time.
        if self._session is None or self._session.closed:
            # total = full (streamed) generation budget. sock_connect/sock_read are opt-in
            # (None => unset => legacy): set only by streaming callers (the E fleet) so a dead
            # connect or a mid-stream tunnel drop (no bytes for sock_read) aborts fast -> the
            # caller retries / fails over to a healthy sibling and the fleet breaker sees a
            # timely signal, instead of a ~60min hang blocking a rollout group. Left None for
            # the non-streaming judge so a long reward call is never cut short.
            timeout = aiohttp.ClientTimeout(
                total=self.timeout_s,
                sock_connect=self.sock_connect_s,
                sock_read=self.sock_read_s,
            )
            # limit=0 (unlimited) so aiohttp's default TCPConnector cap of 100 does NOT
            # throttle concurrency below the semaphore. Without this, an ExternalEClient
            # with max_concurrency>100 (e.g. 1024) was silently capped at 100 concurrent
            # E calls -> each per-layer E burst serialized into ~8 waves. The semaphore
            # (max_concurrency) is the real bound. Matches inference/e_remote.py.
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._headers,
                connector=aiohttp.TCPConnector(limit=0),
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        reasoning_effort: Optional[str] = None,
        thinking_mode: Optional[bool] = None,
        skip_special_tokens: Optional[bool] = None,
        extra_body: Optional[dict[str, Any]] = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Issue one chat-completions call. Returns content + telemetry."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self.max_tokens_param: max_tokens,
        }
        # Reasoning models (OpenAI o-series / gpt-5-*) only accept the default
        # temperature/top_p and 400 otherwise — pass None to omit them.
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p

        # gpt-oss-style reasoning effort (low/medium/high). Pass-through.
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort

        # vLLM-only: keep added special tokens (e.g. the meta-reasoning
        # <summary>/<direction>) in the returned text. MUST be top-level — vLLM
        # ignores it nested under extra_body. Omit (None) for OpenAI judges.
        if skip_special_tokens is not None:
            body["skip_special_tokens"] = skip_special_tokens

        # Stream the response as SSE so the tunnel forwards each token chunk
        # immediately. This resets the Cloudflare edge write-deadline on every
        # chunk, so long external-E generations don't hit the ~252s cutoff that
        # buffered (non-streaming) responses do on the Free plan.
        if stream:
            body["stream"] = True

        # qwen-style thinking-mode toggle. Pass through extra_body so the
        # server can route it into apply_chat_template kwargs.
        merged_extra: dict[str, Any] = {}
        if extra_body:
            merged_extra.update(extra_body)
        if thinking_mode is not None:
            merged_extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = bool(thinking_mode)
        if merged_extra:
            body["extra_body"] = merged_extra

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                session = await self._get_session()
                t0 = time.perf_counter()
                async with session.post(self.endpoint_url, json=body) as resp:
                    import json as _json

                    if resp.status >= 500 or resp.status == 429:
                        raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:500]}")
                    if resp.status >= 400:
                        # 4xx (other than 429) is a non-retryable client error.
                        raise PermissionError(f"HTTP {resp.status}: {(await resp.text())[:500]}")

                    if stream:
                        # SSE: accumulate delta content as it arrives, so each chunk the
                        # tunnel forwards resets the edge write-deadline and long
                        # generations don't get cut at ~252s. resp.content may yield
                        # arbitrary byte chunks (and aiohttp's line iterator caps a line at
                        # 64 KiB), so we buffer raw bytes and split on newlines ourselves —
                        # robust to multi-line chunks, lines split across reads, oversized
                        # lines, and UTF-8 chars split across reads. Mirrors the
                        # non-streaming path (choices[0], reasoning_content + content).
                        parts: list[str] = []
                        flags: dict[str, Any] = {"done": False, "finish": None}

                        def _feed(line_bytes: bytes) -> None:
                            line = line_bytes.decode("utf-8", "replace").strip()
                            if not line.startswith("data:"):
                                return  # blank line, SSE comment/keep-alive, or event:/id:
                            payload = line[5:].lstrip()
                            if payload == "[DONE]":
                                flags["done"] = True
                                return
                            try:
                                chunk = _json.loads(payload)
                            except Exception:  # noqa: BLE001 -- partial/garbled line, skip
                                return
                            if isinstance(chunk, dict) and chunk.get("error"):
                                # mid-stream error frame -> raise so the retry loop reruns
                                raise RuntimeError(f"SSE error frame: {str(chunk['error'])[:300]}")
                            choices = chunk.get("choices") or []
                            if choices:
                                ch = choices[0] or {}
                                delta = ch.get("delta") or {}
                                parts.append((delta.get("reasoning_content") or "") + (delta.get("content") or ""))
                                if ch.get("finish_reason"):
                                    flags["finish"] = ch["finish_reason"]

                        buf = b""
                        async for raw in resp.content.iter_any():
                            buf += raw
                            while b"\n" in buf:
                                line_bytes, buf = buf.split(b"\n", 1)
                                _feed(line_bytes)
                                if flags["done"]:
                                    break
                            if flags["done"]:
                                break
                        if not flags["done"] and buf.strip():
                            _feed(buf)  # stream may end without a trailing newline

                        # A well-formed SSE stream ends with [DONE] or at least a terminal
                        # finish_reason. Neither => the connection was cut mid-stream (e.g.
                        # a tunnel drop) and the text is truncated -> treat as transient so
                        # the retry loop reruns instead of grading partial content.
                        if not flags["done"] and flags["finish"] is None:
                            raise RuntimeError("SSE stream truncated (no [DONE]/finish_reason)")

                        return {
                            "content": "".join(parts),
                            "latency_s": time.perf_counter() - t0,
                            "attempts": attempt + 1,
                            "raw": None,
                            "finish_reason": flags["finish"],
                        }

                    text = await resp.text()
                    latency = time.perf_counter() - t0
                    data = _json.loads(text)
                    choice = data["choices"][0]
                    message = choice.get("message", {})
                    # OpenAI-compatible: the final answer is in `content`.
                    # gpt-oss-style servers may also emit a separate `reasoning`
                    # field that we *don't* want to grade on; if `content` is
                    # null/empty (typically because max_tokens cut off before
                    # the model exited reasoning), surface "" so parse_score
                    # cleanly hits the parse_failure path.
                    content = message.get("content") or ""
                    return {
                        "content": content,
                        "latency_s": latency,
                        "attempts": attempt + 1,
                        "raw": data,
                        "finish_reason": choice.get("finish_reason"),
                    }
            except PermissionError:
                # Don't retry on 4xx.
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries:
                    backoff = self.retry_backoff_s * (2**attempt)
                    logger.warning(
                        "JudgeClient retry %d/%d after %.1fs: %s",
                        attempt + 1,
                        self.max_retries,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)
        # Exhausted retries.
        assert last_exc is not None
        raise last_exc

    async def chat_with_semaphore(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Convenience wrapper: take the global semaphore around ``chat``."""
        async with self._semaphore:
            return await self.chat(*args, **kwargs)
