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
    ):
        if not endpoint_url:
            raise ValueError("endpoint_url is required")
        if not model:
            raise ValueError("model is required")

        self.endpoint_url = endpoint_url
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

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
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout, headers=self._headers)
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
        extra_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Issue one chat-completions call. Returns content + telemetry."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        # gpt-oss-style reasoning effort (low/medium/high). Pass-through.
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort

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
                    text = await resp.text()
                    latency = time.perf_counter() - t0
                    if resp.status >= 500 or resp.status == 429:
                        raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
                    if resp.status >= 400:
                        # 4xx (other than 429) is a non-retryable client error.
                        raise PermissionError(f"HTTP {resp.status}: {text[:500]}")

                    import json as _json

                    data = _json.loads(text)
                    content = data["choices"][0]["message"]["content"]
                    return {
                        "content": content,
                        "latency_s": latency,
                        "attempts": attempt + 1,
                        "raw": data,
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
