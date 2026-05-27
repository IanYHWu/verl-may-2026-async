#!/usr/bin/env python3
"""Hot-swappable reverse proxy for the LLM judge endpoint.

The trainer points its judge at a STABLE loopback URL (http://127.0.0.1:PORT/...);
this proxy forwards each request to whatever URL currently sits in UPSTREAM_FILE,
which it re-reads on every request. So when the ephemeral Cloudflare tunnel churns
you just rewrite that one file (see repoint.sh) — no proxy restart, no trainer
restart. Loopback => plain HTTP, so there is no TLS/cert problem (the reason an
/etc/hosts hijack of the live run can't work: HTTPS SNI cert is for *.trycloudflare.com).

Transparent passthrough: forwards method + body + headers (minus hop-by-hop),
returns the upstream's status + body. JudgeClient (verl) treats HTTP >=500/429 as
retryable, so a 502 during the swap gap just makes the client retry — at worst one
wasted training step (on_error_score=0), never a crash.

Env:
  JUDGE_PROXY_PORT       (default 8009)        loopback port to listen on
  JUDGE_UPSTREAM_FILE    (required)            file holding the current full upstream URL
  JUDGE_PROXY_TIMEOUT_S  (default 300)         per-request upstream timeout (> client's 180s)
"""
from __future__ import annotations

import os

import aiohttp
from aiohttp import web

PORT = int(os.environ.get("JUDGE_PROXY_PORT", "8009"))
UPSTREAM_FILE = os.environ["JUDGE_UPSTREAM_FILE"]
TIMEOUT_S = float(os.environ.get("JUDGE_PROXY_TIMEOUT_S", "300"))

# Hop-by-hop / connection-specific headers we must NOT forward. Host is dropped so
# aiohttp sets it from the upstream URL (Cloudflare needs Host+SNI = the tunnel host);
# Content-Length is recomputed; Accept-Encoding dropped so we return plain bytes.
_DROP = {
    "host", "content-length", "connection", "keep-alive", "transfer-encoding",
    "te", "trailer", "upgrade", "proxy-authorization", "proxy-authenticate",
    "accept-encoding",
}


def _current_upstream() -> str:
    with open(UPSTREAM_FILE) as f:
        return f.read().strip()


async def _handle(request: web.Request) -> web.Response:
    body = await request.read()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP}
    upstream = _current_upstream()
    try:
        async with request.app["session"].request(
            request.method, upstream, data=body, headers=headers,
        ) as resp:
            data = await resp.read()
            out = web.Response(status=resp.status, body=data)
            ct = resp.headers.get("Content-Type")
            if ct:
                out.headers["Content-Type"] = ct
            return out
    except Exception as exc:  # noqa: BLE001 — surface as 502 so JudgeClient retries
        return web.Response(status=502, text=f"judge-proxy upstream error: {exc!r}")


async def _health(request: web.Request) -> web.Response:
    return web.Response(text=f"ok upstream={_current_upstream()}\n")


async def _on_startup(app: web.Application) -> None:
    conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=10)
    app["session"] = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=TIMEOUT_S), connector=conn,
    )


async def _on_cleanup(app: web.Application) -> None:
    await app["session"].close()


def main() -> None:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    app.router.add_get("/healthz", _health)
    app.router.add_route("*", "/{tail:.*}", _handle)
    web.run_app(app, host="127.0.0.1", port=PORT, access_log=None)


if __name__ == "__main__":
    main()
