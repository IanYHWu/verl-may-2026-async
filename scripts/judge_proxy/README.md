# Judge proxy

A tiny hot-swappable reverse proxy for the LLM-judge endpoint used by the
reward manager (`verl/utils/judge`). The trainer points its judge at a **stable
loopback URL** (`http://127.0.0.1:8009/v1/chat/completions`); the proxy forwards
each request to whatever URL currently sits in `upstream.txt`, which it
**re-reads on every request**.

## Why

Self-hosted judges are often exposed via an **ephemeral** URL (e.g. a Cloudflare
quick tunnel `*.trycloudflare.com`) whose hostname changes whenever the tunnel
restarts. The trainer bakes `JUDGE_ENDPOINT_URL` in at init, so a churn would
silently collapse reward to 0 (or force a run restart). With the proxy in front:

- the trainer points at the fixed loopback URL **once**, and
- when the upstream changes you run `repoint.sh <new-url>` — **no proxy restart,
  no trainer restart**. The proxy picks up the new upstream on its next request.

It is a transparent passthrough (method + body + headers, minus hop-by-hop).
`JudgeClient` treats HTTP ≥500/429 as retryable, so a 502 during the swap gap
just makes the client retry — at worst one wasted step (`on_error_score=0`),
never a crash. Loopback ⇒ plain HTTP, so there is no TLS/SNI/cert problem.

## Files

| File | What |
|---|---|
| `judge_proxy.py` | the aiohttp server (env: `JUDGE_PROXY_PORT`, `JUDGE_UPSTREAM_FILE`, `JUDGE_PROXY_TIMEOUT_S`) |
| `start_proxy.sh` | start it durably (`setsid nohup`); refuses a second instance on the port |
| `repoint.sh` | rewrite `upstream.txt` to a new upstream URL (probes it first) |
| `upstream.txt` | the current full upstream URL (one line); **not committed** — seed from `upstream.txt.example` or `repoint.sh` |

## Usage

```bash
cd scripts/judge_proxy
./repoint.sh https://<judge-host>/v1/chat/completions   # seed upstream.txt
./start_proxy.sh                                         # start on 127.0.0.1:8009
curl -s http://127.0.0.1:8009/healthz                    # -> ok upstream=...

# then point the trainer at the proxy:
#   JUDGE_ENDPOINT_URL=http://127.0.0.1:8009/v1/chat/completions

# when the upstream churns, just:
./repoint.sh https://<new-judge-host>/v1/chat/completions
```

**Notes**
- The proxy is **node-local** (binds `127.0.0.1`). On a multi-node run, start one
  per node; if they share `upstream.txt` on a shared filesystem, a single
  `repoint.sh` updates all of them.
- `PROXY_PYTHON` overrides the interpreter (it needs `aiohttp`).
- `upstream.txt` may point at any OpenAI-compatible chat-completions endpoint
  (self-hosted, a tunnel, or a hosted API).
