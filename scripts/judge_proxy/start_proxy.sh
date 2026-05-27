#!/usr/bin/env bash
# Start the judge proxy durably (survives this shell). Idempotent-ish: refuses to
# start a second one if the port is already listening.
#
# Env:
#   JUDGE_PROXY_PORT       (default 8009)     loopback port to listen on
#   PROXY_PYTHON           (default python3)  interpreter with aiohttp installed
#   JUDGE_PROXY_TIMEOUT_S  (default 300)      per-request upstream timeout
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${JUDGE_PROXY_PORT:-8009}"
PY="${PROXY_PYTHON:-python3}"

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "port ${PORT} already listening — not starting a second proxy"; exit 0
fi

# The proxy reads the current upstream URL from this file on every request.
# Seed it with repoint.sh (or copy upstream.txt.example) before starting.
if [ ! -s "$HERE/upstream.txt" ]; then
  echo "ERROR: $HERE/upstream.txt is missing/empty." >&2
  echo "  Set it first:  ./repoint.sh https://<host>/v1/chat/completions" >&2
  exit 1
fi

export JUDGE_PROXY_PORT="$PORT"
export JUDGE_UPSTREAM_FILE="$HERE/upstream.txt"
export JUDGE_PROXY_TIMEOUT_S="${JUDGE_PROXY_TIMEOUT_S:-300}"

setsid nohup "$PY" "$HERE/judge_proxy.py" >>"$HERE/proxy.log" 2>&1 < /dev/null &
echo "started judge proxy (pid $!) on 127.0.0.1:${PORT}; log: $HERE/proxy.log"
sleep 2
echo "healthz: $(curl -sS -m 5 http://127.0.0.1:${PORT}/healthz || echo 'NOT UP YET')"
