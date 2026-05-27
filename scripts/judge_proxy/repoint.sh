#!/usr/bin/env bash
# Repoint the judge proxy at a NEW upstream URL — the only thing you run when the
# Cloudflare tunnel churns. No proxy/trainer restart: the proxy re-reads this file
# per request. Usage: ./repoint.sh https://NEW-name.trycloudflare.com/v1/chat/completions
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEW_URL="${1:?usage: repoint.sh <new full upstream url incl. /v1/chat/completions>}"

case "$NEW_URL" in
  http://*|https://*) ;;
  *) echo "ERROR: url must start with http:// or https://" >&2; exit 1 ;;
esac

# Sanity-check the new upstream answers BEFORE we cut over (non-fatal warning).
code=$(curl -sS -m 15 -o /dev/null -w '%{http_code}' -X POST "$NEW_URL" \
        -H 'Content-Type: application/json' \
        -d '{"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' || echo "000")
echo "new upstream probe: HTTP $code"
[ "$code" = "200" ] || echo "WARNING: new upstream did not return 200 (got $code) — writing anyway; verify the tunnel is up."

printf '%s\n' "$NEW_URL" > "$HERE/upstream.txt"
echo "upstream.txt now: $(cat "$HERE/upstream.txt")"
echo "proxy /healthz:    $(curl -sS -m 5 http://127.0.0.1:${JUDGE_PROXY_PORT:-8009}/healthz || echo '(proxy not responding!)')"
