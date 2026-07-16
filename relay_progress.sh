#!/bin/bash
# relay_progress.sh — tail data/PROGRESS.md and post each NEW line to the Discord webhook.
# The assistant appends milestone lines to data/PROGRESS.md; run THIS in tmux to get them on Discord:
#   tmux new -s relay
#   bash relay_progress.sh
# Webhook: $DISCORD_WEBHOOK_URL or the git-ignored .discord_webhook file next to this script.
cd "$(dirname "$0")"
HOOK="${DISCORD_WEBHOOK_URL:-$(cat .discord_webhook 2>/dev/null)}"
[ -z "$HOOK" ] && { echo "no webhook (.discord_webhook or \$DISCORD_WEBHOOK_URL)"; exit 1; }
F=data/PROGRESS.md
touch "$F"
echo "relaying new lines of $F -> Discord (Ctrl-C to stop)"
tail -n 0 -F "$F" | while IFS= read -r line; do
  [ -z "$line" ] && continue
  payload=$(python3 -c 'import json,sys; print(json.dumps({"content": sys.argv[1][:1990]}))' "$line")
  curl -s -o /dev/null -H "Content-Type: application/json" -d "$payload" "$HOOK"
done
