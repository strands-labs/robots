#!/usr/bin/env bash
# Roll the droplet to one image tag, health-check it, roll back if it fails.
# Run on the droplet from /opt/strands-cloud (CI does this over SSH):  ./deploy.sh ghcr.io/<owner>/strands-xlerobot:<sha>
set -euo pipefail
cd "$(dirname "$0")"

new="${1:?usage: deploy.sh <image:tag>}"
prev="$(cat current_image 2>/dev/null || true)"

up() { IMAGE="$1" docker compose up -d --remove-orphans caddy dashboard && IMAGE="$1" docker compose restart caddy; }

healthy() {
  for _ in $(seq 1 60); do
    if curl -fsS --max-time 2 http://127.0.0.1:8090/api/health >/dev/null; then return 0; fi
    sleep 2
  done
  return 1
}

docker pull "$new"
up "$new"
if healthy; then
  echo "$new" > current_image
  [ -n "$prev" ] && [ "$prev" != "$new" ] && echo "$prev" > previous_image
  docker image prune -af --filter "until=168h" >/dev/null || true
  echo "deployed $new"
  exit 0
fi

echo "health check failed for $new" >&2
IMAGE="$new" docker compose logs --tail 80 dashboard >&2 || true
if [ -n "$prev" ]; then
  echo "rolling back to $prev" >&2
  up "$prev"
  healthy && echo "rolled back to $prev" >&2
fi
exit 1
