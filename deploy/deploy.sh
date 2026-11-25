#!/usr/bin/env bash
# Deploy, roll back or inspect the ECG API container on the server.
#
#   ./deploy.sh <tag>        deploy ghcr.io/jawad1569/ecganomaly:<tag>  (e.g. a1b2c3d, latest, v2.0-milestone2)
#   ./deploy.sh <image:tag>  deploy any image reference (e.g. ecg-api:latest loaded with docker load)
#   ./deploy.sh --rollback   go back to the previously deployed image
#   ./deploy.sh --status     show what is running, /health and /stats
#
# A deploy only "sticks" if the new container becomes healthy (Docker HEALTHCHECK on
# /health) within HEALTH_TIMEOUT seconds. Otherwise the previous image is started again
# automatically and the script exits with an error.
#
# State lives in $STATE_DIR: current, previous, history.log
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/jawad1569/ecganomaly}"
NAME="${CONTAINER_NAME:-ecg-api}"
HOST_PORT="${HOST_PORT:-80}"
STATE_DIR="${STATE_DIR:-$HOME/.ecg-api}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-120}"

mkdir -p "$STATE_DIR"
current()  { cat "$STATE_DIR/current"  2>/dev/null || true; }
previous() { cat "$STATE_DIR/previous" 2>/dev/null || true; }
log()      { echo "$(date -u +%FT%TZ) $*" >> "$STATE_DIR/history.log"; echo "==> $*"; }

resolve_ref() {  # "a1b2c3d" -> "$IMAGE:a1b2c3d"; full references are kept as they are
  case "$1" in
    */*|*:*) echo "$1" ;;
    *) echo "$IMAGE:$1" ;;
  esac
}

start_container() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --restart unless-stopped -p "$HOST_PORT:8000" "$1" >/dev/null
}

wait_healthy() {
  local waited=0 status running restarts
  while [ "$waited" -lt "$HEALTH_TIMEOUT" ]; do
    running=$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo false)
    restarts=$(docker inspect -f '{{.RestartCount}}' "$NAME" 2>/dev/null || echo 0)
    status=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$NAME" 2>/dev/null || echo missing)
    echo "    health: $status (${waited}s)"
    [ "$status" = "healthy" ] && return 0
    [ "$status" = "unhealthy" ] && return 1
    # with --restart unless-stopped a crashing container restarts instead of staying exited
    if [ "$running" = "false" ] || [ "$restarts" -gt 0 ]; then echo "    container crashed"; return 1; fi
    sleep 3
    waited=$((waited + 3))
  done
  return 1
}

show_status() {
  echo "current:  $(current)"
  echo "previous: $(previous)"
  docker ps --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
  echo "--- /health"; curl -fsS "http://localhost:${HOST_PORT}/health" || echo "(no response)"; echo
  echo "--- /stats";  curl -fsS "http://localhost:${HOST_PORT}/stats"  || echo "(no response)"; echo
}

deploy() {
  local new old
  new=$(resolve_ref "$1")
  old=$(current)

  log "deploying $new (was: ${old:-nothing})"
  if ! docker pull "$new"; then
    # allow images loaded with `docker load` (deploy/ship_image.ps1)
    docker image inspect "$new" >/dev/null 2>&1 || { log "FAILED: image $new not found"; exit 1; }
    echo "    using local image $new"
  fi

  start_container "$new"
  if wait_healthy; then
    if [ -n "$old" ] && [ "$old" != "$new" ]; then echo "$old" > "$STATE_DIR/previous"; fi
    echo "$new" > "$STATE_DIR/current"
    log "OK: $new is healthy on port $HOST_PORT"
    docker image prune -f >/dev/null 2>&1 || true
    return 0
  fi

  echo "---- last logs of the failed container ----"
  docker logs --tail 30 "$NAME" 2>&1 || true
  if [ -n "$old" ]; then
    log "FAILED: $new not healthy, rolling back to $old"
    start_container "$old"
    wait_healthy && log "rolled back to $old" || log "WARNING: rollback container $old is not healthy either"
  else
    log "FAILED: $new not healthy and there is no previous image to roll back to"
  fi
  exit 1
}

case "${1:-}" in
  --rollback)
    prev=$(previous)
    [ -n "$prev" ] || { echo "No previous image recorded in $STATE_DIR/previous"; exit 1; }
    deploy "$prev"
    ;;
  --status) show_status ;;
  ""|-h|--help) sed -n '2,13p' "$0" ;;
  *) deploy "$1" ;;
esac
