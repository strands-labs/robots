#!/usr/bin/env bash
# Disposable cloud stack on DigitalOcean (NYC3), driven by doctl.
#   cloud.sh up        create (or reuse) volume + reserved IP + droplet, wait until it is ready; prints the IP
#   cloud.sh down      destroy the droplet only; the volume (data, passkeys) and reserved IP stay
#   cloud.sh destroy   down + delete the volume and release the reserved IP (everything gone)
#   cloud.sh ip        print the droplet's reserved IP, or nothing if the droplet is down
# Env: DIGITALOCEAN_ACCESS_TOKEN (doctl), DO_SSH_KEY (deploy user's private key), DO_HOST_KEY (droplet's ed25519
# host private key). Optional: DO_REGION (nyc3), DO_SIZE (s-4vcpu-8gb), DO_VOLUME_GB (20).
set -euo pipefail
cd "$(dirname "$0")"

REGION="${DO_REGION:-nyc3}"
SIZE="${DO_SIZE:-s-4vcpu-8gb}"
VOLUME_GB="${DO_VOLUME_GB:-20}"
NAME=strands-control
VOLUME=strands-data
KEYNAME=strands-deploy

droplet_id() { doctl compute droplet list --tag-name strands --format ID,Name --no-header | awk -v n="$NAME" '$2==n{print $1}'; }
volume_id() { doctl compute volume list --region "$REGION" --format ID,Name --no-header | awk -v n="$VOLUME" '$2==n{print $1}'; }
# Our reserved IP: the one in REGION assigned to our droplet, else the first unassigned one there.
reserved_ip() {
  local d; d="$(droplet_id)"
  doctl compute reserved-ip list --format IP,Region,DropletID --no-header | awk -v r="$REGION" -v d="$d" '
    $2==r && d!="" && $3==d {print $1; found=1; exit}
    $2==r && ($3=="" ) && !free {free=$1}
    END {if (!found && free) print free}'
}
pub_of() { ssh-keygen -y -f "$1"; }

cmd_up() {
  : "${DO_SSH_KEY:?}" "${DO_HOST_KEY:?}"
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  printf '%s\n' "$DO_SSH_KEY" > "$tmp/deploy" && chmod 600 "$tmp/deploy"
  printf '%s\n' "$DO_HOST_KEY" > "$tmp/host" && chmod 600 "$tmp/host"
  deploy_pub="$(pub_of "$tmp/deploy")"; host_pub="$(pub_of "$tmp/host")"

  # DO account key (root gets it too, so DO never emails a root password).
  fp="$(ssh-keygen -E md5 -lf <(echo "$deploy_pub") | awk '{print $2}' | sed 's/^MD5://')"
  doctl compute ssh-key get "$fp" >/dev/null 2>&1 || doctl compute ssh-key import "$KEYNAME" --public-key-file <(echo "$deploy_pub") >/dev/null

  vid="$(volume_id)"
  if [ -z "$vid" ]; then
    vid="$(doctl compute volume create "$VOLUME" --region "$REGION" --size "${VOLUME_GB}GiB" --format ID --no-header)"
    echo "created volume $VOLUME" >&2
  fi

  ip="$(reserved_ip)"
  if [ -z "$ip" ]; then
    ip="$(doctl compute reserved-ip create --region "$REGION" --format IP --no-header)"
    echo "reserved IP $ip" >&2
  fi

  if [ -z "$(droplet_id)" ]; then
    awk -v hp="$host_pub" -v dp="$deploy_pub" -v vol="$VOLUME" -v keyfile="$tmp/host" '
      /@@HOST_KEY_PRIVATE@@/ { while ((getline line < keyfile) > 0) print "    " line; next }
      { gsub(/@@HOST_KEY_PUBLIC@@/, hp); gsub(/@@DEPLOY_PUBKEY@@/, dp); gsub(/@@VOLUME@@/, vol); print }
    ' cloud-init.yaml > "$tmp/user-data"
    doctl compute droplet create "$NAME" --region "$REGION" --size "$SIZE" --image ubuntu-24-04-x64 \
      --ssh-keys "$fp" --volumes "$vid" --tag-name strands --enable-monitoring \
      --user-data-file "$tmp/user-data" --wait >/dev/null
    echo "created droplet $NAME" >&2
  fi
  did="$(droplet_id)"
  doctl compute reserved-ip-action assign "$ip" "$did" >/dev/null 2>&1 || true

  # Wait for SSH + cloud-init (Docker installed, volume mounted).
  for _ in $(seq 1 60); do
    if ssh -i "$tmp/deploy" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 \
         "deploy@$ip" 'test -f /var/lib/strands-ready && docker info >/dev/null 2>&1' 2>/dev/null; then
      echo "$ip"; return 0
    fi
    sleep 10
  done
  echo "droplet did not become ready in 10 min" >&2; return 1
}

cmd_down() {
  did="$(droplet_id)"
  if [ -n "$did" ]; then doctl compute droplet delete "$did" --force; echo "droplet destroyed (volume + IP kept)" >&2
  else echo "no droplet" >&2; fi
}

cmd_destroy() {
  ip="$(reserved_ip)"; cmd_down
  sleep 5
  vid="$(volume_id)"; [ -n "$vid" ] && doctl compute volume delete "$vid" --force && echo "volume deleted" >&2
  [ -n "$ip" ] && doctl compute reserved-ip delete "$ip" --force && echo "reserved IP released" >&2
  true
}

cmd_ip() { [ -n "$(droplet_id)" ] && reserved_ip || true; }

case "${1:-}" in
  up) cmd_up ;; down) cmd_down ;; destroy) cmd_destroy ;; ip) cmd_ip ;;
  *) sed -n 2,9p "$0"; exit 2 ;;
esac
