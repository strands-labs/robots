#!/usr/bin/env bash
# One-time: create the cloud stack's keys and store them as GitHub Actions secrets. Run on the desktop
# from the repo root (needs ssh-keygen and gh logged in):  bash deploy/cloud/init-secrets.sh
# Asks for a DigitalOcean API token (read + write; Applications > API > Generate New Token). Nothing is
# printed or left on disk except ~/.ssh/strands-deploy (to SSH into the droplet yourself).
set -euo pipefail
repo="${REPO:-Jacob-Brokloff/robots}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

key="$HOME/.ssh/strands-deploy"
[ -f "$key" ] || ssh-keygen -q -t ed25519 -N '' -C strands-deploy -f "$key"
ssh-keygen -q -t ed25519 -N '' -C strands-host -f "$tmp/host"

read -rsp "DigitalOcean API token: " token; echo
[ -n "$token" ] || { echo "no token" >&2; exit 1; }

printf '%s' "$token" | gh secret set DIGITALOCEAN_ACCESS_TOKEN -R "$repo"
gh secret set DO_SSH_KEY -R "$repo" < "$key"
gh secret set DO_HOST_KEY -R "$repo" < "$tmp/host"
echo "secrets set on $repo. SSH in later with: ssh -i $key deploy@<ip>"
