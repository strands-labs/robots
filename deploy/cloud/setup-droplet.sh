#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 droplet. Run as root:
#   ssh root@<ip> 'bash -s' < deploy/cloud/setup-droplet.sh "<deploy user's ssh public key>"
# Installs Docker, opens 22/80/443 only, creates user `deploy` (docker group) for CI, makes /opt/strands-cloud.
set -euo pipefail
pubkey="${1:?pass the CI deploy public key}"

apt-get update
apt-get install -y ca-certificates curl ufw
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh

ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 443/udp
ufw --force enable

id deploy >/dev/null 2>&1 || useradd -m -s /bin/bash -G docker deploy
install -d -m 700 -o deploy -g deploy /home/deploy/.ssh
grep -qxF "$pubkey" /home/deploy/.ssh/authorized_keys 2>/dev/null || echo "$pubkey" >> /home/deploy/.ssh/authorized_keys
chown deploy:deploy /home/deploy/.ssh/authorized_keys
chmod 600 /home/deploy/.ssh/authorized_keys

# 2 GB swap: the image is large and sims spike memory.
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

install -d -o deploy -g deploy /opt/strands-cloud /opt/strands-cloud/strands /opt/strands-cloud/hf-cache
install -d -m 700 -o deploy -g deploy /opt/strands-cloud/dashboard-auth
echo "ready: create /opt/strands-cloud/.env (see deploy/cloud/.env.example), then run the Deploy workflow"
