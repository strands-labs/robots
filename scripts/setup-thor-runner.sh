#!/bin/bash
# Quick setup using the token from github-runner-setup.md
# Run on Thor: scp this file to Thor then execute, OR paste into serial/ssh
set -e

RUNNER_TOKEN="ACGJKLURIEQL7JTFZIHFDN3JUFWKW"
REPO_URL="https://github.com/cagataycali/strands-gtc-nvidia"

echo "🤖 Setting up GitHub Actions Runner on Thor (aarch64)"
echo "======================================================"

# Create runner directory
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download ARM64 runner (Thor is aarch64, NOT x64!)
LATEST_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | grep -o '"tag_name": "v[^"]*"' | head -1 | cut -d'"' -f4 | sed 's/v//')
echo "📦 Latest runner version: ${LATEST_VERSION}"

if [ ! -f "run.sh" ]; then
  echo "⬇️  Downloading arm64 runner..."
  curl -fsSL "https://github.com/actions/runner/releases/download/v${LATEST_VERSION}/actions-runner-linux-arm64-${LATEST_VERSION}.tar.gz" -o runner.tar.gz
  tar xzf runner.tar.gz
  rm runner.tar.gz
  echo "✅ Extracted"
else
  echo "⏭️  Runner already exists, skipping download"
fi

# Configure
echo "🔧 Configuring runner..."
./config.sh \
  --url "${REPO_URL}" \
  --token "${RUNNER_TOKEN}" \
  --name "thor" \
  --labels "self-hosted,thor,jetson,gpu,arm64" \
  --work "_work" \
  --replace \
  --unattended

echo "✅ Runner configured!"

# Install systemd service
echo "🔄 Setting up systemd service..."
sudo tee /etc/systemd/system/github-runner.service > /dev/null << 'SVCEOF'
[Unit]
Description=GitHub Actions Runner (Thor)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=cagatay
WorkingDirectory=/home/cagatay/actions-runner
ExecStart=/home/cagatay/actions-runner/run.sh
Restart=always
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=30
Environment=DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
LimitNOFILE=65536
StandardOutput=journal
StandardError=journal
SyslogIdentifier=github-runner

[Install]
WantedBy=multi-user.target
SVCEOF

sudo systemctl daemon-reload
sudo systemctl enable github-runner
sudo systemctl start github-runner

echo ""
echo "✅ Done! Runner 'thor' is running as systemd service."
echo ""
echo "Commands:"
echo "  sudo systemctl status github-runner"
echo "  sudo journalctl -u github-runner -f"
echo ""
echo "Check: https://github.com/cagataycali/strands-gtc-nvidia/settings/actions/runners"
