#!/usr/bin/env bash
# setup.sh — one-command environment setup for Strands Robots + Device Connect
#
# Usage:
#   ./strands_robots/device_connect/setup.sh
#
set -euo pipefail

PYTHON_VERSION="3.12"
VENV_DIR=".venv"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "============================================================"
echo "  Strands Robots — Environment Setup"
echo "============================================================"
echo ""

# ── 0. Install uv (if needed) ────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "[0/2] uv not found — installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "[0/2] uv $(uv --version) ✓"
fi

# ── 1. Install Python (if needed) ────────────────────────────────────────────
if ! uv python find "$PYTHON_VERSION" &>/dev/null; then
  echo "[1/2] Python $PYTHON_VERSION not found — installing via uv..."
  uv python install "$PYTHON_VERSION"
else
  echo "[1/2] Python $PYTHON_VERSION ✓"
fi

# ── 2. Create virtual environment and install ────────────────────────────────
if [ ! -d "$REPO_ROOT/$VENV_DIR" ]; then
  echo "[2/2] Creating virtual environment and installing packages..."
  uv venv --python "$PYTHON_VERSION" "$REPO_ROOT/$VENV_DIR"
else
  echo "[2/2] Virtual environment exists, installing packages..."
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/$VENV_DIR/bin/activate"
uv pip install -e "$REPO_ROOT[sim-mujoco,device-connect]"

echo ""
echo "============================================================"
echo "  Setup complete"
echo "============================================================"
echo ""
echo "Activate the environment:"
echo "  source $REPO_ROOT/$VENV_DIR/bin/activate"
