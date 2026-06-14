---
description: Install strands-robots with uv — extras matrix, platform notes, headless rendering.
---

# Installation

Requires **Python >= 3.12**. Examples use [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`); plain `pip install` works too.

## Extras matrix

| Extra | Pulls in | When you need it |
|-------|----------|------------------|
| (none) | core only — Robot factory, registry, lazy imports | Inspect the catalog, write tools |
| `[sim]` | `robot_descriptions` | Sim asset resolution without MuJoCo |
| `[sim-mujoco]` | `sim` + `mujoco`, `imageio`, `imageio-ffmpeg` | Any `Robot()` with default `mode="sim"` |
| `[lerobot]` | `lerobot>=0.5.0,<0.6.0` | `LerobotLocalPolicy` + dataset recording |
| `[groot-service]` | `pyzmq`, `msgpack` | `Gr00tPolicy` (ZMQ to a GR00T container) |
| `[cosmos3-service]` | `msgpack`, `websockets` | `Cosmos3Policy` (WebSocket to Cosmos 3 server) |
| `[mesh]` | `eclipse-zenoh`, `json5` | Multi-robot mesh discovery + RPC |
| `[mesh-iot]` | `mesh` + `awsiotsdk`, `awscrt`, `boto3` | AWS IoT Core transport for mesh |
| `[benchmark-libero]` | `libero` eval deps | LIBERO benchmark suite |
| `[all]` | `groot-service` + `lerobot` + `sim-mujoco` + `mesh` + `mesh-iot` | Demos, CI, exploration |
| `[dev]` | `pytest`, `pytest-cov`, `ruff`, `mypy`, `pytest-timeout` | Contributing |

```bash
uv pip install "strands-robots[sim-mujoco]"                  # sim only
uv pip install "strands-robots[all]"                         # everything
uv pip install "strands-robots[sim-mujoco,cosmos3-service]"  # Cosmos 3
uv pip install "strands-robots[sim-mujoco,lerobot,mesh]"     # pick and choose
```

## Platform notes

**macOS:** works out of the box (arm64 + Intel).

**Linux (headless / real hardware):**
```bash
sudo apt install libosmesa6-dev ffmpeg
sudo usermod -aG dialout $USER   # USB serial access; re-login after
```

**Windows:** WSL2 + Ubuntu 22.04 (native Windows works for sim, not actively tested).

**Jetson / aarch64 (JetPack):**
```bash
uv pip install "numpy<2" "pandas==2.1.4"
uv pip install "strands-robots[sim-mujoco,lerobot]"
```

The `[lerobot]` extra includes `torchcodec` on aarch64 (required because
torchvision 0.26 removed `VideoReader` and lerobot's own torchcodec marker
excludes aarch64). If torch CUDA is needed on Jetson, ensure you install from
NVIDIA's index or set `UV_TORCH_BACKEND=auto`:

```bash
export UV_TORCH_BACKEND=auto   # resolves +cu130 wheels for Thor/Jetson
uv pip install "strands-robots[sim-mujoco,lerobot]"
```

### MolmoAct2 on Jetson (lerobot from source)

MolmoAct2 checkpoints (e.g. `allenai/MolmoAct2-SO100_101`) require lerobot
**from source** (git main) because `MolmoAct2Policy` was added after lerobot
0.5.1 (the latest PyPI release). See
[LeRobot Local: MolmoAct2](../policies/lerobot-local.md#molmoact2) for full
instructions. Quick path:

```bash
# Install lerobot from source (skips pyav if it fails on aarch64):
uv pip install "lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git" --no-build-isolation
uv pip install torchcodec>=0.7   # video decode backend for aarch64
```

This will be unnecessary once lerobot >= 0.5.2 is published to PyPI.

## Headless rendering

```bash
export MUJOCO_GL=osmesa     # software rendering — Linux
export MUJOCO_GL=egl        # hardware EGL
```

Or in Python before first import:
```python
import os
os.environ["MUJOCO_GL"] = "osmesa"
from strands_robots import Robot
```

## Verify

```python
from strands_robots import Robot

sim = Robot("so100")
sim.step()
obs = sim.get_observation("so100")
# obs is a flat dict mixing per-joint state floats and per-camera ndarrays:
#   {'shoulder_pan.pos': 0.0, ..., 'gripper.pos': 0.0, 'default': <HxWx3 uint8>}
print(list(obs.keys()))
```

Assets cache under `~/.strands_robots/assets/`.

## Environment variables

| Env var | What | Default |
|---------|------|---------|
| `STRANDS_ASSETS_DIR` | Robot model asset cache | `~/.strands_robots/assets/` |
| `STRANDS_MESH_AUDIT_DIR` | Safety audit log | `~/.strands_robots/` |
| `MUJOCO_GL` | GL backend | auto |
| `STRANDS_TRUST_REMOTE_CODE` | Allow HF `trust_remote_code=True` | `false` |
| `STRANDS_ROBOT_MODE` | Default `Robot()` mode | `sim` |
| `STRANDS_MESH` | Mesh enabled by default; set to `false` to disable globally | `true` (enabled) |
| `GROOT_API_TOKEN` | GR00T service API token (falls back from `api_token=` kwarg) | unset |

## See also

- [Quickstart](quickstart.md) — five minutes after install.
- [Robot factory](robot-factory.md) — every kwarg `Robot()` accepts.
- [Troubleshooting](../troubleshooting.md) — install gotchas.
