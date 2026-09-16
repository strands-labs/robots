---
description: Install strands-robots with uv - extras matrix, platform notes, headless rendering.
---

# Installation

Requires **Python >= 3.12**. Examples use [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`); plain `pip install` works too.

`uv pip install` installs into the active virtual environment and refuses when there is none (`No virtual environment found; run uv venv`), so create and activate one first:

```bash
uv venv --python 3.12
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

## Extras matrix

| Extra | Pulls in | When you need it |
|-------|----------|------------------|
| (none) | core only - Robot factory, registry, lazy imports | Inspect the catalog, write tools |
| `[sim]` | `robot_descriptions>=1.23.0,<2.0.0` | Sim asset resolution without MuJoCo |
| `[sim-mujoco]` | `sim` + `mujoco`, `imageio`, `imageio-ffmpeg` | Any `Robot()` with default `mode="sim"` |
| `[lerobot]` | `lerobot>=0.6.1,<0.7.0`, `psutil>=6.0.0,<8.0.0` | `LerobotLocalPolicy` + dataset recording + the `lerobot_train` / `lerobot_teleoperate` session tools |
| `[groot-service]` | `pyzmq`, `msgpack` | `Gr00tPolicy` (ZMQ to a GR00T container) |
| `[cosmos3-service]` | `msgpack`, `websockets>=17.0` | `Cosmos3Policy` (WebSocket to Cosmos 3 server) |
| `[earthrover]` | `requests>=2.28.0,<3.0.0` | `Robot("earthrover", mode="real", driver="strands")` - HTTP to the earth-rovers-sdk |
| `[ur]` | `ur-rtde>=1.6.0,<2.0.0` | `Robot("ur5e", mode="real", driver="strands")` - RTDE to a UR controller |
| `[mesh]` | `eclipse-zenoh>=1.6.1,<2.0.0`, `json5` | Multi-robot mesh discovery + RPC |
| `[mesh-iot]` | `mesh` + `awsiotsdk`, `awscrt`, `boto3` | AWS IoT Core transport for mesh |
| `[all]` | 20 of the 32 extras - **not** a union. `[cosmos3-diffusers]`, `[cosmos3-service]`, `[cosmos3-sim]`, `[crazyflie]` (GPLv3), `[curobo]`, `[microduck]`, `[ros2]`, `[sim-gs]`, `[sim-isaac]`, `[sim-newton]` and `[ur]` (compiled binding) stay opt-in | Demos, CI, exploration |
| `[dev]` | `pytest`, `pytest-cov`, `ruff`, `mypy`, `pytest-timeout` | Contributing |

```bash
# inside the activated venv from above
uv pip install "strands-robots[sim-mujoco]"                  # sim only
uv pip install "strands-robots[all]"                         # the 21-extra bundle
uv pip install "strands-robots[sim-mujoco,cosmos3-service]"  # Cosmos 3
uv pip install "strands-robots[sim-mujoco,lerobot,mesh]"     # pick and choose
```

The `[sim]` floor is set by the robot catalog rather than by an API: each entry in
the built-in registry names the `robot_descriptions` submodule that fetches its
MJCF and meshes, and that package gains one module per newly packaged robot.
`robot_descriptions` 1.23.0 is the oldest release providing a module for every
registered robot - on an older one, robots such as `so100` and `so101` have no
module to import and no fallback, so `Robot("so101", mode="sim")` cannot resolve
a model file.

The `websockets` floor in `[inference]` and `[cosmos3-service]` is set the same
way, by what the code needs rather than by preference - and by a *behaviour*
rather than a name. `PolicyServer.stop()` is documented to stop the server
serving, not merely listening; through websockets 16.x `Server.shutdown()` closed
the listening socket alone, so a client that was already connected went on being
answered with action chunks after the caller was told the server stopped, which on
a robot is the policy still driving the arm. websockets 17.0 closes the
connections it accepted and waits for their handlers. Measured against the
released wheels on unchanged sources: 16.1.1 still serves that client, 17.0 does
not - so both extras declare `>=17.0`. (13.0 remains the floor of the API *names*
reached for, `websockets.sync.server.Server`; the higher of the two wins.) They
declare the *same* floor on purpose: an environment resolves one `websockets`, so
two different floors would leave the lower one describing an install nobody gets.

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
uv pip install "strands-robots[sim-mujoco,lerobot]"
```

The same line as everywhere else. `lerobot >= 0.6` requires `numpy >= 2`, and
JetPack's torch (R38.2, torch 2.11 `+cu130`) runs on it - `strands-robots doctor`
passes on a Thor devkit with numpy 2.2.6. Do not pin `numpy < 2` first: the
resolver replaces it on this very line, so the pin buys nothing, and a package
that only works on numpy 1.x cannot share an environment with lerobot at all.

lerobot 0.6 pulls `torchcodec` on aarch64 itself (its dependency marker now
covers linux aarch64 and pins the torch-ABI-matched torchcodec 0.11), so the
video decoder resolves without a strands override. If torch CUDA is needed on
Jetson, ensure you install from NVIDIA's index or set `UV_TORCH_BACKEND=auto`:

```bash
export UV_TORCH_BACKEND=auto   # resolves +cu130 wheels for Thor/Jetson
uv pip install "strands-robots[sim-mujoco,lerobot]"
```

### MolmoAct2 on Jetson

MolmoAct2 checkpoints (e.g. `allenai/MolmoAct2-SO100_101`) resolve straight from
PyPI now that lerobot >= 0.6 ships `MolmoAct2Policy` (it was added after lerobot
0.5.1). See [LeRobot Local: MolmoAct2](../policies/lerobot-local.md#molmoact2)
for full instructions. Quick path:

```bash
# The [molmoact2] extra layers transformers, peft, scipy on top of lerobot >= 0.6;
# lerobot 0.6 pulls the aarch64 torchcodec decoder itself:
uv pip install "strands-robots[molmoact2]"
```

## Headless rendering

```bash
export MUJOCO_GL=osmesa     # software rendering - Linux
export MUJOCO_GL=egl        # hardware EGL
```

Or in Python before first import:
```python
import os
os.environ["MUJOCO_GL"] = "osmesa"
from strands_robots import Robot
```

## Verify

`doctor` checks this machine the way the runtime will read it - the interpreter
and package, each extra, the GL backend, the torch/torchcodec pair, the GPU, the
serial and Hub credentials, and the device-connect and mesh postures - and exits
non-zero if any row fails:

```bash
python -m strands_robots doctor           # run every check
python -m strands_robots doctor --list    # print the check names, probe nothing
```

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
| `STRANDS_MESH` | Set to `true` to opt a bare `Robot()` into the mesh; `false` disables it globally | unset (mesh off) |
| `GROOT_API_TOKEN` | GR00T service API token (falls back from `api_token=` kwarg) | unset |

## See also

- [Quickstart](quickstart.md) - five minutes after install.
- [Robot factory](robot-factory.md) - every kwarg `Robot()` accepts.
- [Troubleshooting](../troubleshooting.md) - install gotchas.
