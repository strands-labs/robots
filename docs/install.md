---
description: Install strands-robots from source with uv, choose the extras you need, and add LeRobot.
---

# Installation

strands-robots needs Python 3.10 or newer; LeRobot policies and the Isaac Sim backend need Python 3.12.
The core package depends only on `numpy` and `robot_descriptions`. Everything else is an optional extra, imported the
first time it is used; a missing extra raises an `ImportError` that names the install line.

## Install from source

```bash
git clone https://github.com/strands-labs/robots.git
cd robots
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[all]"
```

`[all]` installs every extra that installs from PyPI on a CPU machine. To install a subset, name the extras:

```bash
uv pip install -e ".[mujoco,onnx,mesh]"
```

## Extras

| Extra | Installs | Needed for |
|---|---|---|
| `mujoco` | `mujoco>=3.3.4` | the MuJoCo simulation backend, reading model files |
| `feetech` | `pyserial` | Feetech STS3215 buses (SO-100, SO-101) |
| `onnx` | `onnxruntime`, `huggingface_hub` | ONNX policies such as `holosoma` and the microduck skills |
| `lerobot` | `eclipse-zenoh` | LeRobot policies, in-process or over Zenoh (LeRobot itself is installed separately, below) |
| `tools` | `strands-agents`, `pillow` | `robot.tool`, `sim.tool`, `mesh.tool` for Strands agents |
| `mesh` | `eclipse-zenoh`, `cryptography` | the robot mesh over Zenoh |
| `iot` | `awsiotsdk`, `boto3`, `cryptography` | the mesh over AWS IoT Core |
| `dashboard` | `fastapi`, `uvicorn`, `websockets` | the browser dashboard |
| `isaac` | `isaacsim` | the Isaac Sim backend (CUDA, Python 3.12) |
| `newton` | `mujoco-warp`, `warp-lang` | the Newton backend, batched worlds on a GPU (CUDA) |
| `dev` | pytest, ruff, mypy, import-linter, onnx, moto | running the test suite and linters |

`isaac` and `newton` are not part of `[all]`; install them on a machine with CUDA.

## LeRobot

LeRobot policies use LeRobot's remote inference, which is on LeRobot's main branch. A wheel cannot depend on a git
revision, so install LeRobot with its own line:

```bash
uv pip install "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@53feaef37b67e8f4c23b6953e880ff535e39d6fe"
```

The `dataset` extra is what `run(record=...)` uses to write LeRobot v3 datasets. The `remote` extra provides
`lerobot-policy-server`.

## Check the install

```python
import strands_robots

print(strands_robots.__version__)
```

To run the test suite:

```bash
uv pip install -e ".[all,dev]"
pytest -q tests
```

## Build these docs locally

```bash
uv pip install -r docs/requirements.txt
mkdocs serve
```

The site is served at `http://127.0.0.1:8000`. If a dashboard is listening on `127.0.0.1:8090`, the home page embeds it
live; see [Dashboard](dashboard.md).
