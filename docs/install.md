---
description: Install strands-robots from source with uv, choose extras, and add LeRobot.
---

# Installation

strands-robots needs Python 3.10 or newer. LeRobot and Isaac Sim need Python 3.12.

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install "strands-robots[all] @ git+https://github.com/strands-labs/robots@main"
```

`[all]` installs every extra that works on a CPU machine. To install only some, name them:

```bash
uv pip install "strands-robots[mujoco,onnx,mesh] @ git+https://github.com/strands-labs/robots@main"
```

These docs follow the `dev` branch on GitHub, so install from there. The `strands-robots` release on PyPI is
an older API.

The core needs only `numpy` and `robot_descriptions`. If you use a feature without its extra, the
`ImportError` tells you what to install.

## Extras

| Extra | Needed for |
|---|---|
| `mujoco` | MuJoCo simulation |
| `feetech` | SO-100 and SO-101 arms |
| `onnx` | ONNX policies |
| `lerobot` | LeRobot policies (install LeRobot itself below) |
| `tools` | Strands agent tools |
| `mesh` | The Zenoh robot mesh |
| `iot` | The mesh over AWS IoT Core |
| `dashboard` | The browser dashboard |
| `voice` | Voice in the dashboard |
| `isaac` | Isaac Sim (CUDA, not in `[all]`) |
| `newton` | Newton GPU worlds (CUDA, not in `[all]`) |
| `dev` | Tests and linters |

## LeRobot

LeRobot is a normal dependency of the `lerobot` extra (its 0.7 release carries `remote_inference`), so `[lerobot]`
and `[all]` install it for you. `remote` adds `lerobot-policy-server` and Zenoh, `dataset` lets you record datasets,
and `feetech` drives the SO arms. LeRobot requires Python 3.12.

## Check the install

```python
import strands_robots

print(strands_robots.__version__)
```

## Work on the library

Clone it, install it editable, and run the tests:

```bash
git clone https://github.com/strands-labs/robots.git && cd robots
uv pip install -e ".[all,dev]"
pytest -q tests
```

## Build these docs

```bash
uv pip install -r docs/requirements.txt
mkdocs serve
```

Open `http://127.0.0.1:8000`.
