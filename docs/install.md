---
description: Install strands-robots from source with uv, choose extras, and add LeRobot.
---

# Installation

strands-robots needs Python 3.10 or newer. LeRobot and Isaac Sim need Python 3.12.

```bash
git clone https://github.com/strands-labs/robots.git
cd robots
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[all]"
```

`[all]` installs every extra that works on a CPU machine. To install only some, name them:

```bash
uv pip install -e ".[mujoco,onnx,mesh]"
```

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

LeRobot comes from its main branch, so it has its own install line:

```bash
uv pip install "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@53feaef37b67e8f4c23b6953e880ff535e39d6fe"
```

`dataset` lets you record datasets. `remote` adds `lerobot-policy-server`.

## Check the install

```python
import strands_robots

print(strands_robots.__version__)
```

Run the tests:

```bash
uv pip install -e ".[all,dev]"
pytest -q tests
```

## Build these docs

```bash
uv pip install -r docs/requirements.txt
mkdocs serve
```

Open `http://127.0.0.1:8000`.
