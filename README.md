<p align="center"><img src="docs/assets/img/mark.svg" width="56" alt="Strands logo"></p>

<h1 align="center">strands-robots</h1>

<p align="center">A Python library for building robots: simulation, real hardware, policies and agents.</p>

<p align="center"><img src="docs/assets/media/agent_duck.gif" width="520" alt="An agent walks, turns and rolls the microduck"></p>

## Overview

strands-robots is built on three classes:

- **`Simulation`** is a world. MuJoCo runs on the CPU; Isaac Sim and Newton run on a CUDA GPU.
- **`Robot`** is a body in a world, or a real robot on a serial bus or network link.
- **`Policy`** turns an observation into an action: a Python function, an ONNX skill or a LeRobot checkpoint.

<!-- test: network -->
```python
from strands_robots import Policy, Robot

duck = Robot("microduck")
duck.run(Policy("microduck?vx=0.3"), seconds=10)   # walks about 1.3 m
```

## Features

| Area | What the library provides |
|---|---|
| Robot registry | 76 robots by name (arms, hands, humanoids, quadrupeds), MuJoCo Menagerie and `robot_descriptions` models, or your own MJCF/URDF. `Robot("g1")`, `Robot("panda")` and `Robot("my_arm.xml")` share one API. |
| Simulation | Several robots in one world with shared physics; agents can add robots, objects and cameras through `sim.tool`. |
| Hardware | The same `Robot` drives a real SO-101 over a Feetech bus, with a digital twin in simulation. |
| Policies | Python functions, ONNX skills, and LeRobot checkpoints in-process or served from another machine over Zenoh. Joint names, cameras and units are matched before anything moves, and a mismatch is refused by name. |
| Recording | `run(record=...)` writes a LeRobot v3 dataset, in simulation or on hardware. |
| Safety | Every robot generates a manifest, checks every outside command through one safety chain, and has an e-stop that latches in about 10 ms. |
| Mesh | Robots discover each other over Zenoh on a LAN, or over AWS IoT Core, accept remote commands, stream state, and stop on one broadcast e-stop. |
| Agent tools | `robot.tool`, `sim.tool` and `mesh.tool` for Strands agents, generated from each robot's manifest. |
| Dashboard | A browser view of a world with a live 3D twin, joint controls, an agent chat and e-stops. |

## Agents

```python
from strands import Agent

Agent(tools=[duck.tool])("walk forward a bit, turn left, do a roll, then stand still")
```

The agent selects skills (Pollen Robotics' open microduck policies), the skills drive the motors, and the run loop
enforces the safety chain. The animation above is a recorded agent turn.

[`notebooks/examples/agent_builds_a_scene.py`](notebooks/examples/agent_builds_a_scene.py) gives one agent `sim.tool`
and `arm.tool`; it places three cubes and a camera, moves the arm and renders the result.

## Installation

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[all]"
uv pip install "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@53feaef37b67e8f4c23b6953e880ff535e39d6fe"  # LeRobot main
pytest -q tests
```

## Documentation

- **Docs site:** [`docs/`](docs/). Build it with `pip install -r docs/requirements.txt && mkdocs serve` and open
  http://127.0.0.1:8000. With a dashboard running on port 8090, the home page embeds it live.
- **Notebooks:** [`notebooks/`](notebooks/) take one arm from a first observation to an agent driving it over the
  mesh. They are saved with their outputs and videos.

## Contributing

Open one pull request per change, with a test that fails before the change and passes after it, and a `changelog.d/`
fragment ([how](changelog.d/README.md)). Before you push:

```bash
pip install hatch
hatch run lint
hatch run test
```

Apache-2.0. See [LICENSE](LICENSE).
