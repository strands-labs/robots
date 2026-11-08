<p align="center"><img src="docs/assets/img/mark.svg" width="56" alt="Strands logo"></p>

<h1 align="center">strands-robots</h1>

<p align="center">A Python library for building robots.</p>

<p align="center"><img src="docs/assets/media/agent_duck.gif" width="520" alt="An agent walks, turns and rolls the microduck"></p>

strands-robots runs the same code on a simulated robot and a real one. It has three main classes:

| Class | Is |
|---|---|
| `Simulation` | A world. MuJoCo on CPU, Isaac Sim and Newton on GPU. |
| `Robot` | A body in a world, or a real robot |
| `Policy` | Turns an observation into an action |

<!-- test: network -->
```python
from strands_robots import Policy, Robot

duck = Robot("microduck")
duck.run(Policy("microduck?vx=0.3"), seconds=10)  # walks about 1.3 m
```

## Features

| Area | Provides |
|---|---|
| Robots | 76 robots by name, or your own MJCF or URDF |
| Hardware | SO-100 and SO-101 arms, Unitree robots, with a digital twin |
| Policies | Python functions, ONNX skills, LeRobot checkpoints, local or remote |
| Recording | LeRobot v3 datasets |
| Safety | A manifest, one safety chain and an e-stop for every robot |
| ROS 2 | ros2_control robots and the Unitree G1 as a `Robot`, plus the `use_ros` agent tool |
| Mesh | Robots on other machines, over Zenoh or AWS IoT Core |
| Agent tools | `robot.tool`, `sim.tool` and `mesh.tool` for Strands agents |
| Dashboard | A 3D view, controls, cameras, chat, voice and e-stops in the browser |

## Agents

```python
from strands import Agent

Agent(tools=[duck.tool])("walk forward a bit, turn left, do a roll, then stand still")
```

The agent picks skills. The skills move the motors, and the safety chain checks each step.

## Installation

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install "strands-robots[all] @ git+https://github.com/strands-labs/robots@main"
uv pip install "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@main"
```

To work on the library itself, clone it and install it editable: `uv pip install -e ".[all,dev]"`, then
`pytest -q tests`.

## Documentation

- Docs: [`docs/`](docs/). Run `pip install -r docs/requirements.txt && mkdocs serve`, then open
  http://127.0.0.1:8000.
- Notebooks: [`notebooks/`](notebooks/), saved with their outputs.

## Contributing

Open one pull request per change. Include a test that fails before the change, and a `changelog.d/` fragment
([how](changelog.d/README.md)). Before you push:

```bash
pip install hatch
hatch run lint
hatch run test
```

Apache-2.0. See [LICENSE](LICENSE).
