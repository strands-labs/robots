---
description: Eleven notebooks that take one SO-101 from a first observation to an agent driving it over the mesh, saved with their outputs.
---

# Example notebooks

The [`notebooks/`](https://github.com/strands-labs/robots/tree/main/notebooks) folder has eleven short notebooks.
Each teaches one thing. They are saved with their outputs, so you can read them on GitHub.

| Notebook | Teaches |
|---|---|
| [00 quickstart](https://github.com/strands-labs/robots/blob/main/notebooks/00_quickstart.ipynb) | A robot, a world and an image |
| [01 robot and sim](https://github.com/strands-labs/robots/blob/main/notebooks/01_robot_and_sim.ipynb) | Two robots and the run loop |
| [02 recording](https://github.com/strands-labs/robots/blob/main/notebooks/02_recording.ipynb) | Recording a LeRobot dataset |
| [03 agent builds a world](https://github.com/strands-labs/robots/blob/main/notebooks/03_agent_builds_a_world.ipynb) | An agent edits the scene |
| [04 agent drives the robot](https://github.com/strands-labs/robots/blob/main/notebooks/04_agent_drives_the_robot.ipynb) | An agent moves the arm |
| [05 policies](https://github.com/strands-labs/robots/blob/main/notebooks/05_policies.ipynb) | Functions, ONNX and LeRobot policies |
| [06 remote inference](https://github.com/strands-labs/robots/blob/main/notebooks/06_remote_inference.ipynb) | A policy served over Zenoh |
| [07 dashboard](https://github.com/strands-labs/robots/blob/main/notebooks/07_dashboard.ipynb) | The dashboard |
| [08 device and safety](https://github.com/strands-labs/robots/blob/main/notebooks/08_device_and_safety.ipynb) | The manifest, refusals and e-stop |
| [09 mesh](https://github.com/strands-labs/robots/blob/main/notebooks/09_mesh.ipynb) | Robots in different processes |
| [10 GPU backends](https://github.com/strands-labs/robots/blob/main/notebooks/10_gpu_backends.ipynb) | Isaac Sim and Newton |

To run them, install the extras and LeRobot (see the [quickstart](quickstart.md)), then open the folder:

```bash
uv pip install -e ".[all]" ipykernel nbconvert av
```
