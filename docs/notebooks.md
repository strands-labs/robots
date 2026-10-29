---
description: Eleven notebooks, one arm, start to finish. Already run, so you can just read them.
---

# Notebooks

[`notebooks/`](https://github.com/strands-labs/robots/tree/dev/notebooks) in the repo walks one SO-101 (with a G1 next to it) from "hello robot" to "an agent drives it over the
mesh". Every notebook is saved with its videos and numbers, so you can read it on GitHub without installing anything.

| | |
|---|---|
| [00 · quickstart](https://github.com/strands-labs/robots/blob/dev/notebooks/00_quickstart.ipynb) | a robot, a world, a render in ten lines |
| [01 · robot and sim](https://github.com/strands-labs/robots/blob/dev/notebooks/01_robot_and_sim.ipynb) | two robots, one clock, the run loop |
| [02 · recording](https://github.com/strands-labs/robots/blob/dev/notebooks/02_recording.ipynb) | LeRobot v3 datasets, read back with LeRobotDataset |
| [03 · agent builds a world](https://github.com/strands-labs/robots/blob/dev/notebooks/03_agent_builds_a_world.ipynb) | a real model edits the scene from English |
| [04 · agent drives the robot](https://github.com/strands-labs/robots/blob/dev/notebooks/04_agent_drives_the_robot.ipynb) | a real model moves the arm |
| [05 · policies](https://github.com/strands-labs/robots/blob/dev/notebooks/05_policies.ipynb) | functions, ONNX presets, a Hub checkpoint, units |
| [06 · remote inference](https://github.com/strands-labs/robots/blob/dev/notebooks/06_remote_inference.ipynb) | the same checkpoint over Zenoh, latency measured |
| [07 · dashboard](https://github.com/strands-labs/robots/blob/dev/notebooks/07_dashboard.ipynb) | the twin in a browser |
| [08 · device and safety](https://github.com/strands-labs/robots/blob/dev/notebooks/08_device_and_safety.ipynb) | the manifest, refusals, the red button |
| [09 · mesh](https://github.com/strands-labs/robots/blob/dev/notebooks/09_mesh.ipynb) | robots across processes |
| [10 · GPU backends](https://github.com/strands-labs/robots/blob/dev/notebooks/10_gpu_backends.ipynb) | Isaac Sim and MuJoCo-Warp, measured on an L40S |

Want to run them? `uv pip install -e ".[all]" ipykernel nbconvert av` (plus LeRobot, see the [quickstart](quickstart.md)) and open the folder.
