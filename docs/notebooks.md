---
description: Nine notebooks, one arm, start to finish. Already run, so you can just read them.
---

# Notebooks

[`notebooks/`](https://github.com/strands-labs/robots/tree/dev/notebooks) in the repo walks one SO-101 (with a G1 next to it) from "hello robot" to "an agent drives it over the
mesh". Every notebook is saved with its videos and numbers, so you can read it on GitHub without installing anything.

| | |
|---|---|
| [00 · v1 at a glance](https://github.com/strands-labs/robots/blob/dev/notebooks/00_v1_at_a_glance.ipynb) | the parts, the package, what needs a real arm |
| [01 · one world](https://github.com/strands-labs/robots/blob/dev/notebooks/01_one_world.ipynb) | two robots, one physics |
| [02 · policies](https://github.com/strands-labs/robots/blob/dev/notebooks/02_policies.ipynb) | functions, ONNX skills, LeRobot local and served |
| [03 · recording](https://github.com/strands-labs/robots/blob/dev/notebooks/03_recording.ipynb) | datasets in three lines |
| [04 · device and safety](https://github.com/strands-labs/robots/blob/dev/notebooks/04_device_and_safety.ipynb) | the manifest, the bouncer, the red button |
| [05 · mesh](https://github.com/strands-labs/robots/blob/dev/notebooks/05_mesh.ipynb) | robots across processes |
| [06 · agent tools](https://github.com/strands-labs/robots/blob/dev/notebooks/06_agent_tools.ipynb) | a real model driving the arm |
| [07 · dashboard](https://github.com/strands-labs/robots/blob/dev/notebooks/07_dashboard.ipynb) | the twin in a browser |
| [08 · GPU backends](https://github.com/strands-labs/robots/blob/dev/notebooks/08_gpu_backends.ipynb) | Isaac and Newton |

Want to run them? `uv pip install -e ".[mujoco,onnx,lerobot,mesh,tools,dashboard]" ipykernel av` (plus LeRobot, see the [quickstart](quickstart.md)) and open the folder.
