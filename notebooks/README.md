# strands-robots v1, end to end

Nine notebooks, saved with their outputs (videos, drawings, measured numbers). One story runs through all of them: an
SO-101 arm, with a Unitree G1 next to it, goes from a first robot to a policy-driven, recorded, safe, meshed,
agent-controlled device. Read them as they are; nothing needs to run.

| # | notebook | what it shows |
|---|---|---|
| 00 | [v1 at a glance](00_v1_at_a_glance.ipynb) | the twelve parts and their gates, the package measured against v0.5 (10,459 vs 258,503 lines), the ten top-level names, what still needs a real arm, credentials or a GPU |
| 01 | [one world](01_one_world.ipynb) | `Description`, the SO-101 and G1 in one world, observe / act / set_pose, cameras, both robots in one video, `Robot.run` → `RunResult`, `Simulation.run` in lock-step |
| 02 | [policies](02_policies.ipynb) | `Policy(...)` picks its provider by the form of the spec: a function, ONNX presets (the G1 stands, the duck walks), a LeRobot ACT checkpoint in-process and the same one over Zenoh, units, the LeRobot policy matrix from an L40S |
| 03 | [recording](03_recording.ipynb) | `run(record=dir)` writes LeRobot v3: 3 episodes with a camera, resume, reload with `LeRobotDataset`, `meta/units.json`, `hf://` targets |
| 04 | [device and safety](04_device_and_safety.ipynb) | the arm as a device: manifest, refused writes, the safety chain, estop mid-run, resume with approval, hold, state subscription, watchdog `on_disconnect` |
| 05 | [mesh](05_mesh.ipynb) | `Mesh`: mount, discover with selectors, invoke and refusals, state streams, broadcast estop, resume authority, `connect()` to a remote robot, `use_mesh`, a robot in a second process |
| 06 | [agent tools](06_agent_tools.ipynb) | `robot.tool` / `sim.tool` generated from the manifest, a scripted turn, a real Bedrock agent turn moving the arm, the operator gate on a hardware-like robot |
| 07 | [dashboard](07_dashboard.ipynb) | `Dashboard` serving the world to a browser twin, headless screenshots before and after a server-side act; [examples/leader_twin.py](examples/leader_twin.py) drives the sim arm from a real SO-101 leader |
| 08 | [GPU backends](08_gpu_backends.ipynb) | Isaac Sim and Newton results measured on an L40S (shown, not re-run without CUDA) |

## Run them again

From the folder above this one (Python 3.12):

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[mujoco,onnx,lerobot,mesh]" strands-agents fastapi "uvicorn[standard]" playwright ipykernel av
playwright install chromium          # 07 only
jupyter lab notebooks/               # or open the folder in VS Code, kernel "Python 3"
```

`_nb.py` is a small helper for these notebooks (inline MP4 video, the design drawings from `docs/assets/drawings`,
tables); it is not part of the package. 06 makes one real model call on Amazon Bedrock (us-west-2) and prints the error
instead when no credentials are configured. 00 compares against a v0.5 checkout when `STRANDS_ROBOTS_V05` points to one.
