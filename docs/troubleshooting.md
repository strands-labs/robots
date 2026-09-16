---
description: Error → fix table for the most common gotchas across install, sim, hardware, policies, and mesh.
---

# Troubleshooting

## Diagnose first

`doctor` checks the Python version, which extras are importable, GPU/CUDA,
serial permissions, the MuJoCo GL backend, HuggingFace auth and a sim smoke
test, then prints a pass/fail table:

```bash
strands-robots doctor            # or: python -m strands_robots doctor
```

`strands-robots --help` lists the commands the package ships: `doctor` and
`verify-dataset`.

## Install

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: mujoco` | Missing `[sim-mujoco]` | `uv pip install "strands-robots[sim-mujoco]"` |
| `move_to: IK bridge unavailable: ... No module named 'mink'` | Missing `[sim-mujoco]` (the extra declares the IK solver) | `uv pip install "strands-robots[sim-mujoco]"` |
| `ModuleNotFoundError: lerobot` | Missing `[lerobot]` | `uv pip install "strands-robots[lerobot]"` |
| `training failed: 'accelerate' is required but not installed` | Missing LeRobot's `[training]` extra. `strands-robots[lerobot]` does not pull `accelerate` in, and `train()` requires it on CPU as well as GPU | `uv pip install "lerobot[training]"` |
| `ImportError: cannot import name '...' from 'lerobot'` | LeRobot version skew | `uv pip install "strands-robots[lerobot]"` (pins `lerobot>=0.6.1,<0.7.0`) |
| `ImportError: cannot import name 'MolmoAct2Policy'` | `lerobot < 0.6` (`MolmoAct2Policy` ships in lerobot >= 0.6) | `uv pip install "strands-robots[molmoact2]"` |
| pyav build fails on Jetson/aarch64 | No prebuilt wheel for sm_110 | Use `--no-build-isolation` or install `torchcodec>=0.7` and skip pyav. See [installation](getting-started/installation.md#molmoact2-on-jetson) |
| `numpy.dtype size changed` / `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` on Jetson | A wheel built against numpy 1.x (apt `python3-pandas`, an old cached wheel) imported under the numpy 2 that `[lerobot]` requires | In a venv, rebuild the offender through the extra so the resolver keeps lerobot's ranges: `uv pip install --reinstall-package pandas "strands-robots[lerobot]"`. A bare `--reinstall pandas` resolves pandas 3 / numpy 2.5 and leaves `lerobot` requiring `numpy<2.3.0`; pinning `numpy<2` is undone by the next install, since `lerobot >= 0.6` requires `numpy >= 2` |
| `uv pip install -e .` errors | Wrong cwd | `cd` to repo root first |
| `uv pip install` fails with `No virtual environment found; run uv venv` | `uv pip` installs into the active venv only and none is active | `uv venv --python 3.12 && source .venv/bin/activate`, then install; or `uv pip install --system` to opt out of the venv |

## Simulation

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `GLXBadFBConfig` (Linux) | Missing OSMesa | `sudo apt install libosmesa6-dev` + `export MUJOCO_GL=osmesa` |
| Black frames from `render(...)` | Headless, no GL backend | `export MUJOCO_GL=osmesa` (Linux) or `=egl` |
| Rendering very slow (~100x), warning `rendering on a CPU software rasterizer` | `MUJOCO_GL=egl` but NVIDIA's EGL vendor does not answer, so EGL falls back to Mesa `llvmpipe` (CPU) | On an NVIDIA host the library auto-registers the vendor ICD: when `libEGL_nvidia` is installed but no NVIDIA ICD is present, it stages one in `~/.strands_robots/egl_vendor.d/` and points glvnd at it via `__EGL_VENDOR_LIBRARY_FILENAMES` (no root needed) before importing mujoco. If the warning persists, ensure `NVIDIA_DRIVER_CAPABILITIES` includes `graphics`, or register the ICD system-wide: write `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` = `{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}`. Set `__EGL_VENDOR_LIBRARY_FILENAMES` yourself to opt out. If an NVIDIA ICD *is* already registered and `libEGL_nvidia` *is* installed, a missing ICD is not the cause - the driver is installed but unreachable from this process, so check `nvidia-smi` works here (a container can expose `/dev/nvidia*` and still deny opening it via the device cgroup, which glvnd can only report as this fallback). The warning names whichever of these three causes is still standing. Verify with `GL_RENDERER` (should report the NVIDIA GPU, not `llvmpipe`). |
| `Robot("foo")` raises ValueError | Unknown name | Check `list_robots("all")`; or pass `urdf_path=...` |
| `add_robot` refuses with `is registered but its model file is not on disk` | The name is correct - the registry knows the robot, its MJCF is just not present on any asset search path | Do what the refusal names. It prints the `<dir>/<model_xml>` it looked for and every path it searched. An entry it reports as `auto_download=false` (`google_robot`, `trossen_wxai`) never fetches its own asset - place the file there yourself, under `STRANDS_ASSETS_DIR`. Any other entry is fetchable: `download_assets(robots="<name>")` |
| Sim hangs on `create_world` | Asset download | Wait - first call downloads MJCF, then cached |
| `ModuleNotFoundError: trs_so_arm100_mj_description` | Auto-install failed | `uv pip install trs-so-arm100-mj-description` |
| `move_to: the requested POSE is not achievable ... The position ... on its own IS reachable` | The point is fine, the `orientation` is not - a damped least-squares solve honours the rotation and gives up the position, and an arm with fewer than 6 DOF (SO-100/SO-101) cannot realize an arbitrary full pose | Omit `orientation=` for a position-only solve, or command the pose on an arm with enough DOF. Loosening `tol` would only accept a solve that still points the wrong way |
| `move_to` reached the point but the wrist points the wrong way | `tol` bounds the POSITION in meters; the rotation is bounded by `orientation_tol` in radians (default 0.1, ~5.7 deg) | Read `orientation_error_rad` in the result's json block, and pass a tighter `orientation_tol=` if the default is too loose for the task |
| `move_to: 'orientation_tol' only bounds an 'orientation' target` | `orientation_tol` passed for a position-only move, where it would have nothing to bound | Pass `orientation=[w, x, y, z]` to command a full pose, or drop `orientation_tol` |
| `add_robot` raises after `load_scene` | Scene XML overrides world | Use `add_robot` before `load_scene` |
| `add_robot` refuses with `has no single index to drive it by` | The MJCF declares a multi-control actuator - mujoco 3.12 gave `<pid>` two controls (`input="pos vel"`) and `<orientation>` up to four, so the model compiles with more control slots than actuators. This backend addresses an actuator by its control index, which such an actuator has no single value for | Do what the refusal names. It prints each wide actuator and the control slots it owns, so the MJCF line is findable. Replace it with a single-control actuator (`<position>`, `<motor>`, `<velocity>`, `<intvelocity>`, `<general>`), or drive that joint outside this backend. The refused add leaves the scene exactly as it was, so a corrected model reuses the same robot name |
| `render(output_path=...)` refuses with `is outside the sandbox` | `output_path` resolved outside the render sandbox (`~/.strands_robots/renders`, `STRANDS_ROBOTS_RENDER_ROOT`, or this Simulation's `render_dir=`). Artifact sinks confine LLM-supplied paths | Write under the sandbox (a bare filename like `frame.png` is placed INTO it). A *relative* path with a directory part (`views/front.png`) is resolved against the process CWD, so it is refused; that refusal quotes the sandbox-anchored spelling to pass instead. Or, in code, construct with `Robot(name, render_dir="<the directory you want>")` so that directory IS the sandbox. Or set the variable the refusal names - `STRANDS_ROBOTS_RENDER_ALLOW_ABS=1` for `render`, `STRANDS_ROBOTS_VIDEO_ALLOW_ABS=1` for the video/recording sinks under `STRANDS_ROBOTS_VIDEO_ROOT` |
| `move_to` refuses with `is unreachable ... The same target solves to ... once the N degree(s) of freedom move_to does not command are free too` | The target needs motion `move_to` does not produce. It drives the end-effector frame the refusal names with position servos only, so a mobile base, a floating pelvis or any unactuated joint is not available to the solve - and 35 of the shipped sim robots have one | Move those degrees of freedom first (drive the base to the work area), then call `move_to`. The refusal's `uncommanded_joints_moved` names them and `unrestricted_ik_residual_m` is what the whole robot could reach |

## Hardware

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `PermissionError: /dev/ttyUSB0` | Not in `dialout` group | `sudo usermod -aG dialout $USER` + re-login |
| Arm twitches at startup | Stale calibration | Re-run `lerobot-calibrate` |
| Camera frames black | Wrong `index_or_path` | `lerobot_camera(action="list")` |
| Servo error mid-rollout | Velocity limit | Bump `control_frequency` or relax calibration limits |
| `Robot("so100", mode="real")` raises | Calibration missing | Run `lerobot-calibrate` first |
| Real robot moves wrong way | Joint mapping mismatch | Verify `data_config` matches recording |

## Policies

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `UntrustedRemoteCodeError` | `lerobot_local` needs HF exec | `export STRANDS_TRUST_REMOTE_CODE=1` |
| `Gr00tPolicy` connection refused | Container not running | `gr00t_inference(action="start_container", ...)` |
| `Gr00tPolicy` returns garbage | `data_config` mismatch | Use same `data_config` as training |
| `Cosmos3Policy` connection refused | Service not running | `uv pip install 'strands-robots[cosmos3-service]'` + start server |
| Policy import slow | Heavy dep at module top | Defer to `__init__` or `get_actions` |

## Recording

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `start_recording` / `DatasetRecorder.create` fails: `lerobot is not installed` | `[lerobot]` not installed | `uv pip install "strands-robots[lerobot]"` |
| `lerobot X is installed, but 'pyarrow' (or `datasets`, `pandas`, `av`, `torchcodec`), which its dataset stack needs, is not` | lerobot is installed **without** its `[dataset]` extra. Installing lerobot again does not pull those in | `uv pip install "lerobot[dataset]"` |
| `...importing its dataset stack failed ... a conflict between installed packages` | Nothing is missing (commonly a `pandas` built against a different `numpy`), so no install of lerobot or its extra fixes it | Reconcile the conflicting packages |
| Need MP4 without LeRobot | - | Use `start_cameras_recording` / `stop_cameras_recording` |
| Empty MP4 files | Stopped before any frames | Check `get_recording_status()` frame count |
| Push fails | Not logged into HF | `hf auth login` (the `huggingface-cli` entry point is not published at the declared `huggingface_hub>=1.5` floor) |

## Mesh

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `mesh.peers` empty | Other peer not running | Wait ~1s; verify `mesh.alive == True` on both |
| Port already bound | Another zenoh process | Mesh auto falls back to client mode; or set `STRANDS_MESH_PORT` |
| `mesh.alive` is `False`, `mesh.peers` stays empty | `eclipse-zenoh` missing (logged at WARNING: "eclipse-zenoh is not installed") | `uv pip install "strands-robots[mesh]"` |
| Want mesh off | - | `STRANDS_MESH=false` or `Robot(..., mesh=False)` |
| Peer is present and `connected`, but publishes no `state` (no joints) | A `_read_state` probe raised. Logged once per category at WARNING: "state probe 'hw_joints' failed" | Read the named probe: `hw_joints` is the motor bus (contended port, missing calibration), `sim_world` / `sim_joints` a sim back-reference, `task_state` the task record. Repeat failures are at DEBUG |

## Agent integration

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Agent picks wrong action | Tool spec confusion | Rephrase instruction; check `robot.tool_spec` |
| `Agent(tools=[robot])` errors | `strands-agents` missing | `uv pip install strands-agents` |
| Agent hangs | Long-running action | Bound the rollout: `run_policy(n_steps=...)`, or `stop_when={'predicate': ...}` to end it on a world state. On MuJoCo `start_policy` also returns immediately; on the other backends it is a blocking passthrough, so it is not the fix there |
| Bedrock/Anthropic auth fails | Provider credentials | See [Strands Agents docs](https://strandsagents.com/) |

Bug reports: [GitHub issues](https://github.com/strands-labs/robots/issues) - include `pip show strands-robots`, Python + OS, minimal repro, full stack trace.

## See also

- [Installation](getting-started/installation.md) - extras matrix.
- [Real hardware](hardware/robot-control.md) - bring-up sequence.
- [Contributing](contributing.md) - fix it yourself.
