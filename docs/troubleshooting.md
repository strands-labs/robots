---
description: Error → fix table for the most common gotchas across install, sim, hardware, policies, and mesh.
---

# Troubleshooting

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
| numpy ABI mismatch on Jetson | System pandas vs pip numpy | `uv pip install "numpy<2" "pandas==2.1.4"` then reinstall |
| `uv pip install -e .` errors | Wrong cwd | `cd` to repo root first |

## Simulation

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `GLXBadFBConfig` (Linux) | Missing OSMesa | `sudo apt install libosmesa6-dev` + `export MUJOCO_GL=osmesa` |
| Black frames from `render(...)` | Headless, no GL backend | `export MUJOCO_GL=osmesa` (Linux) or `=egl` |
| Rendering very slow (~100x), warning `rendering on a CPU software rasterizer` | `MUJOCO_GL=egl` but no GPU EGL vendor ICD registered, so EGL falls back to Mesa `llvmpipe` (CPU) | On an NVIDIA host the library auto-registers the vendor ICD: when `libEGL_nvidia` is installed but no NVIDIA ICD is present, it stages one in `~/.strands_robots/egl_vendor.d/` and points glvnd at it via `__EGL_VENDOR_LIBRARY_FILENAMES` (no root needed) before importing mujoco. If the warning persists, ensure `NVIDIA_DRIVER_CAPABILITIES` includes `graphics`, or register the ICD system-wide: write `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` = `{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}`. Set `__EGL_VENDOR_LIBRARY_FILENAMES` yourself to opt out. Verify with `GL_RENDERER` (should report the NVIDIA GPU, not `llvmpipe`). |
| `Robot("foo")` raises ValueError | Unknown name | Check `list_robots("all")`; or pass `urdf_path=...` |
| Sim hangs on `create_world` | Asset download | Wait - first call downloads MJCF, then cached |
| `ModuleNotFoundError: trs_so_arm100_mj_description` | Auto-install failed | `uv pip install trs-so-arm100-mj-description` |
| `move_to: the requested POSE is not achievable ... The position ... on its own IS reachable` | The point is fine, the `orientation` is not - a damped least-squares solve honours the rotation and gives up the position, and an arm with fewer than 6 DOF (SO-100/SO-101) cannot realize an arbitrary full pose | Omit `orientation=` for a position-only solve, or command the pose on an arm with enough DOF. Loosening `tol` would only accept a solve that still points the wrong way |
| `move_to` reached the point but the wrist points the wrong way | `tol` bounds the POSITION in meters; the rotation is bounded by `orientation_tol` in radians (default 0.1, ~5.7 deg) | Read `orientation_error_rad` in the result's json block, and pass a tighter `orientation_tol=` if the default is too loose for the task |
| `move_to: 'orientation_tol' only bounds an 'orientation' target` | `orientation_tol` passed for a position-only move, where it would have nothing to bound | Pass `orientation=[w, x, y, z]` to command a full pose, or drop `orientation_tol` |
| `add_robot` raises after `load_scene` | Scene XML overrides world | Use `add_robot` before `load_scene` |
| `render(output_path=...)` refuses with `is outside the sandbox` | `output_path` resolved outside the render sandbox (`~/.strands_robots/renders`, or `STRANDS_ROBOTS_RENDER_ROOT`). Artifact sinks confine LLM-supplied paths | Write under the sandbox (a bare filename like `frame.png` is placed INTO it), or set the variable the refusal names - `STRANDS_ROBOTS_RENDER_ALLOW_ABS=1` for `render`, `STRANDS_ROBOTS_VIDEO_ALLOW_ABS=1` for the video/recording sinks under `STRANDS_ROBOTS_VIDEO_ROOT` |
| `move_to` refuses with `is unreachable ... The same target solves to ... once the N degree(s) of freedom move_to does not command are free too` | The target needs motion `move_to` does not produce. It drives the arm's position servos only, so a mobile base, a floating pelvis or any unactuated joint is not available to the solve - and 35 of the shipped sim robots have one | Move those degrees of freedom first (drive the base to the work area), then call `move_to`. The refusal's `uncommanded_joints_moved` names them and `unrestricted_ik_residual_m` is what the whole robot could reach |

## Hardware

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `PermissionError: /dev/ttyUSB0` | Not in `dialout` group | `sudo usermod -aG dialout $USER` + re-login |
| Arm twitches at startup | Stale calibration | Re-run `lerobot_calibrate` |
| Camera frames black | Wrong `index_or_path` | `lerobot_camera(action="list")` |
| Servo error mid-rollout | Velocity limit | Bump `control_frequency` or relax calibration limits |
| `Robot("so100", mode="real")` raises | Calibration missing | Run `lerobot_calibrate` first |
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
| `start_recording` fails: lerobot missing | `[lerobot]` not installed | `uv pip install "strands-robots[lerobot]"` |
| Need MP4 without LeRobot | - | Use `start_cameras_recording` / `stop_cameras_recording` |
| Empty MP4 files | Stopped before any frames | Check `get_recording_status()` frame count |
| Push fails | Not logged into HF | `huggingface-cli login` |

## Mesh

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `mesh.peers` empty | Other peer not running | Wait ~1s; verify `mesh.alive == True` on both |
| Port already bound | Another zenoh process | Mesh auto falls back to client mode; or set `STRANDS_MESH_PORT` |
| `init_mesh` raises | `eclipse-zenoh` missing | `uv pip install "strands-robots[mesh]"` |
| Want mesh off | - | `STRANDS_MESH=false` or `Robot(..., mesh=False)` |

## Agent integration

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Agent picks wrong action | Tool spec confusion | Rephrase instruction; check `robot.tool_spec` |
| `Agent(tools=[robot])` errors | `strands-agents` missing | `uv pip install strands-agents` |
| Agent hangs | Long-running action | Use `start_policy` instead of `run_policy` |
| Bedrock/Anthropic auth fails | Provider credentials | See [Strands Agents docs](https://strandsagents.com/) |

Bug reports: [GitHub issues](https://github.com/strands-labs/robots/issues) - include `pip show strands-robots`, Python + OS, minimal repro, full stack trace.

## See also

- [Installation](getting-started/installation.md) - extras matrix.
- [Real hardware](hardware/robot-control.md) - bring-up sequence.
- [Contributing](contributing.md) - fix it yourself.
