---
description: Error → fix table for the most common gotchas across install, sim, hardware, policies, and mesh.
---

# Troubleshooting

## Install

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: mujoco` | Missing `[sim-mujoco]` | `uv pip install "strands-robots[sim-mujoco]"` |
| `ModuleNotFoundError: lerobot` | Missing `[lerobot]` | `uv pip install "strands-robots[lerobot]"` |
| `ImportError: cannot import name '...' from 'lerobot'` | LeRobot version skew | `uv pip install "lerobot>=0.5.0,<0.6"` |
| `ImportError: cannot import name 'MolmoAct2Policy'` | MolmoAct2 not in PyPI lerobot (added post-0.5.1) | Install from source: `uv pip install "lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git"` |
| pyav build fails on Jetson/aarch64 | No prebuilt wheel for sm_110 | Use `--no-build-isolation` or install `torchcodec>=0.7` and skip pyav. See [installation](getting-started/installation.md#molmoact2-on-jetson-lerobot-from-source) |
| numpy ABI mismatch on Jetson | System pandas vs pip numpy | `uv pip install "numpy<2" "pandas==2.1.4"` then reinstall |
| `uv pip install -e .` errors | Wrong cwd | `cd` to repo root first |

## Simulation

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `GLXBadFBConfig` (Linux) | Missing OSMesa | `sudo apt install libosmesa6-dev` + `export MUJOCO_GL=osmesa` |
| Black frames from `render(...)` | Headless, no GL backend | `export MUJOCO_GL=osmesa` (Linux) or `=egl` |
| `Robot("foo")` raises ValueError | Unknown name | Check `list_robots("all")`; or pass `urdf_path=...` |
| Sim hangs on `create_world` | Asset download | Wait — first call downloads MJCF, then cached |
| `ModuleNotFoundError: trs_so_arm100_mj_description` | Auto-install failed | `uv pip install trs-so-arm100-mj-description` |
| `add_robot` raises after `load_scene` | Scene XML overrides world | Use `add_robot` before `load_scene` |

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
| Need MP4 without LeRobot | — | Use `start_cameras_recording` / `stop_cameras_recording` |
| Empty MP4 files | Stopped before any frames | Check `get_recording_status()` frame count |
| Push fails | Not logged into HF | `huggingface-cli login` |

## Mesh

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `mesh.peers` empty | Other peer not running | Wait ~1s; verify `mesh.alive == True` on both |
| Port already bound | Another zenoh process | Mesh auto falls back to client mode; or set `STRANDS_MESH_PORT` |
| `init_mesh` raises | `eclipse-zenoh` missing | `uv pip install "strands-robots[mesh]"` |
| Want mesh off | — | `STRANDS_MESH=false` or `Robot(..., mesh=False)` |

## Agent integration

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Agent picks wrong action | Tool spec confusion | Rephrase instruction; check `robot.tool_spec` |
| `Agent(tools=[robot])` errors | `strands-agents` missing | `uv pip install strands-agents` |
| Agent hangs | Long-running action | Use `start_policy` instead of `run_policy` |
| Bedrock/Anthropic auth fails | Provider credentials | See [Strands Agents docs](https://strandsagents.com/) |

Bug reports: [GitHub issues](https://github.com/strands-labs/robots/issues) — include `pip show strands-robots`, Python + OS, minimal repro, full stack trace.

## See also

- [Installation](getting-started/installation.md) — extras matrix.
- [Real hardware](hardware/robot-control.md) — bring-up sequence.
- [Contributing](contributing.md) — fix it yourself.
