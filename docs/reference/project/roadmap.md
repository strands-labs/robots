---
description: What 1.0 keeps, what it replaces, how it lands, and what changes for a 0.5.x user.
---

# Roadmap: the road to 1.0

`main` is 0.5.x. Version 1.0 is a rewrite in progress. This page says what 1.0 keeps, replaces and changes for a 0.5.x user.

## Why a rewrite

The 0.5.x package is {{module_map_total}} lines of Python. It grew by adding: every provider, driver and simulation feature has a module, and the tool layer carries the agent envelope in many places. The layers hold ([architecture](../../concepts/architecture.md)), but the surface is wide and one idea (a driver, a unit conversion, a refusal) is spelled several ways. 1.0 works backwards from what a user needs: one `Robot`, six contracts, a package a team can hold in its head, and one command that proves it against real backends.

## What 1.0 is

The 1.0 design document is the build contract. Its shape:

- **Six contracts, one module each.** `Policy`, `Driver`, `SimEngine`, `Transport`, `Recorder`, `RobotLike`; a class that does two is two files. Every contract returns typed values and raises `core.errors` types; the agent envelope (`{"ok": ..., "refusal": {...}}`) exists only in `tools/` and `dashboard/`.
- **The same seven layers.** `core -> registry -> drivers | mesh -> sim | policies -> app -> tools -> dashboard`, with `scripts/check_layers.py` as referee. Siblings never import each other.
- **Typed observations and unit frames.** `Observation(joints, images, sensors, t, frame)`, `Action` as a mapping of joint name to float, and `UnitFrame(space, dof, scale, offset)` on every robot and policy. `robot.preflight(policy)` refuses a unit mismatch before any motion.
- **Motion is clamped and consented.** Every real write passes `core.motion.clamp_step` with the registry row's per-step travel bound; every tool defaults to `mode="sim"` and `dry_run=True`; real motion requires `confirm=True`; the Strands operator gate wraps the tool layer.
- **Wire formats are copied, never simplified.** Feetech, Dynamixel, Unitree DDS, Franka, UR, Robotiq, Crazyflie, Microduck, Reachy: the protocol files port byte for byte from 0.5.x; everything else is rewritten against the contract.
- **Cheap import.** `python -c "import strands_robots"` on a bare venv is a no-op; base dependency is numpy only; every heavy import goes through `core.optional.require(name, extra)`.
- **Acceptance instead of mocks.** `bash acceptance/all.sh` runs eleven scripts against real backends (MuJoCo step, checkpoint inference, RTC rollout, record and reopen, a real arm, teleop, two mesh peers, live dashboard, agent tools, preflight refusals, an MHS device) and prints `ACCEPTANCE GREEN (n real / m skipped)`.
- **Every `Robot` is an MHS device.** `app/mhs` mounts a robot on a Model Hardware Standard broker with a manifest from its registry row, fleet e-stop bridged to the mesh, and controls lowered to safe values on broker disconnect.

## What changes for you

| in 0.5.x | in 1.0 |
|---|---|
| `Robot(name)` returns a `MuJoCoSimEngine` or `hardware_robot.Robot` with different method sets | `Robot(name, mode)` returns one `RobotLike`: `observe()`, `act()`, `preflight()`, `is_connected()`, `close()` in both modes |
| the simulation tool exposes a large `action` vocabulary | twelve tools over one facade, each with `mode`, `dry_run`, `confirm` |
| `driver="lerobot"` is the default hardware path | native drivers are the default; lerobot is one driver among them |
| policies read raw observation dicts | policies receive a typed `Observation` and declare their `action_frame` |
| refusals are messages, five carry codes | every refusal carries a code from `core/refusals.py` and names the value, the cause and the remedy |
| one extra per feature area in `pyproject.toml`, `[all]` a curated subset | one extra per backend or driver, `[sim]` for MuJoCo, base install is numpy only |

Registry rows, robot names and aliases, wire protocols and safety postures carry over unchanged. A script using `Robot("so101")` with `send_action` and `get_observation` keeps working; `act` and `observe` are the 1.0 names.

## How it lands

1.0 lands in `strands-labs/robots` in reviewed pull requests once the acceptance suite is green on hardware as well as headless. Until then `main` stays 0.5.x and receives fixes.

No date is promised; done is the acceptance run.
