# Unitree G1 Humanoid Simulation Integration

This document explains the work added in the `unitree-g1-humanoid-simulation`
branch to make Strands Robots produce a usable MuJoCo-based Unitree G1
simulation demo on macOS.

## Goal

The original goal was to get Strands Robots to demonstrate humanoid
manipulation with a simulated Unitree G1 EDU robot, with an interactive control
path that follows the Strands policy abstraction layer instead of a one-off
script.

## Device Connect Dependency

Device Connect is the intended networking layer for multi-robot coordination.
The source repo is <https://github.com/Arm/device-connect>. This branch expects
the current packages from that repo:

- `device-connect-edge` for robot/device runtimes.
- `device-connect-agent-tools` for agent-side discovery and RPC invocation.

Older Strands Robots branches and notes may mention `device-connect-sdk`. This
branch keeps a compatibility shim for that legacy package name, but new code and
docs should prefer `device-connect-edge`.

Local source install:

```bash
gh repo clone Arm/device-connect /tmp/device-connect
.venv/bin/python -m pip install /tmp/device-connect/packages/device-connect-edge
.venv/bin/python -m pip install /tmp/device-connect/packages/device-connect-agent-tools
```

## What Was Missing In The Original Strands Robots Repo

The repository already contained useful building blocks:

- `unitree_g1` was already present in the robot registry.
- MuJoCo simulation support already existed.
- The policy abstraction layer already existed.
- MuJoCo viewer support already existed.

However, the repo was still missing several pieces required to get a working
local G1 demo on this Mac:

1. There was no end-to-end Unitree G1 MuJoCo demo script.
2. There was no interactive G1-specific natural-language policy provider that
   could run locally when a real VLA backend was unavailable.
3. The simulation asset cache assumed `~/.strands_robots/assets` was writable.
   In the local Codex sandbox it was not.
4. The direct MuJoCo simulation path depended on `strands-agents` runtime types
   even for local use, which blocked standalone simulation bring-up.
5. The interactive viewer path did not account for macOS requiring `mjpython`
   rather than plain `python` for passive MuJoCo viewer launch.
6. There was no written setup or state-of-work documentation for this specific
   Unitree G1 integration attempt.

## Assets And Model Sources

### What we investigated

During the initial investigation we looked at:

- `strandsagents.com` Strands Robots docs
- `strands-labs/robots`
- `unitreerobotics/unitree_mujoco`

The Unitree repository is still the most relevant official reference for:

- the G1 EDU mesh set
- official Unitree MuJoCo scene/config conventions
- future work if we want tighter fidelity to Unitree's own simulation assets

### What we actually used in this checkpoint

For this working checkpoint, we did **not** end up copying files directly from
`unitree_mujoco` into the repo.

Instead, the working local simulation uses the `unitree_g1` asset set resolved
through:

- `robot_descriptions`
- MuJoCo Menagerie-compatible asset downloads
- local cache: `.strands_robots/assets/unitree_g1`

That choice was pragmatic: it got MuJoCo loading and rendering quickly inside
the existing Strands simulation path with less asset-wrangling than building a
new loader for Unitree's official repo layout.

### Future asset direction

If higher fidelity to the Unitree G1 EDU package is required, the next step is
to compare the Menagerie-resolved `unitree_g1` files against:

- `unitree_mujoco/unitree_robots/g1`

and then decide whether to:

1. keep the current Menagerie-based path,
2. replace the G1 asset source with Unitree's official MuJoCo files, or
3. support both via explicit asset source selection.

## Code Added Or Changed

### Asset/cache fixes

- `strands_robots/assets/__init__.py`

Added fallback asset-cache resolution so local runs no longer fail when
`~/.strands_robots/assets` is not writable. The fallback now prefers:

1. the default home cache when available,
2. repo-local `.strands_robots/assets`,
3. a temp fallback.

### Standalone simulation compatibility

- `strands_robots/simulation/simulation.py`

Added a narrow fallback for `AgentTool`/tool event types so local MuJoCo
simulation can still run even when `strands-agents` is not installable from the
current Python index.

### New policy provider

- `strands_robots/registry/policies.json`
- `strands_robots/policies/g1_demo.py`

Added a new provider, `g1_demo`, that uses the existing policy abstraction
layer but maps simple natural-language instructions into deterministic G1 pose
sequences.

Important:

- `g1_demo` is **not** a trained VLA model.
- It is a local template-based provider used to make the simulator interactive
  through the same policy interface that real providers use.
- It is a bridge/demo policy so that the simulator workflow can be exercised
  now.

### New demo entry points

- `examples/mujoco_bed_making/humanoid_simulation/unitree_g1_mujoco_demo.py`
- `examples/mujoco_bed_making/humanoid_simulation/unitree_g1_interactive.py`

`unitree_g1_mujoco_demo.py`:

- builds a static scene,
- places a G1 robot, table, and cube,
- runs a scripted pose sequence,
- exports PNG frames.

`unitree_g1_interactive.py`:

- stages the same scene,
- opens the MuJoCo viewer,
- accepts live natural-language instructions,
- routes them through `run_policy(...)`,
- defaults to the new `g1_demo` provider,
- automatically re-execs under `mjpython` on macOS when viewer mode is used.

### Tests

- `tests/test_assets.py`
- `tests/test_g1_demo_policy.py`

These cover:

- asset-cache fallback behavior,
- registration/basic behavior of the `g1_demo` provider.

## Environment And Setup Notes

The working path on this Mac used a repo-local Python 3.12 virtual environment:

```bash
cd /Users/wahbro01/workspaces/git/robots
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install pillow mujoco robot_descriptions
export STRANDS_ASSETS_DIR=/Users/wahbro01/workspaces/git/robots/.strands_robots/assets
```

On macOS, the interactive viewer must run under `mjpython`. The interactive
script now handles that automatically.

## Current Demo State

### What works

- The G1 model resolves and loads in MuJoCo.
- A local asset cache is created and reused.
- The scripted G1 demo renders frames successfully in a normal interactive
  Terminal session.
- The interactive demo starts successfully and opens the MuJoCo viewer on
  macOS by re-launching itself under `mjpython`.
- Natural-language instructions can be entered live and are routed through the
  policy abstraction layer.

### What the current demo looks like

Interactive MuJoCo viewer:

![Unitree G1 interactive MuJoCo viewer](https://raw.githubusercontent.com/armwaheed/robots/bed-making-media/examples/mujoco_bed_making/humanoid_simulation/artifacts/unitree_g1_demo/interactive-viewer.webp)

Interactive terminal control loop:

![Unitree G1 interactive terminal commands](https://raw.githubusercontent.com/armwaheed/robots/bed-making-media/examples/mujoco_bed_making/humanoid_simulation/artifacts/unitree_g1_demo/interactive-terminal.webp)

### What is still rough / not physically correct

Observed issues in the current scene and controller include:

- the cube starts floating above the table,
- the cube settles onto the table only after simulation starts advancing,
- the table is too low relative to the humanoid,
- the current controller does not bend knees or waist sufficiently for grasping,
- the `g1_demo` provider is not a learned manipulation policy and does not use
  visual grounding.

This means the current result is a **working interactive simulation demo**, but
not yet a convincing humanoid manipulation system.

## Recommended Future Work

### 1. Fix the scene first

Before replacing the policy, clean up the simulation environment:

- lower the cube directly onto the tabletop at reset,
- tune object mass/friction/contact parameters,
- raise the table or move the robot base so the target is reachable,
- add a proper initial stabilization/reset step before user commands.

### 2. Improve the humanoid controller

The current demo policy only drives upper-body poses. The next controller pass
should:

- use waist pitch and knee flexion,
- support reach envelopes tied to target height,
- separate right-arm and left-arm manipulation plans cleanly,
- add explicit home/stand/recover states.

### 3. Replace the demo policy with a real provider

The current `g1_demo` provider is a placeholder to make the loop interactive.
To align more closely with the Strands Robots product promise, swap the
interactive script to a real policy provider, for example:

- `groot`
- `lerobot_local`
- `lerobot_async`
- another custom G1-capable inference backend

The script is already structured so the provider can be changed with:

```bash
python examples/mujoco_bed_making/humanoid_simulation/unitree_g1_interactive.py --policy <provider>
```

### 4. Revisit official Unitree assets

If EDU-specific geometry or behavior matters, compare and potentially migrate to
the Unitree official MuJoCo assets from:

- `unitree_mujoco/unitree_robots/g1`

### 5. Add explicit documentation for supported workflows

The repo should eventually document:

- headless demo workflow,
- interactive viewer workflow on macOS,
- venv setup,
- asset-cache expectations,
- how to plug a real VLA provider into the interactive G1 script.

## Summary

This branch does **not** complete a production-quality G1 manipulation stack.
It does establish the missing local integration path that the repo did not yet
provide:

- runnable G1 MuJoCo scene,
- interactive natural-language control loop,
- policy-layer integration,
- macOS viewer compatibility handling,
- repo-local asset cache fallback,
- basic tests and written handoff documentation.
