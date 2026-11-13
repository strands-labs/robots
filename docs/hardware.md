---
description: "Drive a real robot with the same Robot class: Feetech buses, Unitree DDS and the microduck. Drivers, cameras, calibration, the digital twin and the operator gate."
---

# Connecting hardware

A real robot is the same `Robot` class with a `port=`. A driver talks to the hardware. Pass `twin=True` and a
digital twin shows it in simulation.

{{drawing:d04_twin}}

<!-- test: skip, needs a real SO-101 arm on USB -->
```python
from strands_robots import Robot

real = Robot("so101", port="/dev/ttyACM0", twin=True, confirm=lambda a: input(f"move to {a}? ") == "y")
real.calibrate(interactive=False)  # loads the LeRobot calibration file
obs = real.observe()               # measured joints, in radians
real.confirm_act({"gripper": 1.0}, seconds=1.5)
real.twin.sim.render()             # the twin shows the measured pose
real.close()                       # torque off, port closed
```

## Drivers

| Driver | Hardware | Support |
|---|---|---|
| `feetech` | SO-100, SO-101 (STS3215 bus) | Read, write, torque, calibrate |
| `unitree` | G1, Go2, H1 over DDS | Observe only |
| `microduck` | The microduck's robotd socket | Observe only |
| `memory` | No hardware | Writes are stored and read back |

The registry picks the driver. Pass `driver=` to choose one. The twin is off by default, so a hardware-only install
needs no MuJoCo. Pass `twin=True` to mirror the robot in `Simulation.default()` (needs the `mujoco` extra).

## Try it without hardware

The `memory` driver acts like a real robot, so the gate and e-stop behave as they would on an arm.

```python
from strands_robots import Robot

real = Robot("so101", driver="memory", confirm=lambda targets: True)
real.confirm_act({"shoulder_pan": 0.3})
print(real.body.writes)  # the targets that reached the "bus"
```

## Driver options and cameras

Extra keywords go to the driver. The Feetech driver takes `robot_id=` and `baud=`. A misspelt option raises a
`TypeError` that lists the valid ones.

Name cameras by device name, not index. Indexes change when you plug in another camera.

```python
cameras = {"front": "Logi 4K", "wrist": {"index": 1, "width": 640, "height": 480, "fps": 30}}
```

The dict keys are the image keys the policy sees. A camera that delivers a different size than asked is refused.

`Robot(..., cameras=...)` opens every camera at connect, all at once, and refuses a camera that cannot open or sends
no frame within 10 s. Each camera then has its own capture thread, so a step takes each camera's latest frame without
waiting for the device. A camera whose latest frame is more than 1 s old has stopped delivering: it is left out of
the observation. `arm.body.cameras.ages()` gives the age in seconds of each frame the last step took.

{{drawing:d17_driver_options}}

## Calibration

Calibration is a Feetech bus's: the `memory` driver has none and takes no `calibration=`.
`calibrate(interactive=False)` loads the LeRobot file from `$HF_LEROBOT_CALIBRATION/robots/`, matched by
`Robot(id=)`. Pass `calibration="my_arm.json"` to use any file. `calibrate()` with no arguments walks you through
LeRobot's procedure at a terminal and saves the result. A Feetech calibration must fit the servo: `range_min` and
`range_max` in 0..4095, `homing_offset` in -2047..2047 and `drive_mode` 0 or 1, or connecting raises `ValueError`
naming the joint and the file. Each joint's span must also fit its limits: the driver maps the limits onto
`range_min..range_max`, so a span covering more than 1.5× the counts the limits take on the servo (4096 a turn) is
refused too, at connect and by `calibrate()` before it saves. A real SO-101 sits at 1.0–1.25×. `extras()` reports
`calibration` (the file) and `calibration_sha256` (a hash of its values) for the calibration in use.

## The operator gate

Your own code calls `act()`. Agents and the dashboard call `confirm_act()`, which asks your `confirm=` function
first. If it returns false, nothing is written. A real robot without `confirm=` refuses `confirm_act()`.

An agent's `run()` asks `approve=` once with the whole plan, not every step. Without `approve=`, the run is refused.
The safety chain still checks every step. When the policy is a checkpoint or dataset on this machine, the plan's
`files` is the SHA-256 of the files the loader reads (a preset name stays a preset, even next to a same-named folder).
The run then loads a private copy of those files, hashed while it is written: if they changed after the yes, the run
is refused and moves nothing, and a later write to the originals never reaches the robot. The copy is removed when
the run ends. The same holds for a checkpoint you registered as a policy object (`tool(policies={"pick":
Policy(dir)})`): it reads its files when it attaches, so it loads the approved copy; your object is not changed.
For a Hub checkpoint (`owner/repo`, as a spec or a `Policy`), `files` is `owner/repo@<commit>`, the commit its
revision points to when you are asked, and the run loads exactly that commit even if the branch moves. If the commit
cannot be looked up (offline), the run is refused before you are asked; pass `revision=` a full 40-character commit
to need no lookup. A spec no agent may pass (a network address) is refused before you are asked.

{{drawing:d16_one_yes_per_run}}

Every refusal raises `RequestRefused`. If a run faults, each joint goes to its declared `on_disconnect` state.

## Timing

Each step of `run()` reads the bus once and writes once. A late step counts in `RunResult.deadline_misses`.
Pass `on_miss="fault"` to stop the run at the first late step.

## Acceptance test

Run the hardware tests on a real SO-101:

```bash
STRANDS_SO101_PORT=/dev/ttyACM0 pytest tests_integ/acceptance
```

Without an arm, the tests skip. See [Safety and e-stop](safety.md) for the e-stop.
