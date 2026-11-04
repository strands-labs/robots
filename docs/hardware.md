---
description: "Drive a real robot with the same Robot class: Feetech buses, Unitree DDS and the microduck. Drivers, cameras, calibration, the digital twin and the operator gate."
---

# Connecting hardware

A real robot is the same `Robot` class with a `port=`. A driver talks to the hardware, and a digital twin shows it
in simulation.

{{drawing:d04_twin}}

<!-- test: skip, needs a real SO-101 arm on USB -->
```python
from strands_robots import Robot

real = Robot("so101", port="/dev/ttyACM0", confirm=lambda a: input(f"move to {a}? ") == "y")
real.calibrate(interactive=False)  # loads the LeRobot calibration file
obs = real.observe()               # measured joints, in radians
real.confirm_act({"gripper": 1.0}, seconds=1.5)
real.twin.sim.render()             # the twin shows the measured pose
real.close()                       # holds the arm (torque stays on); port and cameras close
```

`close()` holds a gravity-loaded arm in place rather than dropping it. Pass `disable_torque_on_disconnect=True` to release torque on close instead.

## Drivers

| Driver | Hardware | Support |
|---|---|---|
| `feetech` | SO-100, SO-101 (STS3215 bus) | Read, write, torque, calibrate |
| `unitree` | G1, Go2, H1 over DDS | Observe only |
| `microduck` | The microduck's robotd socket | Observe only |
| `memory` | No hardware | Writes are stored and read back |

The registry picks the driver. Pass `driver=` to choose one. Pass `twin=False` to skip the twin.

## Try it without hardware

The `memory` driver acts like a real robot, so the gate and e-stop behave as they would on an arm.

```python
from strands_robots import Robot

real = Robot("so101", driver="memory", twin=False, confirm=lambda targets: True)
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

{{drawing:d17_driver_options}}

## Calibration

`calibrate(interactive=False)` loads the LeRobot file from `$HF_LEROBOT_CALIBRATION/robots/`, matched by
`Robot(id=)`. Pass `calibration="my_arm.json"` to use any file. `calibrate()` with no arguments walks you through
LeRobot's procedure at a terminal and saves the result.

## The operator gate

Your own code calls `act()`. Agents and the dashboard call `confirm_act()`, which asks your `confirm=` function
first. If it returns false, nothing is written. A real robot without `confirm=` refuses `confirm_act()`.

An agent's `run()` asks `approve=` once with the whole plan, not every step. Without `approve=`, the run is refused.
The safety chain still checks every step.

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
