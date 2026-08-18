---
description: HardwareRobot - async task execution, status reporting, the LeRobot bridge.
---

# Robot control (real hardware)

`Robot(name, mode="real", ...)` returns a `strands_robots.hardware_robot.Robot`.

```python
from strands_robots import Robot

robot = Robot(
    "so100",
    mode="real",
    cameras={"wrist": {"type": "opencv", "index_or_path": "/dev/video0"}},
    port="/dev/tty.usbserial-A50285BI",
    control_frequency=50.0,
)

robot.start_task(
    instruction="pick up the cube",
    policy_provider="groot",
    policy_port=5555,
    duration=30.0,
)

status = robot.get_task_status()
robot.stop_task()
robot.cleanup()
```

## Constructor parameters

| Param | What |
|-------|------|
| `tool_name` | Tool identifier for the agent. |
| `robot` | LeRobot `Robot` instance, `RobotConfig`, or string (e.g. `"so100"`). |
| `cameras` | `{name: config_dict}`. Config keys are `type` (backend selector, `opencv`) plus the fields of lerobot's `OpenCVCameraConfig`: `index_or_path` (required), `fps`, `width`, `height`, `color_mode`, `rotation`, `warmup_s`, `fourcc`, `backend`. An unknown key raises `ValueError`. |
| `action_horizon` | Actions per inference step (default 8; must be a positive integer). |
| `data_config` | GR00T data_config name. |
| `control_frequency` | Control loop Hz (default 50). |
| `**kwargs` | Forwarded to LeRobot backend (`port`, `robot_ip`, `kp`, `kd`, …). Unknown kwargs raise `ValueError`. |

## Task lifecycle

`TaskStatus`: `IDLE` → `CONNECTING` → `RUNNING` → `COMPLETED` / `STOPPED` / `ERROR`

| Method | What |
|--------|------|
| `start_task(instruction, policy_port, policy_host, policy_provider, duration)` | Async; returns immediately. |
| `stop_task()` | Halt the current task. Covers a task still in `CONNECTING` (bring-up): the rollout is abandoned before the arm is commanded. |
| `get_task_status()` | Returns `RobotTaskState` (status, step count, error). |
| `cleanup()` | Stop tasks, disconnect the robot (motors bus + every camera), stop mesh. Terminal - see below. |
| `stop()` | Async spelling of `cleanup()`; delegates to it off the event loop. Terminal. |

One rollout at a time: the arm has a single command bus, so `start_task` /
`run_policy` / the `execute` action refuse while another task is in flight and
name it in the error. That includes the `CONNECTING` bring-up window - a motors
bus handshake plus per-camera warmup, seconds on a real arm - not just
`RUNNING`. Call `stop_task()` to hand the bus over early.

Every rollout knob is judged before that bring-up window, not inside it.
`duration` must be positive and finite, `n_steps` a positive count, and
`policy_port` a port in `1-65535` - the same domain the policy providers
themselves apply, so a port the arm accepts is a port the provider can dial.
`policy_port` is required unless `run_policy` is given a pre-built
`policy_object`, which is the one case where the port is not read. A value none
of them can honor is reported by name, with the arm still disconnected and the
command bus still free for a task that could run.

`duration` is measured on a monotonic clock, not on the date. It is an elapsed
time rather than a point in time, so an NTP correction or a resume from suspend
cannot cut a rollout short or hold the servo bus past the budget - and the
`duration` the task reports back is the time that actually elapsed. The same
holds for `teleoperate(duration=...)`; see
[Teleoperation](teleoperation.md#mixin-api).

`cleanup()` (and `stop()`, which delegates to it) is terminal: it latches a
shutdown, releases the task executor, tears down the mesh and ROS bridges, and
disconnects the robot. It holds whatever state the robot is in - never
connected, or left disconnected by a failed bring-up - and `stop()` performs no
step of its own, so the two cannot diverge; being `async`, it runs the teardown
off the event loop, because joining the executor and closing a serial port both
block. The disconnect goes through the driver's own `disconnect()` while the
robot is connected - that is where torque disable and gripper release live - and
closes each device individually otherwise, so a half-open device set still ends
with the serial port released and every camera node closed. A serial port is
exclusive, so this is what makes the recovery for a wedged arm - tear down,
construct a new `Robot` - work without exiting the process. There is no
`restart`, so those same three entry points refuse permanently afterwards and
name the shutdown, rather than admitting a rollout that would command the arm
zero times. A rollout already in flight when the shutdown lands is reported
`STOPPED`, not `COMPLETED` - a shutdown truncates a task exactly as
`stop_task()` does, so its step count is a partial one. Construct a new `Robot`
to run another task.

## AgentTool actions

| Action | Blocking? | Needs |
|--------|-----------|-------|
| `execute` | Yes | `instruction` + `policy_port` |
| `start` | No | `instruction` + `policy_port` |
| `status` | - | - |
| `stop` | - | - |

## Teleoperation

High-level: attach one or more LeRobot teleoperators and drive this robot
directly. See **[Teleoperation](teleoperation.md)** for the full API and recipes.

```python
robot.attach_teleop("so101_leader", port="/dev/ttyACM1", id="leader")
robot.teleoperate()                       # local drive; stop_teleoperate() to end
robot.teleoperate(publish=True)           # drive + publish over the mesh
```

## Mesh teleop

Low-level transport primitives for streaming teleop actions between peers
(`teleoperate(publish=True)` builds on `start_teleop_publish`):

```python
robot.start_teleop_publish(teleoperator, device_name="leader", method="joint", hz=50)
robot.start_teleop_receive(source_peer_id="leader-abc123", device_name="follower", apply_fn=fn)
robot.get_teleop_status()
robot.stop_teleop()   # stop all sessions
```

## Sim vs real

| Feature | Simulation | HardwareRobot |
|---------|------------|---------------|
| Joint control | MuJoCo `data.ctrl` | LeRobot servo writes |
| Cameras | `add_camera()` post-construction | `cameras=` at construction |
| Reset | `reset()` rewinds to t=0 | Holds current pose |
| Randomization | `randomize(...)` | N/A |
| Policy execution | `run_policy()` / `start_policy()` | `start_task()` / `execute` action |
| Rollout horizon | `duration` **or** `n_steps` (`n_steps` supersedes it) | `duration` **and** `n_steps` (ANDed, so `duration` always bounds it) |

## See also

- [Hardware tools](tools.md) - calibrate / camera / teleop helpers.
- [Robot factory](../getting-started/robot-factory.md) - every `Robot()` kwarg.
- [Policy providers](../policies/overview.md) - available policy providers.
