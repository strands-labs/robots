---
description: HardwareRobot — async task execution, status reporting, the LeRobot bridge.
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
| `cameras` | `{name: config_dict}`. Config keys: `type`, `index_or_path`, `fps`, `width`, `height`, `serial`. |
| `action_horizon` | Actions per inference step (default 8). |
| `data_config` | GR00T data_config name. |
| `control_frequency` | Control loop Hz (default 50). |
| `**kwargs` | Forwarded to LeRobot backend (`port`, `robot_ip`, `kp`, `kd`, …). Unknown kwargs raise `ValueError`. |

## Task lifecycle

`TaskStatus`: `IDLE` → `CONNECTING` → `RUNNING` → `COMPLETED` / `STOPPED` / `ERROR`

| Method | What |
|--------|------|
| `start_task(instruction, policy_port, policy_host, policy_provider, duration)` | Async; returns immediately. |
| `stop_task()` | Halt running policy. |
| `get_task_status()` | Returns `RobotTaskState` (status, step count, error). |
| `cleanup()` | Stop tasks, close cameras, stop mesh. |

## AgentTool actions

| Action | Blocking? | Needs |
|--------|-----------|-------|
| `execute` | Yes | `instruction` + `policy_port` |
| `start` | No | `instruction` + `policy_port` |
| `status` | — | — |
| `stop` | — | — |

## Mesh teleop

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

## See also

- [Hardware tools](tools.md) — calibrate / camera / teleop helpers.
- [Robot factory](../getting-started/robot-factory.md) — every `Robot()` kwarg.
- [Policy providers](../policies/overview.md) — available policy providers.
