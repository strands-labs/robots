---
description: Eight Strands @tool helpers for hardware bring-up — calibrate, camera, teleop, pose, serial, gr00t inference, mesh, download assets.
---

# Hardware tools

```python
from strands_robots.tools import (
    lerobot_calibrate, lerobot_camera, lerobot_teleoperate,
    pose_tool, serial_tool, download_assets,
    gr00t_inference,   # see GR00T page
    robot_mesh,        # see multi-robot page
)
# All return {"status": ..., "content": [{"text": "..."}]}
```

## Tools

| Tool | Key actions | What |
|------|-------------|------|
| `lerobot_calibrate` | `"list"`, `"info"`, `"search"`, `"compare"`, `"backup"` | Manage existing calibration JSONs under `~/.cache/huggingface/lerobot/calibration/` (this tool inspects/organizes — actual calibration is run via the LeRobot CLI) |
| `lerobot_camera` | `"list"`, `"test"`, `"stream"` | Enumerate, test, stream connected cameras |
| `lerobot_teleoperate` | `"start"`, `"stop"`, `"status"` | Leader-follower teleop session |
| `pose_tool` | `"fk"`, `"ik"`, `"set_gripper"` | Forward/inverse kinematics, gripper control |
| `serial_tool` | `"list"`, `"send"` | Enumerate serial ports, send raw commands |
| `download_assets` | — | Pre-fetch MJCF assets to `~/.strands_robots/assets/` |
| `gr00t_inference` | `"start_container"`, … | GR00T container lifecycle — see [GR00T](../policies/groot.md) |
| `robot_mesh` | `"tell"`, `"broadcast"`, `"emergency_stop"` | Agent-driven mesh ops — see [Multi-robot](../mesh.md) |

Parse results via `result["content"][0]["text"]`, not custom keys like `result["ports"]`.

## Examples

```python
result = serial_tool(action="list")
print(result["content"][0]["text"])

result = lerobot_calibrate(action="list", device_type="robots")
result = lerobot_camera(action="list", camera_type="opencv")
result = pose_tool(action="fk", robot_id="so101_follower", port="/dev/ttyACM0")
```

## Use with an agent

```python
from strands import Agent
from strands_robots import Robot
from strands_robots.tools import lerobot_calibrate, lerobot_camera, pose_tool, serial_tool

agent = Agent(tools=[
    Robot("so100"),
    lerobot_calibrate, lerobot_camera, pose_tool, serial_tool,
])
agent("Find a connected so100, calibrate it, then stream the wrist camera for 10 seconds")
```

## See also

- [Robot control](robot-control.md) — the `HardwareRobot` class.
- [Real hardware](../hardware/robot-control.md) — when each tool runs.
- [GR00T](../policies/groot.md) — `gr00t_inference` container lifecycle.
