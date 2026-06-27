---
description: Strands @tool helpers for hardware bring-up - calibrate, camera, teleop, train, pose, serial, gr00t inference, mesh, download assets.
---

# Hardware tools

```python
from strands_robots.tools import (
    lerobot_calibrate, lerobot_camera, lerobot_teleoperate, lerobot_train,
    pose_tool, serial_tool, download_assets,
    gr00t_inference,   # see GR00T page
    robot_mesh,        # see multi-robot page
)
# All return {"status": ..., "content": [{"text": "..."}]}
```

## Tools

| Tool | Key actions | What |
|------|-------------|------|
| `lerobot_calibrate` | `"list"`, `"info"`, `"search"`, `"compare"`, `"backup"` | Manage existing calibration JSONs under `~/.cache/huggingface/lerobot/calibration/` (this tool inspects/organizes - actual calibration is run via the LeRobot CLI) |
| `lerobot_camera` | `"list"`, `"test"`, `"stream"` | Enumerate, test, stream connected cameras |
| `lerobot_teleoperate` | `"start"`, `"stop"`, `"status"`, `"replay"`, `"dagger"` | Leader-follower teleop session, episode replay, and DAgger correction collection |
| `lerobot_train` | `"start"`, `"status"`, `"stop"`, `"list"` | Fine-tune a policy on a local dataset via `lerobot-train` |
| `pose_tool` | `"fk"`, `"ik"`, `"set_gripper"` | Forward/inverse kinematics, gripper control |
| `serial_tool` | `"list"`, `"send"` | Enumerate serial ports, send raw commands |
| `download_assets` | - | Pre-fetch MJCF assets to `~/.strands_robots/assets/` |
| `gr00t_inference` | `"start_container"`, … | GR00T container lifecycle - see [GR00T](../policies/groot.md) |
| `robot_mesh` | `"tell"`, `"broadcast"`, `"emergency_stop"` | Agent-driven mesh ops - see [Multi-robot](../mesh.md) |

Parse results via `result["content"][0]["text"]`, not custom keys like `result["ports"]`.

## Examples

```python
result = serial_tool(action="list")
print(result["content"][0]["text"])

result = lerobot_calibrate(action="list", device_type="robots")
result = lerobot_camera(action="list", camera_type="opencv")
result = pose_tool(action="fk", robot_id="so101_follower", port="/dev/ttyACM0")

# DAgger / teleop takeover: a policy drives the follower while the leader can
# pre-empt to record corrections (appended to the dataset as new episodes).
# Drives lerobot-rollout with --strategy.type=dagger.
result = lerobot_teleoperate(
    action="dagger",
    robot_type="so101_follower", robot_port="/dev/ttyACM0",
    teleop_type="so101_leader", teleop_port="/dev/ttyACM1",
    policy_path="user/act_fold",            # policy to roll out
    dataset_repo_id="user/fold_corrections",
    dataset_single_task="fold the towel",
    dagger_num_episodes=10,                  # cap collected corrections
)
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

- [Robot control](robot-control.md) - the `HardwareRobot` class.
- [Real hardware](../hardware/robot-control.md) - when each tool runs.
- [GR00T](../policies/groot.md) - `gr00t_inference` container lifecycle.
