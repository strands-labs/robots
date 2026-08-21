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
    use_ros,           # see ROS 2 integration page
    use_rtps,          # see Pure-RTPS ROS 2 page
)
# All return {"status": ..., "content": [{"text": "..."}]}
```

## Tools

| Tool | Key actions | What |
|------|-------------|------|
| `lerobot_calibrate` | `"list"`, `"view"`, `"search"`, `"backup"`, `"restore"` | Manage existing calibration JSONs under `~/.cache/huggingface/lerobot/calibration/` (this tool inspects/organizes - actual calibration is run via the LeRobot CLI) |
| `lerobot_camera` | `"list"`, `"test"`, `"capture"`, `"record"` | Enumerate, test, capture from, and record connected cameras |
| `lerobot_teleoperate` | `"start"`, `"stop"`, `"status"`, `"replay"`, `"dagger"` | Leader-follower teleop session, episode replay, and DAgger correction collection |
| `lerobot_train` | `"start"`, `"status"`, `"stop"`, `"list"` | Fine-tune a policy on a local dataset via `lerobot-train` |
| `pose_tool` | `"store_pose"`, `"load_pose"`, `"read_all"`, `"move_motor"` | Store, recall and replay named servo poses on a real bus, and read or move one motor at a time. This tool is joint-space only - Cartesian IK is `Simulation.move_to` |
| `serial_tool` | `"list_ports"`, `"send"` | Enumerate serial ports, send raw commands |
| `download_assets` | - | Pre-fetch MJCF assets to `~/.strands_robots/assets/` |
| `gr00t_inference` | `"start_container"`, … | GR00T container lifecycle - see [GR00T](../policies/groot.md) |
| `robot_mesh` | `"tell"`, `"broadcast"`, `"emergency_stop"` | Agent-driven mesh ops - see [Multi-robot](../mesh.md) |
| `use_ros` | `"list_topics"`, `"echo"`, `"publish"`, `"service_call"`, `"info"` | Bridge to any ROS 2 robot/sim - see [ROS 2 integration](../ros2-integration.md) |
| `use_rtps` | `"types"`, `"advertise"`, `"publish"`, `"subscribe"`, `"echo"` | Join a ROS 2 graph over pure RTPS (no rclpy) - see [Pure-RTPS ROS 2](../rtps-integration.md) |

Parse results via `result["content"][0]["text"]`, not custom keys like `result["ports"]`.

### Numeric options are checked before the session starts

A teleop session runs in a detached subprocess, so a value the lerobot CLI
cannot parse would not be reported by the call that supplied it - the session
would start, report a pid, and fail minutes later in its log. `lerobot_teleoperate`
therefore refuses an unusable numeric option up front, and only for the options
the requested action actually puts on the lerobot command line:

| Option | Accepted | Why the floor is where it is |
|--------|----------|------------------------------|
| `dataset_fps`, `fps` | positive whole number | lerobot declares both `int`; an integral float (`30.0`) is accepted and emitted as `30` |
| `dataset_num_episodes`, `dagger_num_episodes` | positive whole number | a recording of no episodes cannot be produced |
| `dataset_episode_time_s` | positive whole number | an episode of no length records nothing |
| `dataset_reset_time_s` | non-negative whole number | `0` is a real setting: no operator pause between episodes |
| `replay_episode` | non-negative whole number | `0` is the first episode |
| `teleop_time_s` | positive number, or `None` | lerobot declares it `float \| None`; `None` (the default) means no time limit, and a fractional budget is usable |

`teleop_time_s=0` is refused rather than read as "no limit" - it is the one value
that means "stop at once", so treating it as unset would invert the request.
Passing a value an action ignores is never an error: `action="start"` without a
`dataset_repo_id` teleoperates and reads no `dataset_*` option.

### A session is only forgotten once its process is gone

Because the session runs detached, the on-disk session store is the only place
its pid is recorded - `stop` and `status` both look the session up there. Every
load prunes finished sessions and writes the pruned store back, so what counts
as "finished" decides whether a session stays stoppable:

| What the probe reports | Verdict |
|------------------------|---------|
| the pid no longer exists | finished - pruned |
| `psutil.NoSuchProcess` (reaped between the existence check and the probe) | finished - pruned |
| `is_running()` returns `False` (a zombie, or the pid was reused) | not this session - pruned |
| `psutil.AccessDenied` (the pid exists, this user may not inspect it) | kept, and reported at `WARNING` |

The last row is why a session started under `sudo` - a common way to reach a
serial port - is still listed and still stoppable when the tool is later invoked
as the unprivileged user. Being kept is not a claim that it is running: `list`
and `status` each derive that from the pid's existence at the moment you ask.

### A raw servo write is bounded by the register it encodes into

`serial_tool` writes Feetech registers by masking the value into fixed-width
bytes of the outgoing packet, so an out-of-range value was never rejected on the
wire - it was truncated into a different, reachable command while the success
message quoted the value the caller supplied. `position=70000` put 4464 on the
wire and `position=-1` put 65535, the largest the two-byte field holds. Each
field is therefore bounded before the port is opened:

| Option | Accepted | Why the bound is where it is |
|--------|----------|------------------------------|
| `motor_id` | integer in `[1, 254]` | the frame carries the ID in one byte, and `255` is the header value |
| `position` | integer in `[0, 4095]` | `Goal_Position` is a 12-bit register - the same full scale the reported angle divides by |
| `velocity` | integer in `[0, 65535]` | `Goal_Velocity` is written as two bytes |
| `baudrate` | positive integer | pyserial coerces rather than checks, so `2.7` opens the port at 2 baud |
| `read_bytes` | positive integer | pyserial's read loop is `while len(read) < size`, so a non-positive size returns no bytes and looks like a timeout |
| `timeout` | finite number >= 0 | `0` is pyserial's non-blocking mode (return what is buffered); `nan` waits no time at all and `inf` overflows the deadline |

The same scoping rule applies: `action="read"` never looks at a servo register,
so a bad `motor_id` does not refuse it, and `action="list_ports"` reads none of
these options. An unset `motor_id` / `position` is still reported by the action's
own "required" message rather than as an unusable value.

`pose_tool` writes the same `Goal_Position` register through the same mask and
needs no bound of its own - it clamps to each motor's declared range before
encoding, so the mask only ever sees a value that fits.

### A mesh wait budget is bounded where the command body cannot carry it

`robot_mesh` takes four numeric options. `duration` and `policy_port` travel
inside the command body that
[`validate_command`](../security.md) inspects, so that validator already bounds
them. `timeout` and `limit` never enter a command body, so they are bounded by
the tool:

| Option | Accepted | Why the floor is where it is |
|--------|----------|------------------------------|
| `timeout` | positive finite number | it becomes a `threading.Event` wait; `0`/negative/`nan` return from that wait immediately, so the tool reports `{"status": "timeout"}` for a peer it never gave the chance to answer, and `inf` overflows the deadline |
| `limit` | positive integer | it is a slice index into the `inbox` buffer; a non-positive or `nan` value selected the *whole* buffer, and a fractional one raised out of the dispatcher |

`stop` additionally caps `timeout` at 5s so a stop cannot hang, but the cap
cannot replace the domain: `min(nan, 5.0)` is `nan`, so `nan` passed straight
through it.

The same scoping rule applies: `timeout` is read by `tell` / `send` / `rpc` /
`broadcast` / `stop`, `limit` only by `inbox`, and the rest are never refused
for either. `emergency_stop` fans out on a fixed internal budget, so the
caller's `timeout` is not effective there.

## Examples

```python
result = serial_tool(action="list_ports")
print(result["content"][0]["text"])

result = lerobot_calibrate(action="list", device_type="robots")
result = lerobot_camera(action="list", camera_type="opencv")
result = pose_tool(action="read_all", robot_id="so101_follower", port="/dev/ttyACM0")

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
