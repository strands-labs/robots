---
description: Run a LeRobot checkpoint in-process, or on a GPU machine with lerobot-policy-server over Zenoh. Joint names, cameras and units are checked before the robot moves.
---

# Running a LeRobot policy

strands-robots does not wrap LeRobot. LeRobot loads the model and runs its own client; strands-robots maps joint
names, cameras and units between the robot and the checkpoint, and refuses a mismatch by name before the robot moves.
Install LeRobot as described in [Installation](install.md#lerobot).

## In-process

<!-- test: skip, downloads a LeRobot checkpoint and needs LeRobot installed -->
```python
from strands_robots import Policy, Robot

arm = Robot("so101")
arm.sim.camera("front", size=(640, 480), attach=arm)
arm.run(Policy("lerobot/smolvla_base"), instruction="pick up the cube", seconds=10)
```

## On a policy server

Run the model on a GPU machine with LeRobot's server, and point the robot at it:

```bash
# on the GPU machine
lerobot-policy-server --config_path=smolvla.yaml
```

```python
# on the robot's machine
arm.run(Policy("zenoh://gpu-box:7447/smolvla"), instruction="pick up the cube", seconds=10)
```

`server_config(robot.description, "lerobot/smolvla_base")` writes the server configuration for a robot.

A server on another network uses the link it listens on: `Policy("zenoh://tls/host:7447/deployment")`, with the
locator written as given (`tls/host:7447`). Off the LAN the client waits up to 30 s for the first answer, sends JPEG
frames and connects to the router in client mode; set any of these explicitly to override them.

A deployment that declares the tasks it was trained on (`Policy("zenoh://...", tasks=[...])`, or the server's own
descriptor) refuses `run(instruction=)` with any other string, naming the trained tasks, before a frame is sent.
`policy.tasks` lists them.

A remote policy holds the last target through LeRobot's chunk playback plus a 1.0 s grace period, then raises a fault.
Under either execution mode (`"splice"` or `"sequential"`), `policy.calls` lists every chunk the session received:
`wall_s` (the round trip), `rows`, the `server` timing split and the `instruction`.

## Start pose

A policy that declares where its episodes begin (`start_pose`, in its own units, for example
`Policy("zenoh://...", start_pose={"shoulder_pan": 20.0})`, or the server's own value) is moved there first: `run()`
ramps to it over `start_seconds` (default 3) before the first step, so a chunked model plans from the pose it was
trained from.

## Units

Units are checked, not assumed. A checkpoint's units default to the robot's LeRobot convention (the SO-101 uses range
mode), and the checkpoint's saved min/max statistics must fit them. An SO-101 ACT checkpoint trained in degrees (with
a `wrist_roll` reaching -155) read as range mode is refused, naming the joint and the fix:
`Policy(repo, norm={"*": "degrees", "gripper": "range_0_100"})`.

## Supported policy families

Measured on an NVIDIA L40S, 13 LeRobot policy families drive the registry's robots: ACT, SmolVLA, pi0, pi0.5, GR00T,
XVLA, EVO1, LaWAM, VLA-JEPA, FineART, MolmoAct2, DM05, and SmolVLA in RTC mode. LeRobot itself refuses the other
families with its own message, before anything moves. The same checkpoint produces the same actions in-process and
served.
