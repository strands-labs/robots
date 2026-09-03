# Robots defined in code

A robot you start in your own script is not a second-class citizen of the
dashboard - it is the same thing the dashboard's own Spawn button produces. The
button is a form that builds a `Robot(...)` in a child process; writing that
call yourself skips the form.

```python
# arm1.py - run it anywhere on the mesh, including a Jetson or a Pi
import time

from strands_robots import Robot

robot = Robot(
    "so101",
    mode="real",
    port="/dev/cu.usbmodem5AB0181806",   # Linux: /dev/ttyACM0
    id="follower_arm",                    # the lerobot-calibrate id
    cameras={
        "top":   {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30},
        "wrist": {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    },
    mesh=True,
    peer_id="so101-arm-1",
)

# HardwareRobot connects lazily on its first task; connect once up front and the
# mesh starts publishing joints and camera frames immediately (this opens the
# servo bus - it does not move the arm).
robot.robot.connect(False)   # calibrate=False: use the saved calibration

while True:                   # stay alive so the peer stays on the mesh
    time.sleep(1)
```

```bash
STRANDS_MESH_LOCAL_DEV=1 python arm1.py
```

Open the dashboard and the card is there: same name, same two camera tiles,
same joint strips, same e-stop. Nothing had to be registered.

## The cameras argument is a mapping of mappings

This is the single most common way a first script fails:

```python
cameras={"top": 2}                       # WRONG
# ValueError: Camera 'top' config must be a mapping of option name to value,
#             got int: 2.
cameras={"top": {"index_or_path": 2}}    # right
```

Per-camera keys are the declared fields of lerobot's `OpenCVCameraConfig`
(`index_or_path`, `width`, `height`, `fps`, `color_mode`, `rotation`, ...) plus
`type`, which selects the backend (`opencv` is the one implemented). An unknown
key is **refused, with a did-you-mean** rather than dropped - a silently
ignored option would let a camera report success while streaming at the
default.

`index_or_path` takes an index (`2`) or a device path (`/dev/video0`). Only
integer-ish values participate in index bookkeeping; a path can never collide
with an index probe.

> Known rough edge: the dashboard's Spawn form sends its single camera as a
> bare integer (`{"main": 3}`), which is exactly the shape the validator
> refuses - a UI spawn *with* a camera can fail with that `ValueError` in the
> peer's log while a spawn without one succeeds. Attach cameras from the API or
> from a script until that is fixed, and read
> `GET /api/devices/logs/{peer_id}` when a spawn dies young.

## Which of these does the dashboard need?

| variable | why it matters to a code-defined peer |
|---|---|
| `STRANDS_MESH` | `true` opts a bare `Robot()` into the mesh without a code change; `false` is a hard kill switch that overrides `mesh=True` |
| `STRANDS_MESH_LOCAL_DEV` | single machine, no TLS - what `dashboard --local-dev` sets for itself. Use it on both sides or neither |
| `STRANDS_MESH_CAMERA_HZ` | frame rate this peer publishes at (the dashboard asks for its own default; lower it on a weak uplink) |
| `STRANDS_MESH_MULTICAST` | LAN auto-discovery. **Off by default** and it logs a warning when on: any device on the network can enumerate and attract the fleet. Prefer explicit `ZENOH_CONNECT` endpoints |
| `ZENOH_CONNECT` / `ZENOH_LISTEN` | how a peer on another machine finds the mesh (`tls/robot.lan:7447`) |
| `STRANDS_MESH_OVERRIDE_CODE` | e-stop resume code. Without it, one broadcast e-stop locks this robot until you restart the process - set the **same** value on every peer |

The full table lives in the
[README environment variables section](https://github.com/strands-labs/robots#environment-variables);
mesh security modes are in [Multi-robot Mesh](../mesh.md).

## Deploying the same rig to an edge device

The script above is the deployment artifact. Copy it to the machine the arm is
plugged into, point both sides at the same mesh, and the card reappears -
running on the edge box, visible from your laptop:

```bash
# on the edge device
ZENOH_CONNECT=tls/dashboard-host.lan:7447 STRANDS_MESH=true python arm1.py
# or point the dashboard at the mesh instead
python -m strands_robots dashboard --zenoh-connect tls/robot.lan:7447
```

Camera indices are per-machine facts: an index that meant "wrist" on your
laptop means nothing on the Jetson. Re-check with a frame after the move
([how](quickstart.md#6-identify-the-cameras-1-min-and-the-one-that-saves-an-evening)).

## What a dashboard spawn still owns

Parity is about the robot, not about lifecycle. For peers the dashboard started
itself it additionally offers:

| | dashboard-spawned | code-defined |
|---|---|---|
| appears in the fleet, cameras, joints, tasks, e-stop | yes | yes |
| `POST /api/devices/despawn` can stop it | yes | no - it is your process |
| `GET /api/devices/logs/{peer_id}` | yes (its stdout) | no - the log is your terminal |
| remembered in `~/.strands_dashboard/profiles.json` and auto-respawned on re-plug | yes | no |

So a code-defined peer is a peer the dashboard *observes and commands* but does
not *own* - which is the right split when the robot lives on hardware you do
not want a browser tab to be able to kill.
