---
description: The manifest each robot generates, the safety chain every outside command passes through, and the e-stop.
---

# Safety and e-stop

{{drawing:d06_device}}

Every `Robot` has a generated manifest, a safety chain for commands that do not come from your own code, and an
e-stop.

```python
from strands_robots import Robot
from strands_robots.robot.safety import RequestRefused

arm = Robot("so101")
arm.manifest.controls[0]     # shoulder_pan.pos: its range, its safe range, what to do if the link drops
```

## Manifest

The manifest is generated from the robot, never written by hand. Agent tools, the mesh and the dashboard all read
it, so they always agree on the robot's controls and ranges.

## Safety chain

A command that does not come from your own code (an agent, the dashboard, a mesh peer) goes through one safety
chain: range, safe range, interlocks, validators, then the operator. A refusal names the reason:

```python
try:
    arm.confirm_act({"elbow_flex": 3.1})
except RequestRefused as e:
    print(e)   # so101: elbow_flex=3.1000 rad is outside its range [-1.7453, 1.5708]
```

## E-stop

`estop()` works from any thread, bypasses the command queue and takes no lock, so nothing can block it. It stays
latched until a named operator resumes the robot:

```python
task = arm.start(my_policy, seconds=30)
arm.estop(by="alice")                # the run stops ~10 ms later; a later move is refused "e-stop is engaged by alice"
arm.resume(approved_by="alice")    # a named operator clears the latch
```

<img class="sr-media" src="../assets/media/so101_estop.gif" alt="estop mid-run, the arm holds">

The arm holds its pose instead of dropping. Related APIs: `hold()`, `subscribe("state", fn, hz=)` for a live state
feed, and a `Watchdog` that applies each joint's `on_disconnect` target when commands stop arriving. See
[Connecting hardware](hardware.md#the-operator-gate) for the operator gate and run approval on real robots.
