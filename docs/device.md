---
description: Every robot knows what it is, checks every command, and stops the moment you say so.
---

# Device and safety

{{drawing:d06_device}}

Robots that agents can drive need three things: a description of themselves, a bouncer at the door, and a big
red button. Every `Robot` has all three.

```python
from strands_robots import Robot
from strands_robots.robot.safety import RequestRefused

arm = Robot("so101")
arm.manifest.controls[0]     # shoulder_pan.pos: its range, its safe range, what to do if the link drops
```

**The manifest** is generated from the robot, never written by hand. Agent tools, the mesh and the dashboard all
read it, so they can't disagree about what the robot is.

**The bouncer.** Anything that isn't your own code (an agent, the dashboard, a peer on the mesh) goes through one
safety chain: range → safe range → interlocks → validators → the operator. A refusal says exactly what's wrong:

```python
try:
    arm.confirm_act({"elbow_flex": 3.1})
except RequestRefused as e:
    print(e)   # so101: elbow_flex=3.1000 rad is outside its range [-1.7453, 1.5708]
```

**The red button.** `estop()` works from any thread, skips the queue and takes no lock, so nothing can block it.
It stays pressed until a named person lets go:

```python
task = arm.start(my_policy, seconds=30)
arm.estop()                          # the run stops ~10 ms later, stopped_by="estop"
arm.resume(approved_by="alice")    # nobody else's say-so counts
```

<img class="sr-media" src="../assets/media/so101_estop.gif" alt="estop mid-run, the arm holds">

The arm holds its pose instead of dropping. Other handy bits: `hold()`, `subscribe("state", fn, hz=)` for a live
feed, and a `Watchdog` that applies each joint's `on_disconnect` if commands stop arriving.
