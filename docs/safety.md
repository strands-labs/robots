---
description: The manifest each robot generates, the safety chain every outside command passes, and the e-stop.
---

# Safety and e-stop

Every robot has a manifest, a safety chain and an e-stop.

{{drawing:d06_device}}

```python
from strands_robots import Robot
from strands_robots.robot.safety import RequestRefused

arm = Robot("so101")
try:
    arm.confirm_act({"elbow_flex": 3.1})
except RequestRefused as e:
    print(e)  # elbow_flex=3.1000 rad is outside its range [-1.7453, 1.5708]
```

## Manifest

`arm.manifest` lists each control with its range, its safe range and what it does if the link drops. The library
builds it from the robot. Agent tools, the mesh and the dashboard all read it.

## Safety chain

Commands from agents, the dashboard or the mesh pass one chain: range, safe range, interlocks, validators, then
the operator. Your own `act()` calls skip the operator step. A refusal says why.

## E-stop

`estop()` works from any thread and nothing can block it. The robot stays stopped until a named operator resumes it.

```python
task = arm.start(my_policy, seconds=30)
arm.estop(by="alice")             # the run stops within about 10 ms
arm.resume(approved_by="alice")
```

<img class="sr-media" src="../assets/media/so101_estop.gif" alt="e-stop mid-run, the arm holds">

The arm holds its pose. It does not drop.

| API | Does |
|---|---|
| `hold()` | Stand still |
| `subscribe("state", fn, hz=)` | Stream the robot's state |
| `Watchdog` | Applies `on_disconnect` when commands stop |

For real robots, see [the operator gate](hardware.md#the-operator-gate).
