---
description: A Robot is a body. Look, move, run a policy. The loop is written once, for sim and real.
---

# A robot

{{drawing:d03_one_loop}}

```python
from strands_robots import Robot

arm = Robot("so101")
arm.observe()                                  # what it sees and feels right now
arm.act({"shoulder_pan": 0.3})                 # move, one control step
r = arm.run(lambda obs: {"elbow_flex": 0.4}, seconds=2)
print(r.steps, r.stopped_by)                   # 60 steps (30 Hz by default), "steps"
task = arm.start(my_policy, seconds=30)        # the same loop, in the background
arm.stop()
```

`run` keeps going until `seconds`, `steps` or `stop_when` says stop, then hands back a `RunResult`: how many steps,
how fast, and why it stopped (`"steps"`, `"stopped"`, `"fault"` if the policy gave up, `"estop"` if someone pressed
the button). Before the first step it checks the policy has what it needs, so a missing camera is a clear error
instead of a weird crash ten minutes later.

Unknown joints raise, targets outside the limits are clipped, and joints you don't mention stay where they are.
