---
description: Add record= to any run and you get a LeRobot dataset.
---

# Recording

```python
from strands_robots import Robot

arm = Robot("so101")
arm.sim.camera("front", size=(320, 240), attach=arm)
for _ in range(3):
    arm.run(my_policy, seconds=3, record="datasets/so101-wave")   # one episode each time
```

That's a LeRobot v3 dataset, written by LeRobot's own writer, so anything that trains on LeRobot data can use it.
Run again on the same folder and it adds episodes. A `meta/units.json` says what units the joints are in, so you
never have to guess later.

<img class="sr-media" src="../assets/media/recording.gif" alt="frames read back from a recorded dataset">

A run that crashes leaves nothing behind. A run you stop (or e-stop) keeps what it got. Use
`record="hf://you/so101-wave"` to record under a Hub name; nothing uploads unless you also pass `push=True`.

Recording is quiet: the video encoder reports errors only. Set `SVT_LOG=3` to see its full banner again.
