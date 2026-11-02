---
description: Pass record= to any run to write a LeRobot v3 dataset, locally or under a Hugging Face Hub name.
---

# Recording a dataset

```python
from strands_robots import Robot

arm = Robot("so101")
arm.sim.camera("front", size=(320, 240), attach=arm)
for _ in range(3):
    arm.run(my_policy, seconds=3, record="datasets/so101-wave")   # one episode each time
```

The directory is a LeRobot v3 dataset written by LeRobot's own writer, so any training code that reads LeRobot data
can use it. Each run on the same directory appends an episode. `meta/units.json` records the units of the joint
columns.

<img class="sr-media" src="../assets/media/recording.gif" alt="frames read back from a recorded dataset">

An episode whose run raises is discarded. A run that is stopped or e-stopped keeps the frames recorded so far. Use
`record="hf://you/so101-wave"` to record under a Hub name; nothing uploads unless you also pass `push=True`.

The video encoder logs errors only. Set `SVT_LOG=3` to see its full banner again.
