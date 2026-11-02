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

## Recording from the dashboard

The dashboard's **Record** tab writes the same dataset without code. Start a dashboard (see
[Dashboard](dashboard.md)), open the Record tab and:

1. Tick the robots to record. Each robot lists its own cameras with a live thumbnail, plus any camera in the world
   that is not attached to a robot; tick the ones to include.
2. Enter the task text (stored as each frame's task), a target and an fps (1 to 60). The target is a directory, or
   `hf://owner/name` to use a Hub name. The page never uploads.
3. Press **rec**. Anything that moves the robot is recorded: the joint sliders, a policy run, the chat agent, a
   script. The recorder only observes; each frame's action is the last joint target written to the robot, or the
   measured joints when nothing has been written.
4. Press **save episode** to keep the current take and start the next one, **discard** to drop it, or **stop** to
   keep the take in progress and end the recording.

While a take is live the tab shows the episode, frame count, achieved fps and any frames dropped against the target
fps, and a REC indicator appears in the top bar. Recording several robots writes one dataset per robot, at
`<target>/<robot id>`. Robots that LeRobot knows (for example so101) are recorded in LeRobot's units; others in
radians.

The tab is a front end for these routes, which scripts can call directly:

| Route | Body | Does |
|---|---|---|
| `GET /api/record/options` | | robots, their cameras, world cameras, a default target |
| `POST /api/record/start` | `{robots, cameras: {robot id: [names]}, target, task, fps}` | starts the first episode |
| `POST /api/record/episode/save` | `{}` | keeps the current episode, starts the next |
| `POST /api/record/episode/discard` | `{}` | drops the current episode, starts the next |
| `POST /api/record/stop` | `{"save": true}` | ends the recording, keeping (or dropping) the take in progress |
| `GET /api/record/status` | | state, frames, fps, dropped frames, saved episodes |

Read a recording back with LeRobot:

<!-- test: skip, reads a dataset recorded from the page -->
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/so101-from-the-page", root="/tmp/strands-recording-20261102-041500")
print(ds.meta.total_episodes, ds.fps, ds.meta.video_keys)
```
