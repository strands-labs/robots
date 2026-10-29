---
description: A Robot is a body in a world. observe, act, run; the loop is written once and returns what it measured.
---

# A robot

{{drawing:d03_one_loop}}

```python
from strands_robots import Robot, Policy

arm = Robot("so101")                        # a body in Simulation.default(); arm.sim is that world
g1 = Robot("g1", position=[1, 0, 0])        # the same world: arm.sim is g1.sim
arm.observe()                               # Observation: joints, base pose, images of cameras attached to it
arm.act({"shoulder_pan": 0.3})              # write targets for THIS body, then one control step for the world
arm.act({"gripper": 1.0}, seconds=1.0)      # reach the pose: ramp there from the measured one, then read back
res = arm.run(lambda obs: {"elbow_flex": 0.4}, hz=50, steps=60)   # RunResult(steps, seconds, ..., final)
res.stopped_by                              # "steps" here; "seconds" for run(seconds=), else stop_when/estop/fault...
task = arm.start(my_policy, seconds=30)     # the same loop in the background; arm.stop() returns its RunResult
arm.sim.run({arm: my_policy, g1: my_policy}, steps=20)   # many robots in lock-step, each policy bound as run() binds it
```

`Robot(name)` loads the `Description` and joins `Simulation.default()`, or the world passed as `sim=`;
`position`, `orientation`, `keyframe` and `id` place it. Its `id` is unique in the world (a second
`so101` is `so101_2`). The `Body` is the only thing that differs between bodies: `SimBody` is a handle
into a world, and the loop never knows which one it has.

An `Action` is a vector in `action_keys` order or a `{joint: target}` mapping: a joint it leaves out keeps
its last commanded target, so `act({"elbow_flex": 0.4})` holds the other five against gravity (`act({})` holds
all six). Before the first write, and after `hold()`, an e-stop, `calibrate()` or `sim.reset()`, that target is
the measured pose. An unknown joint raises `KeyError`, a vector of the wrong length raises `ValueError`, and every
target is clipped to the joint limits. `set_pose` teleports, `set_gains(kp, kd)` sets the PD gains of
its actuators, `describe()` is JSON-safe, `close()` removes it from its world (a removed robot raises
`RuntimeError` on its next `observe`).

`act(action, seconds=)` reaches a pose instead of taking one step toward it: the targets ramp from the
measured pose to `action` at 30 Hz, settle there, and the returned `Observation` is the pose reached. Through
an agent tool (`confirm_act`) the operator gate is asked once, for the target. To stand
still, call `hold()`; there is no `"hold"` policy. On a Feetech arm every write is also capped at `max_speed`
(default 2 rad/s, `Robot(..., max_speed=None)` to opt out): a goal 50° away becomes a ramp, a trained chunk's
steps pass as written, and `extras()["slewed"]` counts the writes the cap shortened. The first write after
construction or after torque goes off reads the arm first, so the cap never depends on the caller having read.

A policy that names where its episodes begin (`start_pose`, in its own units; `Policy("zenoh://…",
start_pose={"shoulder_pan": 20.0})`, or the server's own word) is taken there first: `run()` ramps to it over
`start_seconds` (default 3) before step one, so a chunked model plans its first chunk from the pose it was trained
from instead of from wherever the arm was left.

`run(policy, hz=, seconds=|steps=|stop_when=)` takes a `Policy` subclass, anything with `act(obs)` and
`reset()`, or a bare callable (it receives `instruction=` when it declares it). In a world, `hz` sets how
many physics steps each `act()` advances and time is sim time; `realtime=True` also paces the loop to the
wall clock. Before the first inference `policy.requirements()` is checked against a real observation, so a
missing camera or a wrong state length is refused by name. The policy's optional hooks (`hz`, `realtime`,
`gains`, `attach`, `detach`, `requirements`, `needs_observation`) are read once there too, and one of the
wrong type is a `TypeError` naming it. Once `attach` has run, any refusal calls `detach` before it reaches you. `stop(timeout=)` raises `TimeoutError` while the run is still
going, and the error of a run whose policy raised, once. `units="auto"` converts joints between the
robot's radians and the units the policy declares (`policy.units`, for example `lerobot_so10x`).

{{drawing:d08_policy_hooks}}

{{drawing:d13_bind_releases}}

`run(record="data/wave")` appends the run as one episode of a LeRobot v3 dataset; it needs LeRobot with its
`dataset` extra, which the one LeRobot install line carries. `Policy("data/wave?episode=0")` replays that
episode: the recorded `action` column, at the dataset's fps, in the units `meta/units.json` names (a dataset
without that file needs `units=`; one that has it refuses a `units=` that disagrees). `episode` is the query's
only key: `?ep=1` or `?episode` is refused by name, never episode 0 by default. A path that exists is never sent
to the Hub: a directory with neither `config.json` nor `meta/info.json` is a `ValueError`.

{{drawing:d09_record_replay}}
