---
description: The same Robot with a Driver for a body. Feetech STS3215, Unitree DDS, Microduck robotd. A real body never moves because of a default.
---

# Hardware

{{drawing:d04_twin}}

<!-- test: skip, needs a real SO-101 arm on USB -->
```python title="needs the arm on USB"
from strands_robots import Robot

real = Robot("so101", port="/dev/ttyACM0", confirm=lambda action: input(f"move to {action}? ") == "y")
real.calibrate(interactive=False)            # the LeRobot calibration JSON lerobot-calibrate wrote, unchanged
obs = real.observe()                         # measured joints in radians, also pushed into real.twin
real.confirm_act(obs.joints)                 # a hold, through the operator gate; the twin mirrors it
real.confirm_act({"gripper": 1.0}, seconds=1.5)   # reach a pose: one yes, a 30 Hz ramp, the joints read back
real.twin.sim.render()                       # what the arm is doing, drawn in MuJoCo
real.close()                                 # torque off, port closed, twin removed
```

| driver | hardware | in 1.0 |
|---|---|---|
| `feetech` | STS3215 buses: SO-100 / SO-101 (`pip install pyserial`) | read, write, torque, `calibrate()` |
| `unitree` | DDS `rt/lowstate`: G1 / Go2 / H1 (`unitree_sdk2py`) | observe only |
| `microduck` | robotd's JSON-RPC socket (`port=` the socket path) | observe only: robotd has no per-joint write |
| `memory` | no wire: any robot, no arm attached (docs, notebooks, CI) | a write lands at once, the next read returns it |

A `Driver` is `names read() write() pose() images() extras() hold() estop() calibrate() close()` plus
`supports_position_hold`, `calibration` and `resolution` (None when the bus has none); `driver=` takes a name
or an instance, and an instance whose `names` are not the description's `action_keys`, in order, is refused. `Robot(name, port=)` picks the driver from the registry entry and adds a twin of the same
description to `Simulation.default()`: `real.twin` is `so101_twin`, its `twin_of` is `real`, and only
`real.is_real` is true (`twin=False` to skip). `observe()` places the twin at the measured joints
(`Backend.set_joints`, no step), so `real.twin.sim.render()` draws the measured joints (a floating base is not placed); `act()` mirrors the targets and steps the twin's world one control step (`manual_time()` stops it). The twin touches the floor as its plain body would, and nothing else, so it never pushes a simulated robot or object standing where it spawns; a NaN reading, or one beyond MuJoCo's 1e10 (inf included), is not mirrored, and if its world closes the real robot carries on without it. `set_pose` raises on hardware; `real.twin.set_pose` works.

**No arm on the desk.** `driver="memory"` is a hardware body with no wire, so the operator gate, `approve=`,
the e-stop and the watchdog's real-body rule run exactly as they do on an arm. Its `MemoryDriver` logs every
target in `writes` and counts `holds` and `estops`; it takes no `port=` and no `cameras=`.

```python
from strands_robots import Robot
from strands_robots.robot.safety import RequestRefused

real = Robot("so101", driver="memory", twin=False, confirm=lambda targets: False)
try:
    real.confirm_act({"shoulder_pan": 0.3})  # what an agent tool or the dashboard sends
except RequestRefused:
    pass
assert real.body.writes == []                # the operator said no: nothing reached the bus
real.act({"shoulder_pan": 0.3})              # your own code is the operator
assert len(real.body.writes) == 1
```

**Driver options.** `Robot(name, port=, **driver_options)` hands any extra keyword to the driver
unchanged: the `FeetechDriver` takes `robot_id=` (which calibration file to load) and `baud=`, the unitree
and microduck links take `timeout=`. `make_driver` forwards them and checks each against
`driver_parameters` for that driver, so a misspelt option (`baudrate=` for `baud=`) raises a `TypeError`
that names what the driver takes instead of being silently dropped, and a new parameter on a driver with
no public path fails the pass-through test. A robot in a world has no bus, so it takes no driver options:
`Robot("so101", baud=...)` with no `port=` is refused the same way. Hardware camera specs are
`cameras={"wrist": {"index": 1, "width": 640, "height": 480, "fps": 30}}` (LeRobot's `OpenCVCameraConfig`
field names); a bare index takes the device default, and a camera that delivers another size than asked is
refused at the first frame, because every checkpoint is trained at one size. Better than an index — which shifts
whenever a phone or a screen capture joins the machine — is the device's own name: `cameras={"front": "Logi 4K",
"wrist": "USB2.0_CAM1"}` resolves against what the OS lists (AVFoundation on macOS, V4L2 on Linux; a name that
matches nothing or two devices is refused listing them), and a camera with no declared size takes the policy's
own at `run()`. The dict's keys are the policy's image keys; there is nothing to rename.

{{drawing:d17_driver_options}}

{{drawing:d07_twin_steps}}

{{drawing:d09_twin_measured}}

**Calibration.** `calibrate(interactive=False)` returns the LeRobot file it found under
`$HF_LEROBOT_CALIBRATION/robots/`: `so_follower/<id>.json` (LeRobot 0.6 names every SO-10x arm so),
then `<lerobot_type>/<id>.json`, `<id>` being `Robot(id=)`, which never loads another arm's file (a warning names the files there); without one, `<lerobot_type>.json` or the only file in the directory.
`Robot("so101", port=..., calibration="my_arm.json")` takes any path; a missing file is an error that
lists where it looked. `calibrate()` walks a human at a terminal through LeRobot's procedure (middle
pose, then a sweep of every joint but `wrist_roll`, which gets the full turn, 0..4095, as LeRobot gives it) and writes over the file it loaded, else `<lerobot_type>/<id>.json`; with `save=True` (the default) it refuses to record when that file was only a guess (no id, the one file in its directory), which may be another arm's: pass `id=` or `calibration=`. A sweep that leaves a swept joint without a range is refused, as LeRobot refuses it; nothing is saved, and the servos get back the homing offsets they held, so the calibration loaded still maps the arm. With no terminal it refuses before torque goes off. With a calibration loaded it asks first, as LeRobot does; Enter keeps it and, unlike LeRobot, writes nothing to the servos; `c` re-records. Without a
file, joints map the full turn of counts (2048 is 0 rad). `observe()` maps each joint's calibrated
`range_min`..`range_max` onto the description's joint limits, so radians match the twin; LeRobot's own
degrees are 360/4095 per count from the middle of that range, so the same pose can read a few degrees apart. A LeRobot policy
gets LeRobot's degrees: `run()` converts through the bus calibration.

**On your arm.** `STRANDS_SO101_PORT=/dev/ttyACM0 pytest tests_integ/acceptance` runs acceptance check 9
on a physical SO-101: `observe()`, a refused `confirm_act()` that writes nothing, an allowed hold the twin
mirrors, and `calibrate(interactive=False)` against LeRobot's file, and `Driver.read()` under 3 ms. Without an arm it skips and says why.

**The wire.** `FeetechBus` reads each status frame's head, `FF FF ID LEN`, then exactly `LEN` more
bytes, so `ping`, `read` and `sync_read` cost the reply (six servos answer in about a millisecond), not the
20 ms `timeout`, which only bounds a servo that does not answer. An adapter that echoes the request and
a stray byte ahead of the header are both skipped; an echo and then silence is a servo that did not answer.

{{drawing:d06_bus_frame}}

{{drawing:d05_bench}}

{{drawing:d10_what_never_reaches_the_arm}}

{{drawing:d11_fault_goes_safe}}

**The operator gate.** `act()` in your own code is the developer's call. `confirm_act()` is what an
agent tool or a dashboard calls: on a real body it asks `confirm(action)` and refuses on `False`; with
no `confirm=` it raises and says `pass confirm=`. Every refusal (the gate, a limit, the e-stop's `EStopped`) is `from strands_robots import RequestRefused`, a `PermissionError`, and writes nothing to the bus. A NaN or infinite target raises `NonFiniteTarget` (a `ValueError`) before
any write, and ends a `run()` with `stopped_by="fault"`. When a run ends in a fault the body is put in
its declared `on_disconnect` state (each control's `hold`, `zero`, `safe` or `apply`), so a policy that
can no longer act safely leaves the arm somewhere declared safe rather than frozen at its last command.

**One yes per run.** An agent's `run()` on a real body is not a single `act`: instead of asking `confirm=`
on every tick (30 dialogs a second), it asks `Robot(approve=callable(RunPlan) -> bool)` once, with the whole
plan (robot, policy, instruction, rate, duration, units), before anything is acquired; without `approve=` the run is
refused and nothing is written. The safety chain (the e-stop, the joint range, the soft `safe` band, the
interlocks and the validators) still runs on every step, and a target outside `safe` inside an approved run is
refused, never re-asked each tick. `confirm=` stays the per-command gate (`confirm_act`, the dashboard).

{{drawing:d16_one_yes_per_run}}
A remote policy (`Policy("zenoh://host:7447/deployment")`) holds the last target through LeRobot's chunk
playback and a 1.0 s grace, then raises that fault. Under either execution (`"splice"` or `"sequential"`)
`policy.calls` lists every chunk the session received: `wall_s` (the round trip), `rows`, the `server` split and the
`instruction`. Only a real body
takes `confirm=`:

```python
from strands_robots import Robot

try:
    Robot("so101", confirm=lambda action: False)
except TypeError as e:
    print(e)                                 # confirm= are hardware Robot arguments; a simulated body takes none
```

{{drawing:d12_one_tick}}

**One clock, one read, one write per tick.** A hardware robot reads time from `robot.clock` (the
monotonic clock, or `Robot(..., clock=)`); a robot in a world reads the world's time. `Observation.t`,
the latch, the watchdog and `RunResult.seconds` all use it. Each tick of `run()` reads the bus once and
writes once (`robot.command(targets)` writes without reading). Writes are spaced one period apart, less
how late the host woke from the wait before the previous one, so a run keeps its rate: a write not
held back to its period that lands past the next tick's start, after a slow inference or a late wake,
is a deadline miss, counted in `RunResult.deadline_misses` and `worst_late_s`, and the next write
waits a full period instead of catching up. `run(on_miss="fault")` ends the run at the late tick with
`stopped_by="deadline"`.

{{drawing:d11_estop_audit}}

**The e-stop and resume.** `robot.estop(by="alice@corp")` latches now, bypassing the chain. The
`Latch` records `stopped_by`, the `reason`, `engaged_at` on the robot's clock and a wall-clock `engaged_wall`,
so a stop can be attributed and placed on a calendar in an audit record. Resuming takes the operator
who cleared it: `robot.resume(approved_by="alice@corp")`. Pass the engagement `count` you showed the
operator as `resume(approved_by=..., stop=n)` and a yes typed for the stop they saw is refused once a
later, different stop is latched.
