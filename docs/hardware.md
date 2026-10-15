---
description: Plug in a real arm. Same Robot, same verbs, plus a twin in the sim that mirrors it.
---

# Hardware

{{drawing:d04_twin}}

<!-- test: skip, needs a real SO-101 on a serial port -->
```python
from strands_robots import Robot

real = Robot("so101", port="/dev/ttyACM0", confirm=lambda action: input(f"move to {action}? ") == "y")
real.calibrate(interactive=False)   # uses the calibration file lerobot-calibrate wrote
real.observe()                      # real joints, in radians
real.act({"shoulder_pan": 0.3})     # writes the bus; the twin mirrors it and its world steps once
real.twin.sim.render()              # a sim copy that mirrors the real arm
```

| driver | robots | today |
|---|---|---|
| `feetech` | SO-100, SO-101 | read, write, torque, calibrate |
| `unitree` | G1, Go2, H1 | read only |
| `microduck` | microduck | read only |

A real robot never moves because of a default. Your own code can call `act()`. Anything else (an agent, the
dashboard, the mesh) goes through `confirm`, and with no `confirm=` it simply refuses. Every refusal (the gate,
a limit, the e-stop) is `from strands_robots import RequestRefused`, a `PermissionError`, and writes nothing to
the motors. Calibration is LeRobot's own format, so an arm you calibrated with LeRobot just works.

**The twin.** `real.twin` is the same description in `Simulation.default()`. `observe()` pushes the measured
joints into it (a read is not time); `act()` mirrors the targets and steps the twin's world one control step, so
the twin shows the arm without a `sim.step()` of your own (`manual_time()` turns that off).

{{drawing:d07_twin_steps}}

**The wire.** `FeetechBus` reads each status frame's head, `FF FF ID LEN`, then exactly `LEN` more
bytes, so `ping`, `read` and `sync_read` cost the reply (six servos answer in about a millisecond), not the
20 ms `timeout`, which only bounds a servo that does not answer. An adapter that echoes the request and
a stray byte ahead of the header are both skipped; an echo and then silence is a servo that did not answer.

{{drawing:d06_bus_frame}}
