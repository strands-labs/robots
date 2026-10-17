---
description: One world. Every robot you make joins it, and they all share the same physics and the same clock.
---

# One world

{{drawing:d02_one_world}}

```python
from strands_robots import Robot, Simulation

arm, g1 = Robot("so101"), Robot("g1", position=[2, 0, 0])
sim = arm.sim                      # = Simulation.default(); g1 is in it too
sim.add_object("box", shape="box", position=[0.3, 0, 0.1], size=[0.03, 0.03, 0.03])
sim.camera("wrist", position=[0.15, -0.25, 0.25], target=[0.2, 0, 0.05], attach=arm)
img = sim.render()                 # a picture of everything
```

You don't create a world to get started: the first `Robot` makes one, and the next one joins it. Want a separate
world? `Simulation()` and pass `sim=` to the robot.

Each `act()` moves the whole world forward by one control step (20 ms by default), so two robots can never drift out
of sync. Adding a robot, an object or a camera rebuilds the world and keeps everyone's joints where they were. Need to
step by hand? `with sim.manual_time():` and call `sim.step()` yourself.

MuJoCo is the default and runs anywhere. [Isaac Sim and Newton](gpu.md) plug into the same `Simulation`.
On a laptop, an arm plus a humanoid step in 85 µs.
