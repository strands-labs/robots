---
description: Same code, bigger engines. Isaac Sim for photoreal worlds, Newton for thousands of worlds at once.
---

# GPU backends

{{drawing:v04_backend}}

<!-- test: skip, needs CUDA (Isaac Sim, Newton) -->
```python
from strands_robots import Robot, Simulation

sim = Simulation("isaac")                              # Isaac Sim (needs CUDA)
arm, g1 = Robot("so101", sim=sim), Robot("g1", sim=sim, position=[1, 0, 0])
arm.run(my_policy, seconds=2)                          # the same loop as on your laptop

batch = Simulation(physics="newton", num_envs=1024)    # 1,024 worlds on one GPU
```

On an L40S: Isaac holds both robots in one world at about 230 steps a second. Newton runs **1.75 million steps a
second** across 1,024 worlds. Isaac allows one app per process, so a second `Simulation("isaac")` politely says no.
On a machine without CUDA these backends say so and step aside.

A scene file works on Isaac too: `sim.add_object("room", path="room.xml", free=False)` converts the MJCF or URDF
with Isaac's own importer (cached by content, like a robot) and places it as one body. `free=True` first wraps the
file in a free body, as the MuJoCo backend attaches it, and the rigid body the importer makes of it is the one that
moves. Isaac's importer keeps geometry, colliders and materials; a file's lights and skybox do not come through.
Still refused by name on Isaac: `camera(attach=)`, friction and mass randomization, `set_gains`, and a bare mesh
file as `path=`, which goes inside an MJCF.

<div class="sr-grid2" markdown>
<figure><img class="sr-media" src="../assets/media/isaac_so101_g1.png" alt="so101 and G1 in Isaac"><figcaption>Isaac Sim: one world, two robots</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/newton_1024.gif" alt="Newton, 1024 worlds"><figcaption>Newton: 1,024 worlds at once</figcaption></figure>
</div>
