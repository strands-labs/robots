---
description: The Isaac Sim backend for photorealistic worlds and the Newton backend for batched worlds on a CUDA GPU, behind the same Simulation API.
---

# GPU simulation: Isaac Sim and Newton

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
second** across 1,024 worlds. Isaac Sim allows one app per process, so a second `Simulation("isaac")` raises. On a machine without CUDA both
backends raise an error that says so.

<div class="sr-grid2" markdown>
<figure><img class="sr-media" src="../assets/media/isaac_so101_g1.png" alt="so101 and G1 in Isaac"><figcaption>Isaac Sim: one world, two robots</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/newton_1024.gif" alt="Newton, 1024 worlds"><figcaption>Newton: 1,024 worlds at once</figcaption></figure>
</div>
