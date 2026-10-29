---
description: Your world, in a browser tab. Sliders, a live 3D twin, a chat, and an e-stop on every robot.
---

# Dashboard

{{drawing:v07_dashboard}}

<!-- test: skip, serves until Ctrl-C -->
```python
from strands_robots import Robot, Simulation
from strands_robots.dashboard import Dashboard

arm, g1 = Robot("so101"), Robot("g1", position=[1, 0, 0])
Dashboard(Simulation.default()).serve(port=8090)    # http://127.0.0.1:8090
```

The browser runs MuJoCo itself (in WebAssembly) and draws exactly what the server's world is doing; a move on the
server shows up about 27 ms later. Each robot gets a card with sliders and an **E-STOP**. The chat is an agent holding
`sim.tool` and every `robot.tool`. Pass `robots=[real_arm]` to add hardware, `mesh=lab` to see every robot on the mesh.

It only listens on your own machine, and it ignores requests from other websites, so a random page you have open
can't press its buttons. Real robots also need an `X-Confirm` from you before a slider moves them.

A robot is addressed by the name you wrote: `Robot("g1")` is `/api/robots/g1`. The browser twin names joints the
way the model file does (`so101/1` to `so101/6` for the arm); each card in `/api/scene` pairs `joints[i]` with its
action key `labels[i]`, so a custom front end or a browser test reads `window.__twin.qpos(joints[i])`.

<div class="sr-grid2" markdown>
<figure><img class="sr-media" src="../assets/media/dashboard.png" alt="so101 and G1 in the browser twin"><figcaption>an arm and a humanoid, one world</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/dashboard_estop.png" alt="estop latched"><figcaption>e-stop pressed: it waits for you</figcaption></figure>
</div>
