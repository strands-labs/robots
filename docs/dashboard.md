---
description: "Serve a world to the browser: a live 3D twin, joint controls, an agent chat and an e-stop for every robot."
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
server shows up about 27 ms later. Each robot gets a card with joint sliders and an e-stop. The chat is an agent holding
`sim.tool` and every `robot.tool`. Pass `robots=[real_arm]` to add hardware, `mesh=lab` to see every robot on the mesh.

## Start a dashboard locally

With the `dashboard` and `mujoco` extras installed (both are part of `[all]`), one command serves a robot in a
MuJoCo world:

```bash
python -m strands_robots.dashboard                       # the microduck on http://127.0.0.1:8090
python -m strands_robots.dashboard so101 g1 --port 8091  # an SO-101 arm and a G1 humanoid side by side
```

Open `http://127.0.0.1:8090`. Chat and voice are off unless you turn them on; the page says so in the chat panel.

```bash
python -m strands_robots.dashboard so101 g1 --model <bedrock model id> --voice auto
```

| Flag | Turns on | Needs |
|---|---|---|
| `--model <id>` | the chat agent (`model=` to `Dashboard`) | the `tools` extra, AWS credentials |
| `--voice auto` · `openai` · `nova` | speech to speech, the talk button (`voice=` to `Dashboard`) | the `voice` extra; `OPENAI_API_KEY` for `openai`, AWS credentials for `nova` |

## Live view on the docs home page

The home page of a locally served copy of these docs (`mkdocs serve`) checks for a dashboard on
`http://127.0.0.1:8090/`. If one answers, the recording in the page header is replaced by the live dashboard in an
iframe. Only loopback addresses are embedded. To embed a dashboard on another port, add `?live=http://127.0.0.1:<port>/`
to the home page URL.

## Security

The server listens on the loopback interface only and rejects cross-origin requests, so another web page open in
the browser cannot call its API. A command to a real robot also requires an `X-Confirm` header from the operator.

## API

A robot is addressed by the name you wrote: `Robot("g1")` is `/api/robots/g1`. The browser twin names joints the
way the model file does (`so101/1` to `so101/6` for the arm); each card in `/api/scene` pairs `joints[i]` with its
action key `labels[i]`, so a custom front end or a browser test reads `window.__twin.qpos(joints[i])`.

<div class="sr-grid2" markdown>
<figure><img class="sr-media" src="../assets/media/dashboard.png" alt="so101 and G1 in the browser twin"><figcaption>an arm and a humanoid, one world</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/dashboard_estop.png" alt="estop latched"><figcaption>e-stop latched until the operator resumes</figcaption></figure>
</div>
