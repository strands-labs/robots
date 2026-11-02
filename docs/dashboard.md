---
description: "Serve a world to the browser: a live 3D view, joint controls, cameras, chat, voice and an e-stop for every robot."
---

# Dashboard

The dashboard shows a world in your browser. You get a live 3D view, joint sliders, cameras, an agent chat and
an e-stop for each robot.

{{drawing:v07_dashboard}}

```bash
python -m strands_robots.dashboard                       # the microduck
python -m strands_robots.dashboard so101 g1 --port 8091  # an arm and a humanoid
```

Open `http://127.0.0.1:8090`. You need the `dashboard` and `mujoco` extras. For the chat, add the `tools` extra
and pass a Bedrock model id with `--model`.

## From Python

<!-- test: skip, serves until Ctrl-C -->
```python
from strands_robots import Robot, Simulation
from strands_robots.dashboard import Dashboard

arm = Robot("so101")
Dashboard(Simulation.default()).serve(port=8090)
```

| Argument | Does |
|---|---|
| `robots=[real_arm]` | Adds hardware robots |
| `mesh=lab` | Lists every robot on a [mesh](mesh.md) |
| `model=` | Sets the chat model |
| `voice="openai"` | Turns on voice: `"openai"`, `"nova"` or `"auto"` |

## Tabs

| Tab | Does |
|---|---|
| Connect | Attaches robots from other machines over Zenoh, and adds remote LeRobot policies |
| Record | Records LeRobot v3 datasets, see [Recording a dataset](recording.md) |

## Voice

With `voice=`, you can talk to the page. The voice agent has the same tools as the chat. `"openai"` uses
gpt-realtime-2 and needs `OPENAI_API_KEY`. `"nova"` uses Amazon Nova Sonic on Bedrock. Install the `voice` extra.

## Security

The server listens on loopback only and refuses requests from other web pages. A command to a real robot needs
the operator's `X-Confirm` header. Voice never sends that header, so voice cannot move a real robot.

## Live view in these docs

When you serve these docs locally, the home page looks for a dashboard on port 8090 or 8091 and shows it live.
Add `?live=http://127.0.0.1:<port>/` to the home page URL to use another port.

<div class="sr-grid2" markdown>
<figure><img class="sr-media" src="../assets/media/dashboard.png" alt="so101 and G1 in the browser"><figcaption>An arm and a humanoid in one world</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/dashboard_estop.png" alt="e-stop latched"><figcaption>An e-stop stays on until the operator resumes</figcaption></figure>
</div>
