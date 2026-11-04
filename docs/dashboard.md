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

The page's two browser engines (MuJoCo and ONNX Runtime for the web, 20 MB) are not in the package. The first page
load downloads them once, checks their hashes and keeps them in `~/.cache/strands_robots/dashboard`. On a robot
network with no internet, run `python -m strands_robots.dashboard --fetch` on a machine that has it and copy that
folder over.

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

Chat and voice are off unless you turn them on. The chat panel says how.

```bash
python -m strands_robots.dashboard so101 g1 --model <bedrock model id> --voice auto
```

| Flag | Turns on | Needs |
|---|---|---|
| `--model <id>` | the chat agent (`model=` to `Dashboard`) | the `tools` extra, AWS credentials |
| `--voice auto` · `openai` · `nova` | talking to the page (`voice=` to `Dashboard`) | the `voice` extra; `OPENAI_API_KEY` for `openai`, AWS credentials for `nova` |

The voice agent has the same tools as the chat. `"openai"` uses gpt-realtime-2. `"nova"` uses Amazon Nova Sonic
on Bedrock.

## Security

The server listens on loopback only and refuses requests from other web pages. A command to a real robot needs
the operator's `X-Confirm` header. Voice never sends that header, so voice cannot move a real robot.

## Live view in these docs

Serve these docs locally and the home page shows a dashboard running on port 8090 or 8091.
Add `?live=http://127.0.0.1:<port>/` to the home page URL to use another port.

<div class="sr-grid2" markdown>
<figure><img class="sr-media" src="../assets/media/dashboard.png" alt="so101 and G1 in the browser"><figcaption>An arm and a humanoid in one world</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/dashboard_estop.png" alt="e-stop latched"><figcaption>An e-stop stays on until the operator resumes</figcaption></figure>
</div>
