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

Open `http://localhost:8090` and sign in with a passkey (see [Signing in](#signing-in)). You need the `dashboard`
and `mujoco` extras. For the chat, add the `tools` extra
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
| `auth=` | Passkey sign-in: `True` (default), an `Auth(ttl=..., store=...)`, or `False` (see below) |

## Signing in

Every `/api` route and every socket needs a signed-in session. Only the page itself and the sign-in routes are open.
You sign in with a passkey: Touch ID, Windows Hello, a phone or a security key. The private key never leaves that
device. The dashboard keeps only public keys, in `~/.cache/strands_robots/dashboard-auth.json` (only you can read it;
`STRANDS_ROBOTS_DASHBOARD_AUTH` moves it). Deleting that file signs everyone out and starts setup again.

**First run.** With no passkey yet, the server prints a one-time link when it starts:

```text
dashboard: no passkey yet; open http://localhost:8090/?setup=<code> to create one (one-time link)
```

Open it and press **Create passkey**. Without the code nobody can create the first passkey, so another program on
this machine cannot get there before you. Passkeys work on `http://localhost`, not on `http://127.0.0.1`: the page
on `127.0.0.1` sends you to `localhost`.

**More passkeys.** In the **Passkeys** tab you can add a passkey (for another browser or device), rename or remove
one, and sign out. Adding a passkey needs a signed-in session. The last passkey cannot be removed. A session lasts
12 hours (`Auth(ttl=...)` or `STRANDS_ROBOTS_DASHBOARD_SESSION_TTL`, in seconds). Removing a passkey ends its sessions.

**Scripts, notebooks and agents.** Ask the dashboard for a token and send it as a bearer header:

```python
import httpx

from strands_robots import Robot, Simulation
from strands_robots.dashboard import Dashboard

sim = Simulation()
Robot("so101", sim=sim)
dash = Dashboard(sim)
server = dash.start()
token = dash.auth.token()  # valid for the session length; keep it secret
print(httpx.get(f"{server.url}/api/scene", headers={"Authorization": f"Bearer {token}"}).status_code)  # 200
server.stop()
sim.close()
```

From the command line, `--print-token` prints one when the server starts.

**Turning sign-in off.** `Dashboard(sim, auth=False)` (or `--no-auth`) logs a warning and lets any local program use
every route. It is refused while a real robot is attached. You can force it with `unauthenticated_hardware=True`, but
then any program that can reach the port can move that robot.

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

The server listens on loopback only, refuses requests from other web pages, and answers `/api` and the sockets only
to a signed-in session. A command to a real robot also needs the operator's `X-Confirm` header, which the page sends
after asking. Voice never sends that header, so voice cannot move a real robot.

## Live view in these docs

Serve these docs locally and the home page shows a dashboard running on port 8090 or 8091.
Add `?live=http://127.0.0.1:<port>/` to the home page URL to use another port. The docs page is another site, so the
session cookie does not reach the embedded dashboard: start that simulated dashboard with `--no-auth`.

<div class="sr-grid2" markdown>
<figure><img class="sr-media" src="../assets/media/dashboard.png" alt="so101 and G1 in the browser"><figcaption>An arm and a humanoid in one world</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/dashboard_estop.png" alt="e-stop latched"><figcaption>An e-stop stays on until the operator resumes</figcaption></figure>
</div>
