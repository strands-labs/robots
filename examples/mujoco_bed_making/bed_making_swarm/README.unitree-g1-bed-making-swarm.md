# Unitree G1 Bed-Making Swarm (Device Connect)

This document describes the **swarm coordination implementation** for the
two-humanoid bed-making demo (GitHub issue #2). It is intentionally kept
separate from Arm's Device Connect documentation
(`strands_robots/device_connect/GUIDE.md`), which is general-purpose product
documentation — this file covers only the bed-making example.

For the MuJoCo scene, meshes, cloth physics, and how to run the visualisation,
see [`README.unitree-g1-bed-making-demo.md`](README.unitree-g1-bed-making-demo.md).

## Setup

This repo targets **Python 3.12+**. Run from a Python 3.12 venv, and add the Device Connect edge
runtime the swarm driver needs (Isaac Sim 5.1 bundles Python 3.11, so a separate 3.12 venv avoids the
version conflict):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[sim-mujoco]"
pip install device-connect-edge device-connect-agent-tools
```

## Model: equal peers, no control/worker split

Following the issue feedback, the two Unitree G1 humanoids are **equal swarm
peers**. There is no master/worker hierarchy. Both peers independently pursue
the shared goal state **"the bed is made"**, and **AI Fabric acts as the swarm
orchestrator**: every peer can *see* every other peer over
[Arm Device Connect](https://github.com/arm/device-connect) and can **ask any
peer for help** or **offer help** to any peer that asks.

The implementation lives in:

- `examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_g1_driver.py` — the per-peer
  Device Connect driver (`BedMakingG1Driver`) and its `SwarmAgent` state.
- `examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_swarm_demo.py` — the AI-Fabric-orchestrated
  demo that drives the swarm and exercises ask-for-help / offer-help.
- `examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_device_connect_sidecar.py` — registers both
  peers without running a scenario (for dashboard testing).

## Callable actions (functions in the dashboard)

Each peer exposes its high-level behaviours as Device Connect functions
(`@rpc`), so they appear as invocable actions in the portal and can be driven
by an orchestrator via `invoke_device(...)`:

| Function | Purpose |
|---|---|
| `askForHelp(corner, reason, target)` | Ask the swarm (or a specific peer) for help with a sheet corner |
| `offerHelp(target, corner)` | Offer to help a peer that asked (defaults to the last requester) |
| `pickUpBedSheet(corner)` | Pick up a sheet corner |
| `walkToNextCorner(direction)` | Walk to the next bed corner (`counterclockwise` by default) |
| `putDownBedSheet(corner, bed_corner)` | Place a sheet corner on a bed corner |
| `getStatus()` | Full swarm snapshot for the dashboard |
| `getGoalState()` | Shared goal + this peer's progress (`N/4` corners) |
| `listPeers()` | The other peers this robot currently sees |
| `getEventHistory(limit)` | Recent activity events (see below) |
| `getHelpHistory(limit)` | History of help requests and offers (see below) |
| `emergencyStopAll(reason)` | Broadcast a network-wide stop |

The function signatures, parameter types, and docstrings are advertised through
the driver's Device Connect capabilities, so the portal renders typed input
fields and a description for each action.

## Events and autonomous help

When a peer claims, picks up, places, or releases a corner — or asks/offers
help — it emits a Device Connect event (`helpRequested`, `helpOffered`,
`walkingTo`, `goalReached`, `cornerHeld`, `cornerPlaced`, `cornerReleased`,
`claimCorner`, `heartbeat`, `emergencyStop`). Every other peer subscribes
(`@on`) and updates its local view of the swarm.

A peer constructed with `auto_offer=True` **autonomously offers help** when it
sees a `helpRequested` it can service (it is free and not stopped). This makes
the swarm self-organising — the "respond to requests for help" behaviour does
not require the orchestrator to intervene.

## Event history and help-request history

The Device Connect server keeps **no persistent event history** — the portal
exposes only a live event stream. So each peer maintains its own bounded,
timestamped ring buffers (default 256 entries) and exposes them as callable
functions:

- `getEventHistory()` — the full activity log ("picked up corner A", "placed
  corner B on bed corner NE", "GOAL REACHED: the bed is made", …).
- `getHelpHistory()` — a dedicated log of **when one robot asked another for
  help** and when help was offered.

Both logs are queryable live from the dashboard at any time, and are mirrored
on every peer (so each robot's dashboard shows the help requests it has seen).
Example help history after a run:

```text
#6 request  beta-unitree-g1-humanoid-0 asked the swarm for help with corner A (placed corner A is drifting while I place C)
#7 offer    beta-unitree-g1-humanoid-1 offered to help beta-unitree-g1-humanoid-0 with corner A
```

## Running the swarm demo

`examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_swarm_demo.py` plays the role of the AI Fabric
orchestrator: it drives the bed-making sub-tasks via the callable actions and
triggers an ask-for-help → offer-help exchange when a previously placed corner
"starts to drift".

> **Which mode shows the robots?** `--mujoco` is the one that opens MuJoCo and
> physically simulates the two G1s making the bed. `--broker` and `--loopback`
> are **coordination-only** modes: they exercise the Device Connect RPCs,
> events, and ask/offer-help logic and print the event + help history to the
> console — they do **not** open a 3D window.

### Watch the robots in MuJoCo

```bash
# Open the interactive MuJoCo viewer and watch the two G1s make the bed.
# This also registers both peers on the Device Connect dashboard while it runs.
.venv/bin/python examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_swarm_demo.py --mujoco

# Headless: render the run to PNG frames and encode artifacts/.../bed_making.mp4
# (no window needed; uses GPU offscreen rendering).
.venv/bin/python examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_swarm_demo.py --mujoco --render-video

# Skip dashboard registration (pure local visualisation):
.venv/bin/python examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_swarm_demo.py --mujoco --no-device-connect
```

`--mujoco` delegates to the companion visualisation
[`README.unitree-g1-bed-making-demo.md`](README.unitree-g1-bed-making-demo.md)
(`unitree_g1_bed_making_demo.py`) with stable physics settings, so the full
make-the-bed sequence — including the worker-assist "help" step — completes.

### Coordination-only modes (no 3D window)

```bash
# Live against the real Device Connect broker (both G1s appear in the dashboard):
.venv/bin/python examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_swarm_demo.py --broker

# Keep the peers online afterward so you can invoke their functions from the
# dashboard yourself:
.venv/bin/python examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_swarm_demo.py --broker --hold 120

# Offline (no broker) — coordinates via an in-process event bus, prints the
# full event + help history at the end:
.venv/bin/python examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_swarm_demo.py --loopback
```

Defaults:

- Broker: `nats://fabric.deviceconnect.dev:4222` (override with `--nats-url`
  or `$DEVICE_CONNECT_NATS_URL`).
- Credentials: `.credentials/beta-unitree-g1-humanoid-0.creds.json` and
  `.credentials/beta-unitree-g1-humanoid-1.creds.json` (the embedded internal
  broker URL is overridden so the peers reach the public fabric broker).

In `--broker` mode the orchestrator drives the live peers in-process; their
presence, status, and emitted events traverse the fabric broker so the
dashboard reflects every transition, and their functions remain independently
invocable from the portal UI. (An external `device-connect-agent-tools` client
is intentionally not used: it requires its own client credentials and its
`connect()` targets the Zenoh backend, whereas these device peers authenticate
to the NATS fabric with their JWT device credentials.)

## Register the peers without a scenario (sidecar)

The two peers are only present in the Device Connect dashboard while some
process is hosting their `DeviceRuntime`. The swarm demo and the MuJoCo
visualisation host the peers only for the duration of their run, then exit
and unregister the robots. The **sidecar hosts the two runtimes on their
own** — no simulation, no scripted scenario — which is useful when you want
to:

- keep the peers **online and idle** in the dashboard for as long as you
  like, so a human or an AI Fabric orchestrator can invoke their callable
  functions interactively (click `askForHelp` / `pickUpBedSheet` in the
  portal, or call them via `device-connect-agent-tools`);
- bring the peers up on a machine that **cannot run the simulation** (no
  MuJoCo assets, no GPU/GL, headless CI) — registration only needs the
  network and the credentials;
- validate Device Connect connectivity, credentials, and the dashboard
  presence/RPC surface **in isolation** from the simulation, which makes
  broker/auth debugging much easier.


```bash
.venv/bin/python examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_device_connect_sidecar.py
```

Override credentials with `--credentials role=path` (repeatable).

## Tests

- `tests/test_bed_making_swarm_driver.py` — callable actions, goal tracking,
  event/help history, peer visibility, autonomous help response.
- `tests/test_bed_making_swarm_demo.py` — end-to-end swarm scenario over the
  in-process loopback bus (no broker required).

## Device Connect version

Uses `device-connect-edge` / `device-connect-agent-tools` **0.2.4**, from the
Arm source repo <https://github.com/arm/device-connect> (the PyPI wheels are
published from the same source and are interchangeable). The example driver imports `DeviceDriver` / `emit` / `on` / `periodic` /
`rpc` and the runtime types directly from `device_connect_edge`, so the
example is self-contained and does not depend on `strands_robots` internals.
