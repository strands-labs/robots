---
description: VERA video-to-action policy (two-stage DFoT/WAN planner + Jacobian IDM) over a containerized GPU server. MimicGen MuJoCo rollouts with IK for eef-delta arms.
---

# VERA policy provider

**VERA** ([Video-to-Embodied Robot Action](https://github.com/sizhe-li/VERA),
MIT/CSAIL) is a **two-stage, closed-loop video-to-action** policy:

1. **Video planner** (DFoT / WAN) — a diffusion model that "dreams" the next
   frames from the current observation (+ optional text). **Embodiment-agnostic.**
2. **Jacobian IDM** — translates the dream into robot actions via a frozen
   visual backbone (VGGT/DINO) + a flow→action head. **Embodiment-specific**,
   data-efficient, swappable without retraining the planner.

> *One video planner, many IDMs* — the route to zero-shot, cross-embodiment control.

The strands-robots `vera` provider is a thin, import-light **WebSocket client +
managed GPU server**, mirroring the `cosmos3` service pattern. The host venv
never installs VERA's heavy/conflicting stack (PyTorch 2.6 / CUDA, VGGT, DFoT) —
that lives in the `strands-vera-server` container.

```
host venv (numpy>=2)                 vera-server container (torch 2.6 / CUDA)
  VeraPolicy ─ VeraWebsocketClient ─ws─▶ vera.server.start_vera_server
  (no vera install)                      DFoT/WAN planner + Jacobian IDM + /ckpts
```

## Quick start

```python
from strands_robots.policies import create_policy

# Attach to a running server (see "Server" below) ...
policy = create_policy("vera", embodiment="mimicgen", auto_launch_server=False)
chunk = policy.get_actions_sync(observation, "stack the red block on the green block")

# ... or let the provider manage the container for you:
policy = create_policy(
    "vera", embodiment="mimicgen",
    server_mode="docker", ckpt_root="/abs/path/vera-ckpts",
)
```

The **MimicGen → Panda** path drives a real 7-DoF arm: the WAN planner + Jacobian
IDM emit 6-DoF end-effector deltas, and the provider's IK bridge solves them onto
the Panda's joints (auto-discovering the end-effector frame — no manual wiring).
See [`examples/vera_mimicgen_panda/`](https://github.com/strands-labs/robots/tree/main/examples/vera_mimicgen_panda).

<figure markdown>
  ![VERA MimicGen Panda rollout](../assets/vera/mimicgen_panda.gif)
  <figcaption>VERA MimicGen policy on a Franka Panda — WAN dream + AllTracker +
  Jacobian IDM → eef-deltas → VERA IK bridge → joint targets → MuJoCo.</figcaption>
</figure>

## Embodiments

From VERA's `adapter_factory._EMBODIMENTS`:

| Embodiment | action_space | dims | views | control | ports (policy/viz) | checkpoints |
|------------|--------------|:----:|-------|:-------:|:------------------:|-------------|
| **pusht** | `velocity` (planar) | 2, no gripper | `image` | 10 Hz | 8820 / 8821 | experimental — IDM du path not wired end-to-end upstream |
| **mimicgen** | `eef_delta` | 7 (6-DoF + grip) | `agentview_image`, `robot0_eye_in_hand_image` | 20 Hz | 8800 / 8801 | ✅ Wave-1 (+WAN base) |
| **allegro** | `joint_position` | 16 | 12 cameras | 15 Hz | 8802 / 8803 | 🔜 Wave-2 (code only) |
| **droid** | `cartesian_delta` | 7 | `varied_1`,`varied_2`,`hand` | 15 Hz | 8804 / 8805 | 🔜 Wave-2 (code only) |

**Today, end-to-end:** `mimicgen` (WAN planner; needs the frozen WAN base + a
motion tracker) is the working, faithful embodiment — it exercises the whole
eef-delta → IK path onto a real arm. `pusht`'s server runs, but its IDM `du`
action path is **not wired end-to-end upstream** (VERA's own
`configurations/dataset/pusht.yaml` documents this gap), so it validates the
provider → server → action plumbing rather than producing a solving rollout —
treat it as experimental. `allegro`/`droid` are code-present but
checkpoint-absent upstream (Wave 2).

### The "generalist" claim, accurately

VERA's *architecture* is cross-embodiment: **one** embodiment-agnostic video
planner + **one cheap IDM per robot** (frozen backbone, head trained from
self-play). It is **not** a single checkpoint that drives every robot today — a
robot is drivable iff (its `action_space` matches a served embodiment) **and**
(a checkpoint exists) **and** (IK/validation is done for that arm). For
`eef_delta`/`cartesian_delta` arms (mimicgen/droid), the provider includes an
**IK bridge** that maps the 6-DoF end-effector deltas to joint targets and
auto-discovers the end-effector frame from the compiled MuJoCo model — so any
kinematically-compatible 6/7-DoF arm can be driven once a matching IDM exists.

## Checkpoints

```bash
hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts   # ~42 GB full; ~4 GB is Wave-1
export VERA_CKPT_ROOT=$PWD/vera-ckpts
```

MimicGen additionally needs the **frozen WAN 2.1 base** (text-enc + VAE + CLIP).
Its IDM uses the **AllTracker** point tracker, which the container bundles
(cloned at build time; weights auto-download). The WAN base:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
```

The provider **never auto-downloads** — point it at pre-downloaded roots.

## Server

The server holds the GPU and the two-stage model. Run it as a container:

```bash
docker build -f strands_robots/policies/vera/docker/Dockerfile -t strands-vera-server:latest .

# MimicGen (serves ws on :8800; needs the WAN base + offline resolver)
docker run --rm --gpus all --ipc=host -p 8800:8800 \
    -v "$VERA_CKPT_ROOT":/ckpts:ro -v "$PWD/Wan2.1-T2V-1.3B":/wan:ro \
    -e VERA_EMBODIMENT=mimicgen -e USE_OFFLINE_RESOLVE=1 \
    strands-vera-server:latest
```

The container entrypoint maps the single mounted `/ckpts` root onto VERA's
per-embodiment checkpoint env vars; `USE_OFFLINE_RESOLVE=1` resolves MimicGen's
wandb-run-id IDM to the locally-mounted checkpoint (via `provenance.json`) so the
server boots with **no network**. See
[`policies/vera/docker/`](https://github.com/strands-labs/robots/tree/main/strands_robots/policies/vera/docker).

`server_mode="docker"` lets the provider build/run/stop the container itself;
`server_mode="subprocess"` launches a local `python -m vera.server...` when VERA
is installed in the same env.

Either mode configures the server from the same `VeraConfig`, so every value in
the table below reaches it whichever one runs. The two modes get it there by
different routes: the subprocess inherits the whole environment overlay, while
the container takes a scalar (a backend name, a run id) as `-e` and a host path
as a read-only bind mount forwarded under the container path it was mounted at
(`ckpt_root` -> `/ckpts`, `wan_ckpt_root` -> `/wan`). A value the container
command failed to pass would not be reported: `docker run` has nothing to
object to, so the server would simply start on the default the caller overrode.

## Configuration

`VeraConfig` maps 1:1 to VERA's server flags and is env-overridable (deploy/CI
wins over code defaults):

| kwarg | env var | maps to |
|-------|---------|---------|
| `embodiment` | — | `--embodiment` |
| `server_port` / `vis_port` | `VERA_SERVER_PORT` / `VERA_VIS_PORT` | `--port` / `--vis-port` |
| `algo_config` | `VERA_ALGO_CONFIG` | `--algo-config` (swap to the omni planner) |
| `dynamics_run_id` | `VERA_DYNAMICS_RUN_ID` | `--dynamics-run-id` |
| `text_prompt` | `VERA_TEXT_PROMPT` | `--text` |
| `ckpt_root` | `VERA_CKPT_ROOT` | container `/ckpts` mount |
| `sample_steps` | `VERA_SAMPLE_STEPS` | `--sample-steps` |
| `tracker_backend` | `VERA_TRACKER_BACKEND` | IDM tracker |
| `motion_plan_scale` | `VERA_MOTION_PLAN_SCALE` | live `configure` |
| `server_mode` | `VERA_SERVER_MODE` | `subprocess` \| `docker` |

Both ports take the shared TCP-port domain every port-dialing provider applies:
an `int` in `[1, 65535]`, or `None` for the per-embodiment default. The value is
checked once, on the config, because three consumers read it — the client dials
it, the runner launches the server on it, and `VeraConfig.server_uri` reports it
— so a value outside the range is not merely refused late but resolved
differently by each of them. `vis_port = 0` is the one exception and disables the
live viewer (`--vis-port` is omitted). The `VERA_*_PORT` overrides go through the
same check.

`motion_plan_scale` takes the same domain as the two IK scales below: a positive
finite number, or `None` to leave the server's own scale alone. `0` is not the
opt-out — it scales the plan to nothing — so `None` is the off switch and `0` is
refused. It is checked on the config, not where it is used, because where it is
used cannot refuse it: `_ensure_started` applies it after the server handshake
with a best-effort `configure` call whose failure is logged at INFO and does not
stop the rollout, so a value `float()` cannot convert is neither applied nor
reported. `VERA_MOTION_PLAN_SCALE` goes through the same check; an unparsable
spelling still falls back to `None`, as it does for the ports.

### IK conversion knobs

Three keyword-only numbers shape every joint target the eef-delta path produces,
and each is checked where it is supplied because each is *applied* rather than
forwarded — nothing downstream can refuse them usefully:

| kwarg | surface | domain |
|-------|---------|--------|
| `rotation_dim` | `set_ik_target(...)`, `decode_vera_delta_chunk_to_targets(...)` | `3` (axis-angle) or `6` (rot6d) — the encodings the decoder implements (`None` on the setter keeps the embodiment's convention) |
| `translation_scale` | `set_ik_target(...)`, `decode_vera_delta_chunk_to_targets(...)` | a positive finite number (`None` on the setter leaves the current value) |
| `ik_smoothing` | `VeraPolicy(...)` | `[0, 1)` — `0` disables the smoothing |

`rotation_dim` is an enumeration rather than a range: `delta_to_matrix` implements
axis-angle and rot6d and raises for any other width, so there is no third encoding
to ask for. It is refused at the surface that receives it because it is not refused
usefully later — `0`/`2`/`4` reach that dispatch *mid-rollout*, inside
`get_actions`, after the server handshake and the IK bridge build; a fractional or
non-numeric width reaches the per-step slice `step[3 : 3 + rotation_dim]` and
raises `TypeError: slice indices must be integers`, naming neither the parameter
nor the surface. An integral float (`3.0`, what a config read produces) is accepted
and normalized, since the width has to arrive at that slice as an index.

`translation_scale` multiplies every translation delta on top of the OSC position
scale, so `0` discards the translation half of each action and returns a
rotation-only chunk, a negative value inverts it, and `nan`/`inf` make *every*
returned joint target non-finite — refused one layer later by `send_action`,
where it reads as a wrong-embodiment action-key mismatch rather than as the scale
that caused it. `ik_smoothing` weights the *previous* target in the EMA
`target = (1 - alpha) * solved + alpha * previous`, so `1.0` freezes the arm at
its first solved pose, above `1.0` the targets diverge away from the solution
(measured at `-5.9x` the solved joint travel for `1.5`), and a negative or `nan`
value fails the `alpha > 0` test the blend is gated on — silently applying no
smoothing at all.

## Wire protocol

The provider keeps a rolling **context window** of the last `context_frames`
camera frames (width-concatenated across views) and calls the server's chunked
`infer` when its local action queue drains — the same `RemotePolicy` contract
VERA's own eval harness uses. The server returns `{"action": [H, D]}`; the
provider maps each `D`-vector to robot actuator names (gripper binarized per the
server's `gripper_dim_index`/`gripper_is_raw`), coercing to python floats per the
`Policy` ABC.

## QA the rollouts with a Cosmos 3 reasoner (closed loop)

VERA *generates* video-grounded actions; [NVIDIA Cosmos 3](https://github.com/cagataycali/strands-cosmos)
*reads* video and reasons in text — so it makes a natural **automated QA critic**
for rollouts. Serve the reasoner, then have it grade a rollout MP4:

```bash
uv pip install "strands-cosmos[cosmos3]"
# serve Cosmos3-Nano on :8000 (vLLM + vllm-cosmos3); see strands-cosmos `c3-serve-reason`
python examples/vera_mimicgen_panda/critique_with_cosmos3.py     examples/vera_mimicgen_panda/artifacts/mimicgen_panda.mp4
```

```python
from strands import Agent
from strands_cosmos import Cosmos3ReasonerModel

agent = Agent(model=Cosmos3ReasonerModel(base_url="http://localhost:8000/v1"))
print(agent("Grade this robot rollout — is the motion smooth and purposeful, "
            "any bugs? <video>/tmp/vera-critique/mimicgen_panda.mp4</video>"))
```

This closed `generate → reason → fix` loop surfaced (and fixed) real issues in
the MimicGen→Panda example: an initial **jittery** critique drove the
`ik_smoothing` EMA knob, and a **"the arm is static"** critique root-caused a
near-singular default start pose with the motion off-camera — fixed with a
tabletop-ready seed pose + camera framing. The reasoner's verdict moved
**NEEDS-WORK → PASS**.

## Testing

```bash
# offline unit tests (no GPU, no vera install)
hatch run test tests/policies/vera/

# gated live integration (needs a running server)
VERA_LIVE=1 hatch run test-integ tests_integ/policies/vera/
```

## Install

```bash
# 1. Provider client deps. There is no `vera` extra: the provider needs only a
#    msgpack + websocket client, and VERA itself is distributed as a git
#    repository, which no extra can pull (PyPI rejects package metadata
#    carrying a VCS reference).
pip install strands-robots websockets msgpack 'numpy>=1.24'

# 2. VERA itself, for the managed-server (subprocess) mode.
pip install 'vera @ git+https://github.com/sizhe-li/VERA.git'

# 3. Only for the MimicGen sim example: MimicGen sim deps (also pulls the
#    experimental PushT env, plus `imageio>=2.28.0,<3.0.0` for the rollout
#    clips - the releases below that floor either raise out of the clip encoder
#    or write a GIF no decoder can open).
pip install 'strands-robots[vera-sim]'
```

> **Note on MimicGen.** The `vera-sim` extra does **not** install NVlabs
> MimicGen: that project has no PyPI release, and the `mimicgen` name on
> PyPI is an unaffiliated package (a dependency-confusion risk), so it is
> intentionally not pinned here. The `mimicgen` VERA *embodiment* is just a
> config string and needs no such package. If you genuinely need NVlabs
> MimicGen for data generation, install it from source:
>
> ```bash
> pip install "mimicgen @ git+https://github.com/NVlabs/mimicgen.git"
> ```

> **Note on robomimic.** For the same reason the `vera-sim` extra does **not**
> pin `robomimic`: its highest PyPI release is `0.3.0`, while VERA's examples
> target v0.5.0, which exists only as an ARISE-Initiative GitHub tag. Pinning
> `robomimic==0.5.0` would be both unresolvable (it wedges `uv lock`) and a
> dependency-confusion vector. robomimic is not imported by strands-robots;
> VERA pulls it in itself. If you need v0.5.0, install it from source:
>
> ```bash
> pip install "robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.5.0"
> ```

For the **docker** path the host needs only `websockets` + `msgpack` (the client
transport) — no `vera`, no torch.
