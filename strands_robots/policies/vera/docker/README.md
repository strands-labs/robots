# VERA server — Docker

Run the VERA policy server in a container with VERA's full GPU stack
(PyTorch 2.6 / CUDA 12.4 + VGGT) **isolated from the host robots venv**. The
`VeraPolicy` provider connects over the websocket protocol — the host never
installs VERA's heavy/conflicting deps.

```
host robots venv (numpy>=2)          vera-server container (torch 2.6, CUDA 12.4)
  VeraPolicy ─ VeraWebsocketClient ─ws─▶ vera.server.start_vera_server
  server_mode="docker"                   (VERA + VGGT + sim + /ckpts mounted)
```

## Prerequisites

- NVIDIA GPU + driver, Docker, and the **NVIDIA Container Toolkit**
  (`--gpus all` must work: `docker run --rm --gpus all nvidia/cuda:12.4.0-base nvidia-smi`).
- Downloaded checkpoints:
  ```bash
  hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts
  export VERA_CKPT_ROOT=$PWD/vera-ckpts
  ```
- **MimicGen only** — also download the frozen **Wan2.1-T2V-1.3B** base
  (text-encoder + VAE + CLIP; NOT bundled in `sizhe-li/VERA`):
  ```bash
  hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
  export VERA_WAN_CKPT_ROOT=$PWD/Wan2.1-T2V-1.3B
  ```
  PushT needs neither the WAN base nor wandb — it is fully local.

## Build

```bash
# from the strands-robots repo root
docker build -f strands_robots/policies/vera/docker/Dockerfile \
    -t strands-vera-server:latest .
# pin VERA:  --build-arg VERA_REF=<commit-or-tag>
```

## Run

**Manually** (PushT):

```bash
docker run --rm --gpus all -p 8820:8820 -p 8821:8821 \
    -v "$VERA_CKPT_ROOT":/ckpts:ro \
    -e VERA_EMBODIMENT=pusht \
    strands-vera-server:latest
```

**Manually** (MimicGen — needs the WAN base mounted at `/wan` too):

```bash
docker run --rm --gpus all -p 8800:8800 -p 8801:8801 \
    -v "$VERA_CKPT_ROOT":/ckpts:ro \
    -v "$VERA_WAN_CKPT_ROOT":/wan:ro \
    -e VERA_EMBODIMENT=mimicgen \
    strands-vera-server:latest
```

**Compose**:

```bash
docker compose -f strands_robots/policies/vera/docker/docker-compose.yml up
# MimicGen (set the WAN base first):
#   export VERA_WAN_CKPT_ROOT=/abs/Wan2.1-T2V-1.3B
#   VERA_EMBODIMENT=mimicgen docker compose -f .../docker-compose.yml up
```

**Provider-managed** (the provider starts/stops the container for you):

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "vera",
    embodiment="pusht",
    server_mode="docker",  # <- manage the container, not a subprocess
    ckpt_root="/abs/path/vera-ckpts",
)
chunk = policy.get_actions_sync(obs, "push the T to the goal")
policy.close()  # stops the container it started
```

## Checkpoint wiring

The entrypoint maps the single mounted `/ckpts` root (the `hf download` layout)
onto VERA's per-embodiment checkpoint env vars:

| Embodiment | Maps |
|------------|------|
| `pusht` | `pusht-dfot/model.ckpt` → `VERA_PUSHT_PLANNER_CKPT`; `pusht-idm/model.ckpt` → `VERA_PUSHT_DYNAMICS_CKPT` |
| `mimicgen` | `mimicgen-wan-1.3b/algo_config.yaml` → `VERA_ALGO_CONFIG`; `mimicgen-wan-1.3b/` → `VERA_MIMICGEN_CKPT_DIR` (DiT + flow_decoder); WAN base `/wan` → `VERA_WAN_CKPT_ROOT`; IDM run `37oa162u` (the proven stack), resolved **offline** from `/ckpts` via `wandb_offline_resolve.py` |

An explicit `-e VERA_…` always overrides the auto-wiring.

## Ports

| Embodiment | policy | viz |
|------------|:------:|:---:|
| pusht | 8820 | 8821 |
| mimicgen | 8800 | 8801 |

## Notes

- First boot loads the WAN/DFoT planner — the provider's health-check waits up
  to `server_ready_timeout` (default 600s).
- Headless rendering uses EGL (`MUJOCO_GL=egl`); no X server needed.
- `flash-attn` is optional (WAN falls back to SDPA on the NGC base).
- **Offline IDM resolution:** VERA loads its Jacobian IDM by wandb run id. The
  container resolves that run id to the locally-mounted `/ckpts` dir via each
  checkpoint's `provenance.json` (`wandb_offline_resolve.py`) and sets
  `WANDB_MODE=offline`, so **no wandb network / API key is needed**. Both the
  proven mimicgen IDM (`37oa162u`) and the omni default (`x21o0cwe`) resolve
  locally; an unknown run id transparently falls back to wandb.
- The container only **serves** actions; the host runs the sim. The heavy
  `eval` extra (robosuite/mimicgen) is therefore **not** installed by default
  (build with `--build-arg VERA_WITH_EVAL=1` to bundle it).
