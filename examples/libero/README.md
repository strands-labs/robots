# LIBERO manipulation-benchmark evaluation

LIBERO manipulation-benchmark evaluation driven by the in-tree simulation
backends: the default MuJoCo backend (`strands_robots.simulation.Simulation`)
and the Isaac Sim backend (`strands_robots.simulation.isaac.IsaacSimulation`,
resolved via `create_simulation("isaac")`). One driver (`run.py`) with backend
subcommands shares the CLI base, the GR00T container orchestration, the
eval-and-report loop, and the two grep-stable output lines; only thin setup
shims differ per backend (Isaac loads a real Franka USD and adds explicit RTX
cameras - Isaac does not auto-attach viewport cameras the way MuJoCo does).

## Files

| File | What it shows |
|------|--------------|
| [`run.py`](run.py) | LIBERO eval with backend subcommands. `run.py mujoco`: `Simulation()` -> `create_world` -> `add_robot` -> LIBERO scene pre-warm -> `evaluate_benchmark`, with whole-run MP4 recording; **GPU: Optional** (`--policy mock` runs headless on CPU). `run.py isaac`: `create_simulation("isaac")` -> `create_world` -> `add_robot` (real Franka USD) -> `add_camera` -> `evaluate_benchmark`, with a synchronous rollout-MP4 recorder; **GPU: Yes** (Isaac Sim 6.0+). Both emit the same two grep-stable lines for the backend matrix. |
| [`run_mujoco_agent.py`](run_mujoco_agent.py) | Same MuJoCo eval, driven by a Strands `Agent` in natural language over the full registered `Simulation` tool surface; the script keeps container lifecycle, scene pre-warm, and recording deterministic. |
| [`run_isaac_agent.py`](run_isaac_agent.py) | Same Isaac eval, driven by a Strands `Agent` in natural language via a single `@tool`-wrapped `evaluate_isaac_benchmark`. |
| [`libero_backend_matrix.py`](libero_backend_matrix.py) | Runs one LIBERO task across whichever backend subcommands the host can execute (subprocess-and-parse) and prints a side-by-side `success_rate` / `wall_time` table. Missing drivers show `unavailable`; hosts without Isaac Sim show `skip`. |
| [`gr00t_server_deterministic_wrapper.py`](gr00t_server_deterministic_wrapper.py) | Back-compat shim: the determinism wrapper moved into the package (`strands_robots/policies/groot/server_wrapper.py`, shipped in the wheel). Prefer `gr00t_inference(..., deterministic=True)` (or `run.py <backend> --deterministic`), which mounts and runs it for you; mount the packaged file by hand only for custom container setups. |

## Install

```bash
# run.py mujoco / libero_backend_matrix.py
pip install "strands-robots[sim-mujoco,benchmark-libero]"

# run_mujoco_agent.py additionally needs the Strands agent SDK + an LLM provider
pip install "strands-robots[sim-mujoco,benchmark-libero]" strands-agents
```

The Isaac backend imports stay lazy - installing the extra never imports Isaac
Sim, and `run.py` imports the chosen backend only after subcommand parsing -
but running the Isaac eval needs a working Isaac Sim 6.0+ host (RTX GPU,
Ubuntu 22.04+, CUDA 12+, Python 3.12; installed via the Omniverse Launcher,
Isaac Lab, or the `nvcr.io/nvidia/isaac-sim:6.0` NGC docker image). A pure
`pip install` does not provide the full `isaacsim-extscache-*` extension set
`SimulationApp` needs to boot.

```bash
# run.py isaac
pip install "strands-robots[sim-isaac,benchmark-libero]"

# run_isaac_agent.py additionally needs the Strands agent SDK + an LLM provider
pip install "strands-robots[sim-isaac,benchmark-libero]" strands-agents
```

## Run

```bash
# Smoke test on the default MuJoCo backend (mock policy; no GPU / Docker):
python examples/libero/run.py mujoco --policy mock --n-episodes 5

# Real LIBERO eval against nvidia/GR00T-N1.7-LIBERO (Docker + NVIDIA GPU +
# ~30 GB free disk for the checkpoint; auto-orchestrates the GR00T container):
python examples/libero/run.py mujoco --policy groot --port 8000 --n-episodes 50

# Agent-driven MuJoCo variant (needs a configured LLM provider for Strands):
python examples/libero/run_mujoco_agent.py --policy mock --n-episodes 5

# Smoke test on Isaac Sim (needs Isaac Sim on the host). Loads the bundled
# Franka Panda USD resolved from the Isaac assets root over the Omniverse CDN:
python examples/libero/run.py isaac --policy mock --n-episodes 5

# Bring your own robot asset (Isaac-only flags):
python examples/libero/run.py isaac --policy mock --robot-usd /path/to/robot.usd
python examples/libero/run.py isaac --policy mock --robot-urdf /path/to/robot.urdf

# Real LIBERO eval on Isaac against nvidia/GR00T-N1.7-LIBERO:
python examples/libero/run.py isaac --policy groot --port 8000 --n-episodes 50

# Agent-driven Isaac variant (needs a configured LLM provider for Strands):
python examples/libero/run_isaac_agent.py --policy mock --n-episodes 5

# Side-by-side across every installed backend (mock by default; missing
# drivers -> `unavailable`, non-Isaac hosts -> `skip`):
python examples/libero/libero_backend_matrix.py
```

The `isaac-4096` row of `libero_backend_matrix.py` references a fleet driver
(`run_isaac_fleet.py`) that has not been migrated yet; until it lands, the
matrix prints an `unavailable (no run_isaac_fleet.py)` row for it and
evaluates the `mujoco` and Isaac single-env (`isaac-1`) rows.

## Environment variables

- `MUJOCO_GL=egl` / `PYOPENGL_PLATFORM=egl` - headless rendering on a no-display
  host (set automatically by `libero_backend_matrix.py` for its subprocesses).
- `STRANDS_ISAAC_RTX_PATHTRACING=1` - upgrade the Isaac `render_mode` from
  `rtx_realtime` to photoreal pathtracing.
- `STRANDS_ROBOTS_CHECKPOINT_DIR` - override the GR00T checkpoint cache dir
  (`--policy groot`). Defaults to a non-`/home` path so it clears
  `gr00t_inference`'s `start_container` mount guard.
- `STRANDS_GR00T_IMAGE` / `STRANDS_GR00T_IMAGE_ALLOW` - operator-configured
  GR00T docker image + allowlist (`--image` sets these).
- `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) - HuggingFace token for the gated
  GR00T checkpoint download (`--policy groot`). With neither set, the drivers
  fall back to the Hub's cached login (`hf auth login`), asking
  `huggingface_hub` where that lives - so a relocated `HF_HOME` /
  `XDG_CACHE_HOME` / `HF_TOKEN_PATH` is honoured.
- `STRANDS_GR00T_SERVER_SEED` / `STRANDS_GR00T_STRICT_DETERMINISTIC` - consumed
  by the packaged determinism wrapper
  (`strands_robots/policies/groot/server_wrapper.py`) inside the GR00T
  container; forwarded there automatically by
  `gr00t_inference(..., deterministic=True)` / `run.py <backend> --deterministic`.
