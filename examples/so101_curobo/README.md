# SO-101 synthetic data generation with cuRobo (Isaac / MuJoCo)

Interactive demo: set up an SO-101 tabletop pick-and-place world, plan a motion
with **[cuRobo](https://nvlabs.github.io/curobo/)**, execute it in simulation,
and record the rollout as a **LeRobot dataset** for policy training — repeatable
across randomized scenes for scale.

The control + data-collection loop is backend-agnostic (it speaks the
`SimEngine` surface), so the *same* code runs on MuJoCo today and Isaac once the
`strands-robots[sim-isaac]` runtime is installed.

> **Note on the cuRobo policy provider:** this example ships its own
> demo-local `CuroboMotionPlanner` (`planner.py`) that lazy-imports cuRobo
> directly and pairs it with a dependency-free `ScriptedPlanner` fallback. It is
> intentionally independent of the in-tree `create_policy("curobo")` policy
> provider (`strands_robots.policies.curobo`) — the example never imports that
> provider, so there is no overlap or conflict. Both target the same restructured
> cuRobo `main` API (`curobo.motion_planner.MotionPlanner`).

## What runs today vs. what needs more runtime

| Capability | Status |
|---|---|
| SO-101 scene + execute + **LeRobot dataset** recording | ✅ works now (MuJoCo, real SO-101) |
| Scripted joint-space pick-and-place planner | ✅ works now (demonstrative motion; grasps not guaranteed) |
| CPU/CI smoke (state+action, no GL) | ✅ `smoke_test.py` |
| Strands agent + Gradio UI | ✅ works now (buttons always; chat needs an LLM backend) |
| **cuRobo** collision-aware planning | ✅ **installs + runs the full pick-and-place on driver 550 (validated, #67 T3/T4/T5)**; `--planner curobo --curobo-urdf <so101.urdf>`. Loads the **same URDF** into MuJoCo + cuRobo (aligned joint conventions + EE frame), position-only IK (5-DOF), kinematic execution + a **kinematic grasp-attach** that transports the cube to the bin (validated **success_rate ≈ 0.3-0.4** over episodes (varies with cuRobo plan nondeterminism)). |
| **Isaac Sim** backend (`--backend isaac`) | ⛔ falls back to MuJoCo until the `strands-robots[sim-isaac]` runtime is installed (`create_simulation("isaac")`) |

Missing cuRobo / Isaac / lerobot / LLM each disable only their own feature with
an actionable message — the app still loads and the loop is still demonstrable.

## Quick start

```bash
# from the repo root
# Real SO-101 in MuJoCo + the LeRobot dataset writer used by the collector.
pip install "strands-robots[sim-mujoco,lerobot]"
# The natural-language agent + interactive UI (app.py) additionally need:
pip install "strands-agents>=1.13.0,<2.0.0" "gradio>=4,<7" numpy
# cuRobo and Isaac Sim are optional, out-of-band installs (see "Flipping to
# Isaac + cuRobo" below).

# headless CI smoke (no GPU/GL): build -> scripted plan -> record -> reload
python -m examples.so101_curobo.smoke_test

# headless data generation (writes a LeRobot dataset to --root)
MUJOCO_GL=egl python -m examples.so101_curobo.app --smoke --episodes 5 \
    --root /tmp/so101_curobo_ds

# interactive Gradio app (camera preview + buttons + chat)
MUJOCO_GL=egl python -m examples.so101_curobo.app --server-port 7863
```

> **Re-running data generation?** A second run that reuses the same dataset
> directory currently fails with `FileExistsError` (LeRobot's `create()` does
> `mkdir(exist_ok=False)`, and the collector only clears a prior dir when
> `--root` is given). Until an overwrite/unique-path fix lands, on
> re-run either pass a **fresh `--root`**, use a **unique `--repo-id`**, or
> delete the existing dataset dir (the HF cache dir when no `--root` is set)
> first. The `--root` path above is cleared automatically on re-run.

Load a recorded dataset back (no Hub round-trip):

```python
from strands_robots.dataset_recorder import load_lerobot_episode
ds, start, length = load_lerobot_episode("local/so101_curobo_pickplace", 0, root="/tmp/so101_curobo_ds")
```

## Architecture

```
Strands Agent (Gradio UI)            agent.py / app.py
   │  natural language → tools
   ▼
SO101CuroboDemo                       controller.py
   ├─ sim     = make_sim("mujoco"|"isaac")            scene.py   (SimEngine)
   │            build_pick_place_scene(): SO-101 + cube + bin + cameras
   ├─ planner = make_planner()                        planner.py
   │            ScriptedPlanner | CuroboMotionPlanner → JointTrajectory
   └─ collector = LeRobotDataCollector                collector.py
                send_action(waypoint) + add_frame(obs, action) per step,
                save_episode() + finalize() → LeRobot v3.0 dataset; success check
```

The collector uses the tested `strands_robots.dataset_recorder.DatasetRecorder`
recipe: `create() → add_frame()* → save_episode() → finalize()`.

## Flipping to Isaac + cuRobo

- **Isaac:** `--backend isaac` calls `create_simulation("isaac", render_mode="rtx_realtime")`.
  Needs the Isaac Sim runtime (~30 GB, `pip install "strands-robots[sim-isaac]"`),
  plus a faithful SO-101 USD via `add_robot(usd_path=...)`. Falls back to MuJoCo otherwise.
- **cuRobo:** `--planner curobo` (+ optional `--curobo-urdf` / `SO101_URDF`).
  **SO-101 URDF resolution:** the URDF is resolved in this order — explicit
  `--curobo-urdf` → `SO101_URDF` env → the **auto-downloaded `strands-robots`
  SO-101 cache URDF** (`~/.strands_robots/assets/robotstudio_so101/`, the same
  asset the default MuJoCo demo fetches). So once you've run the MuJoCo demo
  (or on any box with internet), `--planner curobo` finds a URDF + meshes
  with **no flag needed**; pass `--curobo-urdf` only to override with your own.
  **Driver:** NVIDIA's docs recommend driver ≥ 580.65.06 for cuRobo's latest
  release, but this example is **validated on driver 550 / CUDA 12.4 / L4** —
  the 580 floor is conservative, since CUDA 12.x kernels run on a 12.4 driver.
  Treat **550** as the validated minimum here and **580** as NVIDIA's
  recommendation for the upstream latest release. This is the canonical install
  recipe:
  ```bash
  export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9
  python -m venv --system-site-packages .venv && source .venv/bin/activate
  pip install -U pip setuptools wheel ninja
  git clone --depth 1 https://github.com/NVlabs/curobo && cd curobo
  sed -i '/Topic :: Scientific\/Engineering :: Robotics/d' pyproject.toml  # newer setuptools rejects it
  pip install -e . --no-build-isolation
  pip install 'cuda-core[cu12]'   # the refactored cuRobo's runtime kernel backend (required)
  ```
  (`uv` users can substitute `uv venv --python 3.11` + `uv pip install -e .
  --no-build-isolation` for the venv/build steps; the `cuda-core[cu12]` step is
  still required either way.)
  `CuroboMotionPlanner` builds the SO-101 model from a URDF via the new
  `RobotBuilder` (T4, auto-derives the 5-DOF arm chain to `gripper_frame_link`)
  and chains `MotionPlanner.plan_pose` segments into the full pick-place (T5,
  validated end-to-end: a 434-waypoint collision-free reach→grasp→lift→place→
  release the MuJoCo arm executes, recorded as a LeRobot episode).
  **5-DOF handling:** the SO-101 has only 5 arm DOF, so a fully-constrained
  6-DOF pose goal is infeasible. The planner uses **position-only**
  tracking (`ToolPoseCriteria.track_position`, `position_only=True`), leaving
  orientation free so tabletop targets are reachable; the bin
  (`scene.DEFAULT_PLACE_POSITION`) is set within the arm's reach.
  **Top-down grasp (`top_down_grasp=True`, default):** strict vertical is
  infeasible on 5 DOF, so the pick segments (`reach`/`grasp`/`lift`) add a
  *soft* downward orientation bias (`ToolPoseCriteria.track_position_and_orientation`,
  `rpy` weight `top_down_weight≈0.05`) plus a relaxed success tolerance
  (`orientation_tolerance≈1.6 rad`) and keep the most-vertical of
  `top_down_attempts` solves (cuRobo is nondeterministic; best-of-N tames the
  variance). Validated through the planner (cube `[0.2,0.2]`, bin `[0.0,0.25]`):
  `reach 2.0°`, `grasp 10.9°`, `lift 6.2°` from straight-down vs ~84° (sideways)
  with free orientation. The `place` segments stay **position-only** (the bin
  pose is not vertical-reachable and a top-down drop is unnecessary). Any
  unreachable oriented solve falls back to position-only, so this never
  regresses below the position-only path.
  **Matched model (key):** cuRobo (URDF) and a MuJoCo `data_config` SO-101 use
  different joint conventions/zero-poses/EE frames, so cuRobo's plan executes
  *wrongly* on the data_config arm. The fix: when `--planner curobo` + a URDF
  are set, the sim loads the arm from the **same URDF** (`add_robot(urdf_path=...)`),
  so cuRobo's plan executes exactly (FK matches to mm). That URDF loads without
  position actuators, so the collector drives it **kinematically**
  (`set_joint_positions`, `kinematic=True`); the cube responds via stepped
  contact. Validated: the gripper reaches the cube pose precisely.
  **Grasp (success>0):** the actuator-less arm can't hold via friction, so the
  collector models the grasp with a **kinematic grasp-attach** (`grasp_attach`):
  when the gripper closes within `attach_radius` of the cube it attaches the
  cube to the gripper (zeroing its velocity to avoid teleport flings), carries
  it, and releases over the bin. This transports the cube and yields a real
  per-episode success label — **success_rate ≈ 0.3-0.4** (cuRobo drives the motion; the residual
  variance is plan nondeterminism -- misses land just outside the bin radius).
  It's a standard kinematic grasp for synthetic data; a fully *dynamic* grasp
  would need an actuated model + contact/gripper-geometry. Unreachable targets
  fall back to the scripted planner. Set the URDF so its meshes resolve for
  MuJoCo (e.g. the URDF next to its `assets/` dir); `SO101_ASSET` points cuRobo
  at the meshes.

## Component map

| Component | Where | State |
|---|---|---|
| Isaac backend selection | `scene.make_sim("isaac")` -> `create_simulation("isaac")` | in-tree `strands-robots[sim-isaac]`; falls back to MuJoCo when the runtime is absent |
| Faithful SO-101 asset | `add_robot(urdf_path=...)` (sim + cuRobo share the URDF) | ✅ for cuRobo path (same URDF both sides) |
| cuRobo install validation | `planner.CUROBO_INSTALL_HINT` | ✅ validated on driver 550 (recipe above) |
| cuRobo SO-101 config | `CuroboMotionPlanner._ensure` (`RobotBuilder`) | ✅ builds the 5-DOF model from URDF |
| `CuroboMotionPlanner` | `planner.py` | ✅ cuRobo drives the pick-and-place + kinematic grasp-attach transports the cube to the bin (validated success_rate ≈ 0.3-0.4; nondeterminism-limited); dynamic-grasp realism is further tuning |
| Executor + gripper | `collector._execute_and_record` | ✅ |
| `LeRobotDataCollector` | `collector.py` | ✅ (multi-episode, success check) |
| Domain randomization | `record_dataset(randomize=True)` → `sim.randomize` | ✅ basic |
| Agent + Gradio app | `agent.py`, `app.py` | ✅ |
| Docs + smoke test | this file, `smoke_test.py` | ✅ |

## Scope

The cuRobo path targets **rigid, quasi-static, prehensile** tasks (pick/place,
reach, push). Contact-rich / deformable / dynamic tasks should route to
teleop/RL, not this generator.
