# MuJoCo + 3D Gaussian Splatting hybrid render — strands-robots example

A Python port of the [MuJoCo-GS-Web](https://vector-wangel.github.io/MuJoCo-GS-Web/)
browser demo, built on top of the upstream `strands_robots.simulation.Simulation`
AgentTool. Same idea — a MuJoCo physics scene rendered against a photoreal
3DGS background, with proper depth-aware occlusion — but driven from Python
through a [Strands](https://github.com/strands-agents/sdk-python) agent and
shown live in a Gradio UI.

## Gallery

The default scene: an SO-arm stood up in its `home` ready-pose on a kitchen
benchtop, composited against the **`tabletop`** 3D Gaussian Splatting scene
(from [MuJoCo-GS-Web](https://vector-wangel.github.io/MuJoCo-GS-Web/)) with
per-pixel, depth-aware occlusion.

![SO-arm erected on a 3DGS kitchen benchtop — oblique hero view](assets/hero_oblique.jpg)

![The arm waving on the 3DGS kitchen benchtop](assets/arm_wave.gif)

*The arm waving on the benchtop — the MuJoCo foreground composited
frame-by-frame over the `gsplat`-rasterised kitchen via `HybridCompositor`
(the same path the live view uses).*

| Front | Top-down |
|:---:|:---:|
| ![Front view of the arm on the benchtop](assets/hero_front.jpg) | ![Top-down view showing the arm seated on the counter](assets/hero_topdown.jpg) |

*The arm (and a small red cube on the bench) are the MuJoCo foreground; the
kitchen is a `gsplat`-rasterised background. `HybridCompositor` z-composites
the two so the arm correctly **rests on** the photoreal counter — see the
top-down view, where the base sits on the benchtop rather than floating.
Still frames from the live render pipeline.*

```
   +----------------------------+ +-------------------------------------+
   |  Live composite (RGB)      | |  Strands Agent chat                 |
   |  (MuJoCo + 3DGS / pano)    | |                                     |
   |                            | |  user > make the arm wave           |
   |                            | |  agent > done — showing front view  |
   |  [Preview camera ▼]        | |                                     |
   |  [Background ▼]            | |  user > switch to topdown           |
   |  [Render now] [Reset]      | |                                     |
   +----------------------------+ +-------------------------------------+
```

The example works on day 0 with **zero ML deps** (procedural kitchen
panorama as the background). Drop in a `.ply` and `pip install gsplat` to
upgrade to a real 3DGS scene.

## How it relates to MuJoCo-GS-Web

| Aspect | MuJoCo-GS-Web | This example |
|---|---|---|
| Physics | `mujoco_wasm` (browser) | `strands_robots.simulation.Simulation` (Python, MuJoCo) |
| Background renderer | `@sparkjsdev/spark` (3DGS, Three.js) | `PanoramaBackground` (procedural, default) or `GsplatBackground` (`gsplat`) |
| Composite | three.js depth pass | `HybridCompositor` — per-pixel z-compare in numpy |
| Driving the scene | Keyboard teleop / IK / ONNX RL policies | Strands agent + natural language ⇄ `Simulation` AgentTool actions |
| GS scene format | `.spz` | `.spz` (Niantic v2/v3) or `.ply` |
| Where it runs | Any browser, any device | Any host that can run MuJoCo offscreen rendering |

## Install

```bash
# Minimum (procedural panorama background, no GPU/3DGS):
pip install "strands-robots[sim-mujoco]" strands-agents gradio numpy Pillow

# Or, with optional real 3DGS rendering (CUDA required), add:
pip install "strands-robots[sim-gs]"   # gsplat + plyfile + torch (CUDA GPU)
```

`strands-robots[sim-mujoco]` brings in `mujoco`, `numpy`, etc. On a headless
Linux box you'll usually want `MUJOCO_GL=egl` (set automatically by `app.py`)
or `MUJOCO_GL=osmesa` if EGL isn't available.

The `gsplat` line is *additive*: install it on top of the base line to opt
into the real 3D Gaussian Splatting background. Without it the example still
runs end-to-end using the procedural panorama backdrop.

## Run

```bash
# From repo root:
python -m examples.mujoco_gs.app

# Or with a real panorama image:
python -m examples.mujoco_gs.app --panorama /path/to/kitchen_4k.jpg

# Or with a real 3DGS scene (requires the sim-gs extra):
python -m examples.mujoco_gs.app --gsplat-ply /path/to/scene.ply

# Pick a specific Strands model (optional):
python -m examples.mujoco_gs.app --model anthropic.claude-sonnet-4
```

Then open http://127.0.0.1:7860 in a browser.

### Watching the arm move

The agent drives motion through the **real `Simulation` API only** — no custom
tools. A request like *"have the arm wave"* makes the agent call
`run_policy(robot_name="arm", policy_provider="mock", duration=4.0,
control_frequency=20.0)` — the genuine strands-robots policy engine, which
steps the arm in real time. You watch it in the **live MJPEG view** (top-left);
the still preview shows the composited result afterwards.

> The 3DGS / panorama compositing is the example's *display* layer (the live
> view + still preview render through `HybridCompositor`); it is **not** an
> agent tool. The agent only ever calls real `Simulation` actions.
>
> Because the policy is `mock`, the arm performs *exploratory* motion (it moves
> and sweeps its joints) rather than a trained skill — that's what the stock
> API produces without a trained policy. For a real trained policy driving a
> real task, see the GR00T + LIBERO demo below.

### Try these prompts

* *“Have the arm wave.”* / *“Do a demo.”* — agent calls
  `run_policy(policy_provider="mock", duration=4.0, …)`; watch it in the live
  view.
* *“Render the front view.”* — agent calls `render(camera_name="front")`.
* *“Move the cube 10 cm to the left and render the topdown view.”* —
  agent uses `move_object` then `render`.
* *“Apply a 5 N upward force to the cube and render.”* —
  `apply_force` + `step` + `render`.
* *“Pose the elbow at 1.2 rad.”* — `set_joint_positions` + `step`.

## Real GR00T policy (Panda + LIBERO) — separate, agentic demo

The scripted wave is great for showing the rendering pipeline, but you can
also hand control to a **real NVIDIA GR00T vision-language-action policy**
driving a **Franka Panda** through a **LIBERO** manipulation task. This is a
**separate demo** — the SO-101 hybrid-render app above is untouched by it.

It's **agentic**: a Strands `Agent` is given the `Simulation` tool and a
natural-language instruction, and it picks `evaluate_benchmark` off the tool's
action surface, fills the kwargs, runs the eval, and reports the
`success_rate` in plain language (the pattern from
`examples/mujoco_gs/agent.py`). A background thread renders the scene
through the `HybridCompositor` into the `/live` MJPEG buffer so you watch the
arm in near-real-time, and a clip is recorded.

**Standalone Gradio app** — `app_groot_libero.py` (its own UI on port 7861, so
it runs alongside the SO-101 app on 7860):

```bash
python -m examples.mujoco_gs.app_groot_libero --groot-port 8000
# open http://127.0.0.1:7861
```

Pick a task, press **Run GR00T policy**, and the agent runs it — live view +
clip + success rate.

**Headless script** — `libero_groot.py` (run one episode → MP4):

```bash
# Needs a GR00T inference server reachable over ZMQ + libero + robosuite.
python -m examples.mujoco_gs.libero_groot --suite libero_10 --task 0 --port 8000

# Validate the pipeline without a policy server:
python -m examples.mujoco_gs.libero_groot --provider mock
```

### Getting a *successful* episode (verified recipe)

Measured **`success_rate=1.00`** against `nvidia/GR00T-N1.7-LIBERO`
(`libero_10` checkpoint) on
`libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_…`. The recipe:

* **Match the task suite to the served checkpoint.** The bundled container
  serves `/data/checkpoints/libero_10`, so run **`libero_10`** tasks. A
  different suite against it is a skill mismatch → ~0% success. (Bring up a
  suite-matched checkpoint with `gr00t_inference(action="lifecycle", …,
  hf_subfolder=<suite>)`.)
* **Don't cap `max_steps`.** LIBERO-Long needs ~500 steps; capping truncates
  the episode before completion. The runner/script use the adapter default.
* **Pre-warm the scene** (generate BDDL scene → `load_scene` → `prewarm`) so
  the `image`/`wrist_image` cameras + Panda exist before inference.
* **Let `evaluate_benchmark` auto-pick the robot** (omit `robot_name`) and use
  its **default `action_horizon`**.

Other facts / caveats:

* **GR00T is embodiment-locked.** This checkpoint is `LIBERO_PANDA` — it can't
  drive the SO-101 wave scene (wrong robot/cameras/action space).
* **ZMQ only.** strands-robots' GR00T client speaks ZMQ (not HTTP). Point
  `--port` at the ZMQ server (NVIDIA's `gr00t.eval.run_gr00t_server`).
* **N1.7 wire format.** N1.7 servers expect a time axis on observations; the
  runner passes `groot_version="n1.7"`. Use `n1.5`/`n1.6` for older servers.
* **`libero` + `robosuite` required** (they ship the BDDL tasks + scenes).
* **The panorama backdrop is mostly hidden for LIBERO.** LIBERO scenes are
  enclosed (table/walls/floor), so there's little "sky" for the GS/panorama to
  show through; the GS backdrop shines on open scenes like the SO-101 cube
  demo. The compositor is still used (it also applies the LIBERO `viz_option`
  that hides collision-geom/site debug markers, so the arm renders clean).

## Architecture

```
   ┌───────────────────────────────┐
   │  app.py — Gradio chat + live  │
   │           preview UI          │
   └────────────┬──────────────────┘
                │ user msg
                ▼
   ┌───────────────────────────────┐    ┌─────────────────────────────┐
   │  agent.py — MujocoGsAgent     │───▶│  Strands Agent              │
   │            (chat history)     │    │  - Simulation tool          │
    └────────────┬──────────────────┘    │  - Simulation tool (only)   │
                │                       └────────────┬────────────────┘
                │                                    │ tool calls
                │ render_now()                       ▼
                ▼                       ┌─────────────────────────────┐
   ┌───────────────────────────────┐    │ strands_robots.simulation   │
   │  compositor.py                │◀──▶│ Simulation (MuJoCo backend) │
   │  HybridCompositor             │    │  - create_world / add_robot │
   │  - per-pixel z-compare        │    │  - step / set_joint_pos     │
   │  - feathered seam             │    │  - render / render_depth    │
   └────────────┬──────────────────┘    └─────────────────────────────┘
                │
                ▼
   ┌───────────────────────────────┐
   │  backgrounds.py               │
   │  - PanoramaBackground (def.)  │
   │  - GsplatBackground (extra)   │
   └───────────────────────────────┘
```

* **`camera_utils.py`** — pulls the pinhole `K`, world-from-camera pose, and
  metric depth from MuJoCo's internal state (intrinsics aren't exposed by
  the AgentTool surface, so we reach through `sim.mj_model` / `sim.mj_data`).
* **`backgrounds.py`** — `BackgroundRenderer` protocol and the two
  implementations. Hot-swappable from the Gradio UI.
* **`compositor.py`** — depth-aware composite with optional edge feathering.
* **`scene.py`** — default arm + red cube + cameras setup. The arm is the
  SO-101 when its MuJoCo asset resolves, otherwise it auto-falls back to the
  SO-100 (identical 6-DoF kinematics) and then the Franka Panda, and verifies
  `add_robot` actually succeeded — so the agent never has to "repair" an empty
  scene. The robot that loaded is reported in the build summary and reflected
  in the agent's system prompt.
* **`agent.py`** — wires the real `Simulation` AgentTool (only) into a Strands
  agent; "wave" → `run_policy`. The `HybridCompositor` is the display layer,
  not an agent tool.
* **`app.py`** — Gradio UI: chat panel + live preview + scene controls +
  background switcher.

## Bringing your own GS scene

This example's GS loader reads both `.spz` (Niantic's binary format, v2/v3 —
the format of the built-in `tabletop` preset) and `.ply`. The Gradio
*Background* upload widget currently accepts `.ply` only, so to bring your own
capture, export a `.ply`:

| Source                | How to get a `.ply`                                        |
|---|---|
| **Nerfstudio**        | `ns-export gaussian-splat --load-config <run>/config.yml --output-dir <out>` |
| **Polycam**           | "Export → Gaussian Splat (PLY)" in the web UI               |
| **gsplat training**   | `model.export(<path>)` after training                       |
| **World Labs Marble** | "Export → Splats → PLY" (also gives `.spz`; pick PLY)       |

Then either pass it via CLI:

```bash
python -m examples.mujoco_gs.app --gsplat-ply path/to/scene.ply
```

…or upload it through the Gradio UI's *Background* panel.

### Aligning the GS scene to MuJoCo's world frame

3DGS scenes are usually in an arbitrary capture frame. To line one up with
the SO-101 + cube setup, pass a 4×4 SE(3) matrix:

```python
from examples.mujoco_gs import GsplatBackground
import numpy as np

# Example: rotate 180° around Z, lift 1.0 m up.
T = np.array([
    [-1, 0, 0, 0],
    [ 0,-1, 0, 0],
    [ 0, 0, 1, 1.0],
    [ 0, 0, 0, 1],
], dtype=np.float64)

bg = GsplatBackground(ply_path="kitchen.ply", transform=T)
```

The MuJoCo-GS-Web README's tip — *“add boxes in Marble's studio and feed
the bounding-box info to AI to generate a `collision.xml`”* — applies here
too: build a small MJCF with `<geom type="box">` collision proxies for the
walls and counters, then load it via `Simulation.load_scene(...)` before
`build_default_scene` adds the robot.

## Limitations vs. MuJoCo-GS-Web

* **GS *uploads* are `.ply`-only** — the loader itself also reads `.spz` (the
  built-in presets use it), but the Gradio upload widget filters to `.ply`.
* **No spherical-harmonics view-dependent color** for the GS background —
  we use the DC term only, which is fine for backdrop rendering but loses
  some specular fidelity vs. sparkjs.
* **No live keyboard teleop** — driving is via the agent or by hand-coded
  `Simulation` calls. (Agent + voice/text is the demo's selling point.)
* **No real-time RL policy on Unitree G1** — the example ships SO-101 by
  default; swap `data_config="so101"` → `"unitree_g1"` in `scene.py` and
  point `run_policy` at a real ONNX checkpoint to recreate that part.

These are all deliberate scope cuts to keep this an *example* rather than a
full feature. PRs welcome — swap in `gsplat`'s SH evaluation or wire up
the Isaac Sim backend for the heavier parallel cases.

## License

Apache-2.0, same as the rest of `strands-robots`.
