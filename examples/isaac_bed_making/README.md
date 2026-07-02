# Two Unitree G1s make a bed in NVIDIA Isaac Sim — coordinated over Arm Device Connect

Two **Unitree G1 humanoids** (with **Inspire 5‑finger hands**) **walk up to a bed and make it
together** in **NVIDIA Isaac Sim / Isaac Lab**, coordinating as **equal peers over Arm Device
Connect**. Each robot **balances on its own two feet the entire time** — it walks in, leans over the
bed, grips the sheet and draws it toward the head — with **no kinematic cheats** (no base pinning,
teleporting or joint freezing), so every motion is one a real G1 could reproduce on hardware.

Self‑contained: it does **not** modify the `strands_robots` product package or Arm's Device Connect,
and **bundles its own copy** of the equal‑peer swarm driver ([`swarm_driver.py`](swarm_driver.py)) — a
copy of the original from the [MuJoCo demo](../mujoco_bed_making/) — so it has no cross‑demo dependency.

![Two G1s make a bed](https://raw.githubusercontent.com/armwaheed/robots/bed-making-media/examples/isaac_bed_making/media/rl/bedmaking_scene.webp)

A full headless run renders an mp4 to **[`media/isaac_bed_making.mp4`](https://github.com/user-attachments/assets/28eb7850-6f58-427f-bf28-631423ca1fa9)**.

## Setup & Python version

This repo targets **Python 3.12+** (`strands_robots` pins `requires-python >=3.12`), while **NVIDIA
Isaac Sim 5.1 bundles Python 3.11**. Use a dedicated **virtual environment** to avoid the version
conflict:

- **Repo tooling, asset download, tests, and the MuJoCo/Newton examples** — a Python 3.12 venv:
  ```bash
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install -e "."
  ```
- **The Isaac Sim scene itself** runs under Isaac's own Python 3.11 via `<IsaacLab>/isaaclab.sh -p`.
  It imports `strands_robots` for asset resolution, so install that into Isaac's env with
  `pip install -e . --ignore-requires-python --no-deps`; note that the `strands-agents` chain does
  **not** fully resolve on 3.11 (one dep enforces 3.12 at build time). **Known limitation** until
  Isaac Sim ships Python 3.12 — keep the 3.12 venv for anything that imports `strands_robots`, and
  launch the scene with Isaac's interpreter:
  ```bash
  <IsaacLab>/isaaclab.sh -p examples/isaac_bed_making/demo.py
  ```

> **Benchmark (current):** both G1s walk in, hand off to the whole-body reach policy, lean/squat to the
> draped sheet, grip it and draw it headward — **each balancing on its own two feet through the entire
> demo, no topple, no kinematic cheats** (eye-verified, 195 frames). The two robots adopt *different*
> reach postures (one squats deep, one leans) because the policy solves the reach **within each robot's own
> actuation envelope — learned control, not scripted choreography.** Open work is **manipulation quality**:
> drawing the sheet up to the pillows (see [`RL_WHOLE_BODY_REACH.md` §8](RL_WHOLE_BODY_REACH.md#8-status--whats-next)).

## How it works, end to end

1. **Walk in.** Each G1 spawns ~1 m off its side of the bed and walks to the bedside, **arms at its
   sides**, under Unitree's official [`unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab)
   G1 velocity‑walk policy — a pretrained, Isaac‑Lab‑native whole‑body RL policy Unitree ships for real
   G1s. We reproduce its exact 480‑D observation, joint order and PD gains from the deploy config and run
   the MLP **on the GPU via torch** (the DGX Spark has no onnxruntime GPU provider). The full 9‑inch‑
   overhang sheet drapes during the walk.
2. **Hand off to a whole‑body bed‑reach RL policy.** One **ambidextrous** policy owns all 29 body
   joints, so it **balances on its own two feet while leaning over the bed to reach** — the loco‑
   manipulation skill a walking‑balance policy can't hold (the deep bend throws the centre of mass past
   the feet). Trained in Isaac Lab — full story in **[`RL_WHOLE_BODY_REACH.md`](RL_WHOLE_BODY_REACH.md)**.
3. **Grip + draw the sheet** headward with the hand on the target's side — left for one robot, right for
   its mirror, a natural same‑side motion — the policy balancing the whole body throughout.

Coordination is real Device Connect: both G1s pursue the goal *"the bed is made,"* claim work, emit
events, and ask for / offer help. With `--broker` both peers register live on the dashboard.

## The whole‑body bed‑reach RL policy &nbsp;→&nbsp; [`RL_WHOLE_BODY_REACH.md`](RL_WHOLE_BODY_REACH.md)

Bed‑making for a free‑standing humanoid is a **loco‑manipulation** problem: the deep bend over the bed
throws the centre of mass past the feet, so a walking‑balance policy topples on the reach. The fix is a
**single whole‑body RL policy** (legs + waist + arms; 29 joints; 50 Hz) that owns **balance *and* reach
at once** — trained on Isaac Lab's locomotion RL rails with a station‑keeping reward (reach by leaning,
stay planted), a bed‑obstacle constraint, and a FALCON‑style grip‑slip force load. It is
**ambidextrous**: each robot reaches with the hand on the target's side, so the two flanking robots both
pull the sheet headward as a natural same‑side motion rather than a cross‑body sweep. Free base, no
kinematic cheats — physically valid for sim‑to‑real. The deployable policy is committed at
[`rl/policy/`](rl/policy/); the method, its four iterations (to the two‑G1 benchmark), results, training
config and reproduce steps are in **[`RL_WHOLE_BODY_REACH.md`](RL_WHOLE_BODY_REACH.md)**.

| Reach over the bedside (ambidextrous) | Walk in, arms at the sides |
| --- | --- |
| ![ambidextrous reach](https://raw.githubusercontent.com/armwaheed/robots/bed-making-media/examples/isaac_bed_making/media/rl/ambidextrous_reach.webp) | ![walk in](https://raw.githubusercontent.com/armwaheed/robots/bed-making-media/examples/isaac_bed_making/media/rl/walk_in_arms_at_sides.webp) |

## Real‑to‑sim sensing with robotics‑connect &nbsp;→&nbsp; unitree/g1

A simulator doesn't fully expose a real robot's **sensor and effector envelope** — the G1's sensors are
tilted, range‑limited and occluded by its own body in ways Isaac Sim won't tell you. We close that gap
with **robotics‑connect**, Arm's
Unitree G1 EDU control stack, where each sensor is **characterized on the physical robot**:

* the head **Intel RealSense** is calibrated to a **51.29° downward tilt** — it sees the floor and the
  near bed, not the room, though a mattress edge is still resolvable 2–3 m out;
* the crown **Livox MID‑360** LiDAR's **near‑field fidelity** (a surface reads cleanly at 0.4 m) and its
  **self‑occlusions** (the robot's face‑frame blanks ±40–45° of azimuth, its chin blanks below −10°
  elevation).

[`perception.py`](perception.py) reproduces those exact characteristics in sim — a downward head camera
and an Isaac `RayCaster` LiDAR carrying the real blind spots — so a bed‑detector trained in sim works on
the hardware. That same on‑hardware ground truth also **cuts RL training cycles** by getting the rewards
and constraints right up front (e.g. the arms‑at‑sides neutral is taken from the real walking policy's
own default pose). See
[`RL_WHOLE_BODY_REACH.md` §6](RL_WHOLE_BODY_REACH.md#6-closing-the-real-to-sim-gap-with-robotics-connect).

## No kinematic cheats (why it's harder, and worth it)

The whole point of an Isaac Sim demo is **sim‑to‑real**: every motion must be something a real robot
could do. So this example refuses kinematic shortcuts — the robots are **free‑base articulations** that
stand, walk and reach **entirely under their controllers**. **No base pinning, no teleporting, no joint
freezing.** That makes balance‑while‑reach a genuine loco‑manipulation problem (the reason for the
whole‑body RL policy above) rather than a scripted animation.

## Real, physically‑simulated cloth

A PhysX particle‑cloth sheet drapes over the bed under gravity (the "MuJoCo recipe": featherlight +
coarse + thick) with a render‑side shell for visual thickness — not a scripted animation. It is built as
a **quad** grid (not triangles): Isaac's auto particle‑cloth turns every mesh edge into a stiff stretch
spring, so a triangulated grid locks into a rigid plate, while quads leave the diagonal as a soft shear
spring and it drapes.

**Rendering gotcha — cloth and robot together.** The robot articulation only renders its motion with
**Fabric on** (`use_fabric=True`), but PhysX does **not** sync particle‑cloth deformation to Fabric. Per
NVIDIA's forums, *mesh* updates do cross Fabric (point‑instancer ones don't), and our bedsheet is a
`UsdGeom.Mesh` — so the demo runs `use_fabric=True` and **blits the live cloth positions** (from a PhysX
tensor cloth‑view) **into the Fabric mesh points** each render. Both the moving robot and the deforming
sheet show at once.

## View the robots on the Device Connect dashboard

Run the demo with **`--broker`** and both G1s register live on the Arm Device Connect
dashboard, each with its callable functions, event stream, and identity:

![Device Connect dashboard: the bed-making G1 (beta-unitree-g1-humanoid-0, type unitree_g1_bed_making) online in the beta tenant with its callable RPCs, event stream, and Unitree G1 EDU identity](https://raw.githubusercontent.com/armwaheed/robots/bed-making-media/examples/isaac_bed_making/media/device_connect_dashboard.webp)

```bash
cd ~/workspaces/git/IsaacLab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
    ~/workspaces/git/robots/examples/isaac_bed_making/demo.py --broker --gui --render
```

The demo defaults to **`--loopback`** (an offline, in-process bus) which does **not** read
`.credentials/` and registers nothing, so the robots will **not** appear on the dashboard.
**If the robots didn't show up on an earlier run, that run used the default `--loopback`
instead of `--broker`** — only `--broker` uses the credentials.

`--broker` reads the two JWT credential files in `.credentials/`
(`beta-unitree-g1-humanoid-0.creds.json` and `beta-unitree-g1-humanoid-1.creds.json`) and
registers both G1s on the real Device Connect NATS fabric
(`nats://fabric.deviceconnect.dev:4222`; override with `--nats-url`). On the dashboard each
peer shows its live status and the shared goal *"the bed is made,"* its callable functions
(`pickUpBedSheet`, `walkToNextCorner`, `askForHelp`, `offerHelp`, `putDownBedSheet`,
`getStatus`, `getGoalState`, `listPeers`, `getEventHistory`, `getHelpHistory`,
`emergencyStopAll`), and the event/help stream that fills in as the peers claim corners and
trade help during the run.

**Timing for screenshots:** the peers are registered for the duration of the run (the
walk-in and bed-making sequence, ~1-2 min), then they unregister cleanly on exit, so take
screenshots while the demo is running. `--gui` opens the Isaac window too, so you can watch
the robots and the dashboard side by side. (There is intentionally no standalone
"register-and-hold" sidecar here — it would interfere with Device Connect's normal
lifecycle; the swarm is driven only by the running demo.)

## Run it

```bash
cd ~/workspaces/git/IsaacLab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"   # aarch64 caveat
PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
    ~/workspaces/git/robots/examples/isaac_bed_making/demo.py --loopback --render
```

| Flag | Effect |
| --- | --- |
| `--loopback` | Coordinate via an in‑process bus (offline, default). |
| `--broker` | Register both peers on the real Device Connect NATS fabric (`.credentials/`). |
| `--no-device-connect` | Skip Device Connect entirely. |
| `--gui` | Open the Isaac Sim window to watch live (instead of headless). |
| `--render` | Capture frames and encode an mp4 into `artifacts/isaac_bed_making/`. |
| `--no-walk` | Spawn at the bedside (skip the approach‑walk). |
| `--walk-only` | Stop after the approach‑walk (inspect locomotion). |
| `--pink` | Legacy A/B: velocity‑walk + **NVIDIA Pink‑IK** stand‑and‑reach instead of the RL policy. |
| `--replay` | Legacy: make the bed from a **recorded teleop trajectory** (`--replay-speed` / `--lag`). |

## Optional: real motion from teleoperation data (`--replay`)

With `--replay`, the waist + both arms are driven from a recorded trajectory of a *real* teleoperated
Unitree G1 making a bed — source:
[`unitreerobotics/G1_WBT_Brainco_Make_The_Bed`](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Make_The_Bed)
(LeRobot v3.0, Apache‑2.0). Because the sim uses the actual Unitree G1 model, the recorded joint angles
transfer 1:1. `tools/extract_trajectory.py` pulls one episode into the compact
`data/bed_making_traj.npz` the demo replays (regenerate with `pip install pyarrow huggingface_hub numpy`;
not needed just to run the demo).

## Files

| Path | Role |
| --- | --- |
| `demo.py` | Entry point: scene, walk‑in, whole‑body RL bed‑reach (default), Device Connect, render. Legacy `--pink` / `--replay` paths. |
| [`rl/`](rl/) | The whole‑body bed‑reach **RL package** (env, reward/force terms, training, eval, deployable policy) — see [`RL_WHOLE_BODY_REACH.md`](RL_WHOLE_BODY_REACH.md). |
| `perception.py` | robotics‑connect‑calibrated **LiDAR + head‑camera** sensing (real‑to‑sim). |
| `locomotion.py` | `VelocityWalker` (Unitree `unitree_rl_lab` velocity walk) + `BedReachPolicy` (our whole‑body reach policy) + floating‑base setup. |
| `cloth.py` | PhysX particle‑cloth bedsheet (quad grid) + tensor‑view read + Fabric blit + render‑side thickness shell. |
| `scene.py` | Scene geometry — two walk‑in G1s + bed + headboard + pillows + camera + sheet parameters. |
| `coordination.py` | In‑process Device Connect swarm (loopback / real broker), wrapping the shared swarm driver. |
| `behavior.py` | Per‑robot autonomous decision state machine (emergent ordering / help). |
| `manipulation.py` | (legacy `--pink`) `PinkArmIK` (NVIDIA Pink IK, arm + waist) + `apply_hand_friction`. |
| `replay.py` | (legacy `--replay`) Drives waist + arms (+ Inspire fingers) from the recorded dataset trajectory. |

## Status

**The full two‑G1 demo runs end‑to‑end (the benchmark).** The **walk‑in** (velocity policy, arms at
sides), the **walk→reach handoff**, the **whole‑body ambidextrous bed‑reach RL policy**, the
**full‑overhang sheet drape‑during‑walk**, the cloth physics + rendering, and the Device Connect swarm all
work together: **both robots balance on their own two feet through walk → reach → grip → pull, no topple**
(eye‑verified, 195 frames). The handoff that a warm‑start could only make *marginally* stable is fixed by a
**from‑scratch retrain** on the velocity‑walk's own arm neutral.

**Next — manipulation quality:** draw the sheet **up to (or over) the pillows** and relax the "made" goal
so a sheet corner pulled within a wider radius of a headboard‑end mattress corner (out to the pillow)
counts as placed; then sustained pull‑load robustness and the robotics‑connect‑calibrated LiDAR/RGB
*detect → approach → switch* behaviour layer. See the issue #2 continuation comments and
[`RL_WHOLE_BODY_REACH.md` §8](RL_WHOLE_BODY_REACH.md#8-status--whats-next).
