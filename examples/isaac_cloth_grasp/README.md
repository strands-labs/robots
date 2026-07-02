# A sim-to-real cloth representation for grasping (issue #5)

Issue [#2](https://github.com/armwaheed/robots/issues/2)'s two-G1 bed-making demo holds the
cover with a **spring peel-off grip** — honest physics, but the grab tabs it needs are a
workaround for a simulator limitation, not a real effector. This package answers issue
[#5](https://github.com/armwaheed/robots/issues/5): **build and test a different cloth
representation** on which a real **frictional / enclosure grasp** holds, the way a Unitree G1's
Inspire/BrainCo hand actually grips a sheet.

![Newton grippable fabric — sensor-gated grip and headward drag](https://raw.githubusercontent.com/armwaheed/robots/bed-making-media/examples/isaac_cloth_grasp/media/newton_grippable_fabric_held_drag.gif)

*The discovery: on Newton's VBD FEM cloth, a **force-limited, sensor-gated** pinch grips the
hanging hem and drags the whole cover headward across the bed — 5.3 cm slip over a 45 cm stroke,
grip held — with no attachment, no grab tabs, no kinematic pinning. PhysX (the engine the issue
[#2](https://github.com/armwaheed/robots/issues/2) demo runs on) cannot do this with either of
its cloth models. Eye-verified; see [the discovery section](#the-discovery-newton-makes-fabric-grippable) below.*

Two candidate representations were built and tested on the DGX Spark (aarch64 / GB10), with the
same pinch-and-drag protocol so they compare apples-to-apples against the PhysX PBD particle
cloth baseline (whose frictional-grasp failure is NVIDIA-documented-unsolved — forum 332704,
IsaacLab #4291):

| | representation | engine | drape | frictional grasp |
|---|---|---|---|---|
| baseline | PBD particle cloth (thin membrane) | PhysX 5 (Isaac Sim 5.1) | ✅ (issue #2) | ❌ penetrate + slip (NVIDIA-documented; reproduced here incl. with the DexGarmentLab adhesion recipe — 3 configs, pinch closes on 30 particles, transmits nothing) |
| candidate 1 | **VBD FEM triangle cloth** | **Newton** (Warp / MuJoCo-Warp) | ✅ eye-verified | ✅ **holds a 45 cm drag, slip ≈ 1 cm** (kinematic-plate protocol, eye-verified); force-limited sensor-gated harness below |
| candidate 2 | **FEM surface deformable** (deformable beta, real `surface_thickness`) | PhysX 5 (Isaac Sim 5.1) | ✅ eye-verified | ❌ slips the full stroke (3 grab geometries incl. adaptive aim) — **corroborated by NVIDIA staff**: rigid↔surface-deformable collision "not fully supported" in 5.1 (forum 359023) and the deformable solver has **dynamic friction only** (no static friction → grasp creep by design) |

## Headline findings

1. **Newton runs on the GB10 from stock pip wheels** — the issue's "known gamble on the source
   build" did not materialize: `pip install newton` (1.2.1) + `warp-lang` (1.13) initialize CUDA
   on the Spark (sm_121) out of the box. No source build.
2. **A pure frictional pinch HOLDS on Newton's VBD FEM cloth.** `spikes/spike_newton_pinch_drag.py`
   drapes a bed-cover-sized sheet over a mattress box, pinches the hanging overhang between two
   high-friction plates and hauls it up over the edge and across the box top through a **full
   45 cm stroke: final slip 1.09 cm** — no attachment, no tabs, no springs, no kinematic pinning
   of the cloth. Eye-verified frame-by-frame (`spikes/out/newton_pinch_drag/`).
3. **Isaac Sim 5.1 quietly ships a second cloth representation in-engine**: the *deformable beta*
   (`/persistent/physics/enableDeformableBeta`) adds FEM **surface deformables** — cloth with a
   real `surface_thickness`, FEM elasticity (not PBD springs), self-collision and CCD. It
   simulates and drapes beautifully on the GB10 (`spikes/spike_surface_deformable.py`).
   * Gotcha (same family as the particle-cloth one): GPU FEM deformation **never syncs to
     USD or Fabric** — render by reading the tensor view
     (`create_surface_deformable_body_view(...).get_simulation_nodal_positions()`) and blitting
     into the sim mesh each frame.

## The grasp library (`newton_cloth/`)

* `bedsheet.py` — the bed-cover FEM cloth recipe (VBD, cm-scale regime of Newton's own cloth
  examples). Notes the **mass-scale gotcha**: the VBD contact constants are conditioned around
  the bundled examples' particle masses; force budgets are therefore expressed as **ratios to
  sheet weight** (a real Inspire two-finger pinch ≈ 20 N vs a real ~0.5 kg cover ≈ 4.9 N → 4.1×),
  which is the dimensionless quantity that decides hold-vs-peel.
* `gripper.py` — **force-limited pinch**: the finger plates are free *dynamic* bodies driven by
  clamped wrenches (pinch ≈ 1.5× sheet weight — only what friction needs to beat the drag load;
  carriage servo ≤ 8×). Nothing kinematic ever touches the cloth, so *holding* and *peeling under
  overload* both emerge from contact physics. (Newton's own cloth_franka example drives its robot
  velocity-kinematically — it never feels the cloth; this gripper does.) The hard-won lesson, in
  the code comments: a grasp needs pinch ≈ load/μ ≈ 1× the cover weight — commanding the Inspire's
  *max* spec (4.1× weight) onto a light free pad is a numerical impulse that launches it through
  the compliant contact, not a grasp.
* `sensing.py` — the issue-#2 short-range proximity sensor + grasp-decision state machine
  (bundled copy, repo convention), so targeting is **sensor-derived, never privileged state**:
  the wrist descends along a bed-geometry prior and the grab fires only when the proximity
  sensor reports cloth within centimetres — exactly the #2 robots' protocol.

## The acceptance harness (`harness_grip_drag_peel.py`)

The full issue-#5 success-criteria lifecycle in one run: SETTLE → SEEK_DOWN/SEEK_IN (sensor-gated
side-approach + slide-in) → GRIP (force-limited close) → RETRACT (clear the mattress edge) → LIFT
→ DRAG A (the full 45 cm headward stroke; PASS = cloth stays in the pinch) → DRAG B (the cover's
far edge is pinned mid-world — a snag — and the wrist keeps pulling; PASS = the cloth **peels
physically** out of the bounded-force fingers and the decision layer releases — requirement E
without a scripted break force).

**Current verdicts (eye-verified, run v21 — `media/newton_grippable_fabric_held_drag.gif`):**

| criterion | result |
|---|---|
| sensor-gated grip (no privileged state) | ✅ PASS — 21 particles pinched on the proximity reading |
| **drag the full 45 cm stroke without slipping** | ✅ **PASS — 5.27 cm slip, grip held** |
| stability (no NaN / blow-up) | ✅ PASS |
| peel under snag overload (req E) | ⚠️ open — the grip is now robust enough that the scripted snag didn't exceed its friction capacity, so the release never triggered (4.89 cm slip). The **physical peel was shown in earlier runs**; the test needs a stronger overload to exercise req E cleanly. Deferred to the next session. |

_Frames: `spikes/out/grip_drag_peel/`; encoded clips: `spikes/out/*.mp4`._

## The discovery: Newton makes fabric grippable

The headline result of issue #5 is a **representation swap, not a parameter tweak**. Issue #2's
bed-making demo runs on PhysX, where a sheet is a thin PBD particle membrane that rigid fingers
penetrate and slip off of — NVIDIA-documented-unsolved (forum 332704), which is why #2 needed the
spring peel-off grip + grab-tab workaround. Swapping the cloth to **Newton's VBD FEM
representation** makes the *same* frictional pinch simply hold: the fingers grip a bunched hem and
haul the whole cover headward, the grasp transmitting the drag load through honest contact.

| grip (LIFT, t≈10.7 s) | drag headward (DRAG, t≈13 s) |
| --- | --- |
| ![grip](https://raw.githubusercontent.com/armwaheed/robots/bed-making-media/examples/isaac_cloth_grasp/media/still_lift.webp) | ![drag](https://raw.githubusercontent.com/armwaheed/robots/bed-making-media/examples/isaac_cloth_grasp/media/still_drag.webp) |

The same run, as captured during review: [`media/screenshot_lift.png`](https://raw.githubusercontent.com/armwaheed/robots/bed-making-media/examples/isaac_cloth_grasp/media/screenshot_lift.webp),
[`media/screenshot_drag_b.png`](https://raw.githubusercontent.com/armwaheed/robots/bed-making-media/examples/isaac_cloth_grasp/media/screenshot_drag_b.webp). Animated:
[`media/newton_grippable_fabric_held_drag.gif`](https://raw.githubusercontent.com/armwaheed/robots/bed-making-media/examples/isaac_cloth_grasp/media/newton_grippable_fabric_held_drag.gif).

> **For the #2 writeup ([`RL_WHOLE_BODY_REACH.md`](../isaac_bed_making/RL_WHOLE_BODY_REACH.md) §7
> cloth-manipulation).** This discovery illustrates that section, but the cross-link/embed should be
> added there by the #2 work — this branch's copy of that file is stale relative to the concurrent
> spring-grip session's commits, so editing it from here would clobber that work. The GIF + stills
> in `media/` are ready to drop in. (See the isolation note at the top of this session's commit.)

## Run

```bash
# one-time: the Newton venv (stock wheels, aarch64-clean)
python3 -m venv .spikes/newton-venv
.spikes/newton-venv/bin/pip install newton warp-lang mujoco mujoco-warp torch numpy pillow matplotlib

# Newton spikes + harness
.spikes/newton-venv/bin/python examples/isaac_cloth_grasp/spikes/spike_newton_pinch_drag.py
.spikes/newton-venv/bin/python examples/isaac_cloth_grasp/harness_grip_drag_peel.py

# PhysX deformable-beta spikes (Isaac Lab python on the Spark)
cd ~/workspaces/git/IsaacLab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
./isaaclab.sh -p .../examples/isaac_cloth_grasp/spikes/spike_surface_deformable.py
./isaaclab.sh -p .../examples/isaac_cloth_grasp/spikes/spike_physx_pinch_drag.py
```

## Decision (issue #5 decision rule)

**The cloth representation on which a real frictional grasp holds is Newton's VBD FEM
cloth — and it is the only one on this stack.** Both PhysX representations fail the same
pinch-and-drag protocol for engine-level reasons NVIDIA documents (particle cloth:
forum 332704; FEM surface deformables: rigid contact not fully supported + dynamic-only
friction). Per the issue's decision rule:

* **Issue #2 keeps the spring peel-off grip** (the legitimate "grip-lock" abstraction, like
  MuJoCo's kinematic weld) — it is the honest in-PhysX approximation of the grasp this
  library demonstrates for real.
* **This library is the transfer-realism testbed**: the grasp itself is contact physics —
  force-limited fingers, sensor-gated targeting, physical peel under overload.
* **The integration path to #2 exists upstream**: Isaac Lab 3.0 beta (develop) has merged
  two-way Newton-VBD cloth coupling (isaac-sim/IsaacLab PR #5443, `Isaac-Lift-Cloth-Franka`),
  and Newton officially supports the DGX Spark (aarch64). When the #2 demo migrates to
  Isaac Lab/Newton, the spring grip is replaced by this library's frictional pinch directly.

## References

* NVIDIA forum 332704 — particle-cloth gripper penetrate/slip (the baseline failure)
* isaac-sim/IsaacLab#4291 — deformable↔articulation attachment unsupported
* Newton physics — github.com/newton-physics/newton (Warp + MuJoCo-Warp; announced future
  physics engine of Isaac Lab); bundled `example_cloth_franka.py` / `example_cloth_h1.py`
* Issue #2 `RL_WHOLE_BODY_REACH.md` §7 (FlingBot / SoftGym / "Bodies Uncovered" / Seita et al.)
