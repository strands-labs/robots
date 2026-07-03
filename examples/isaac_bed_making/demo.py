"""Two Unitree G1 humanoids make a bed in NVIDIA Isaac Sim, coordinating as equal
peers over Arm Device Connect.

End to end, each G1:

1. **Walks in** ~1 m to its side of the bed under Unitree's **official
   ``unitree_rl_lab`` G1 velocity-walk policy** (a pretrained, Isaac-Lab-native
   whole-body RL policy), arms at its sides, while the sheet drapes. The policy MLP
   runs on the GPU via torch.
2. **Hands off** to our **whole-body bed-reach RL policy** (``rl/``): one ambidextrous
   policy owns all 29 joints, so it **balances on its own two feet while leaning over
   the bed to reach** — the loco-manipulation skill a walking-balance policy can't hold.
3. **Grips + draws the sheet** headward with the hand on the target's side (left for
   one robot, right for its mirror — a natural same-side motion), the policy balancing
   the whole body throughout.

**Physical validity (sim-to-real).** Nothing is kinematically faked: the robots are
free-base articulations that stand, walk and reach entirely under their controllers,
so every motion is one a real G1 could reproduce on hardware — **no base pinning, no
teleporting, no joint freezing**. The bed-reach policy is documented in
``RL_WHOLE_BODY_REACH.md``. (The legacy velocity-walk + Pink-IK / dataset-replay paths
remain under ``--pink`` / ``--replay``.)

The bed has a **headboard + pillows** (static; never touched) and a
**particle-cloth sheet** that starts **flat, gathered toward the foot** (head half
bare), full bed width + a 9-inch overhang on each side.

Coordination is real Device Connect: each robot claims work, emits events, and
asks for / offers help. With ``--broker`` both peers register live on the
dashboard.

Run with the Isaac Lab Python on the DGX Spark::

    cd ~/workspaces/git/IsaacLab
    export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
    ./isaaclab.sh -p ~/workspaces/git/robots/examples/isaac_bed_making/demo.py --loopback --render

Rendering note: the demo runs with ``use_fabric=True`` so the robot articulations
render their true motion (Isaac Lab only pushes link poses to the renderer when
Fabric is on). PhysX does not auto-sync particle-cloth deformation to Fabric, so we
read the live cloth positions from a PhysX tensor cloth-view and write them into the
Fabric mesh points each render (mesh updates propagate through Fabric — see cloth.py).

Self-contained: does not modify ``strands_robots`` or Arm's Device Connect; bundles its
own copy of the equal-peer swarm driver (``swarm_driver.py``) — no cross-demo dependency.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--loopback", action="store_true", help="Coordinate via in-process bus (offline, default).")
    mode.add_argument("--broker", action="store_true", help="Register both peers on the real Device Connect NATS fabric.")
    p.add_argument("--no-device-connect", action="store_true", help="Skip Device Connect entirely.")
    p.add_argument("--gui", action="store_true",
                   help="Open the Isaac Sim window to watch live (instead of headless).")
    p.add_argument("--render", action="store_true", help="Capture frames and encode an mp4.")
    p.add_argument("--frames-dir", default=str(REPO_ROOT / "artifacts" / "isaac_bed_making"))
    p.add_argument("--nats-url", default=os.environ.get("DEVICE_CONNECT_NATS_URL", "nats://fabric.deviceconnect.dev:4222"))
    p.add_argument("--max-seconds", type=float, default=120.0, help="Safety cap on sim wall-time.")
    p.add_argument("--no-walk", action="store_true", help="Skip the learned approach-walk (spawn at the bedside).")
    p.add_argument("--walk-only", action="store_true", help="Stop after the approach-walk (debug the locomotion).")
    p.add_argument("--replay", action="store_true",
                   help="Use the open-loop dataset arm replay instead of closed-loop corner manipulation.")
    p.add_argument("--replay-speed", type=float, default=1.0, help="Playback speed of the recorded arm motion.")
    p.add_argument("--lag", type=float, default=0.4, help="Seconds robot 1 trails robot 0 in the replay.")
    p.add_argument("--pink", action="store_true",
                   help="Use the legacy Pink-IK stand-and-reach for PHASE 2 (legs balanced by the "
                        "velocity policy, arms+waist by Pink IK) instead of the default whole-body RL "
                        "reach policy. Kept for A/B; topples on a deep reach.")
    return p.parse_known_args()[0]


ARGS = parse_args()

# The default path: the whole-body bed-reach RL policy drives the manipulation. It owns the
# legs (balance) AND the arms+waist (reach), and the robot starts standing with its hands at
# its SIDES — the policy's own neutral pose — so there is no velocity-walker handoff and no
# scripted arm motion to topple it. The legacy velocity-walk + Pink-IK / dataset-replay paths
# (--pink / --replay) still use the walker.
RL_PATH = not (ARGS.pink or ARGS.replay)

# Import eigenpy + pinocchio BEFORE the Isaac app launches so eigenpy registers its
# StdVec_StdString converter first. Launching the app afterwards otherwise shadows
# that registration and Pinocchio's ``model.names.tolist()`` throws inside the Pink
# IK controller (manipulation.PinkArmIK). Harmless if pinocchio isn't installed.
try:
    import eigenpy  # noqa: F401
    import pinocchio  # noqa: F401
except Exception:
    pass

# ── Launch the simulator first (Isaac Lab requires this before other imports) ──
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": not ARGS.gui, "enable_cameras": bool(ARGS.render or ARGS.gui)})
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # noqa: E402

from examples.isaac_bed_making import cloth as clothmod  # noqa: E402
from examples.isaac_bed_making import coverage as coveragemod  # noqa: E402
from examples.isaac_bed_making import grasp as graspmod  # noqa: E402
from examples.isaac_bed_making import locomotion as locomod  # noqa: E402
from examples.isaac_bed_making import scene as scenemod  # noqa: E402
from examples.isaac_bed_making.manipulation import FingerGrip, PinkArmIK, apply_hand_friction  # noqa: E402
from examples.isaac_bed_making.replay import TrajectoryReplay  # noqa: E402

SIM_DT = 0.002            # 500 Hz physics
DECIMATION = 10           # locomotion policy/control runs every 10 sim steps = 50 Hz
                          # (matches the velocity policy's step_dt = 0.02 s)


def main() -> int:
    frames_dir = Path(ARGS.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    def mark(msg):
        print(f"[setup] {msg}", flush=True)

    mark("building scene cfg (Inspire 5-finger G1s, headboard + pillows)")
    # The G1s spawn ~1 m off their side of the bed and WALK in under the velocity policy (hands at
    # their sides), then hand off to the whole-body reach policy at the bedside. The walk-in doubles
    # as the sheet's settle time — by the time they arrive the side overhang has draped down off the
    # bed edges, clearing their arm space. ``--no-walk`` spawns them at the bedside mark instead.
    SceneCfg = scenemod.build_scene_cfg(G1_INSPIRE_FTP_CFG, spawn_at_manip=ARGS.no_walk)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0", use_fabric=True))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=8.0))
    mark("scene built")

    pre_stage = omni.usd.get_context().get_stage()
    # (1) Turn each Inspire G1 into a FLOATING-base articulation so the policy can
    # walk it (the USD ships fixed-base; see locomotion.make_floating_base), and
    # (2) rubberize the hands so they grip the cloth by friction. Both edit the
    # stage and MUST happen before sim.reset().
    for i in (0, 1):
        rp = f"/World/envs/env_0/Robot_{i}"
        locomod.make_floating_base(pre_stage, rp)
    # HIGH hand friction on BOTH hands: the grip is friction-only (no kinematic
    # attachment) — closed fingers cage a handful of cloth and the rubberized,
    # high-friction palms/fingertips hold it so the robot can actually drag the sheet.
    nb = [apply_hand_friction(pre_stage, f"/World/envs/env_0/Robot_{i}", sd,
                              static_friction=10.0, dynamic_friction=10.0)
          for i in (0, 1) for sd in ("right", "left")]
    mark(f"floating base set; hand friction bound to {sum(len(x) for x in nb)} colliders")

    cam = None
    if ARGS.render or ARGS.gui:
        from isaaclab.sensors.camera import Camera, CameraCfg
        cam = Camera(cfg=CameraCfg(
            prim_path="/World/CameraSensor", update_period=0,
            height=scenemod.CAM_RES[1], width=scenemod.CAM_RES[0], data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, focus_distance=400.0,
                                             horizontal_aperture=20.955, clipping_range=(0.05, 1.0e5))))

    mark("calling sim.reset()")
    sim.reset()
    stage = omni.usd.get_context().get_stage()
    # Round the pillows: hide the square box visuals and draw a rounded superellipsoid
    # pillow in each one's place (the box stays as a hidden, rounded collider so the
    # cloth still drapes over it). Visual-only meshes, so post-reset authoring is fine.
    from pxr import UsdGeom  # noqa: E402

    from examples.isaac_bed_making import props as propsmod  # noqa: E402
    for side, path in (("left", "/World/PillowL"), ("right", "/World/PillowR")):
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
        propsmod.superellipsoid_mesh(stage, f"/World/PillowVis_{side}",
                                     size=scenemod.PILLOW_SIZE, center=scenemod.PILLOWS[side],
                                     color=(0.95, 0.95, 0.97), roundness=scenemod.PILLOW_ROUNDNESS,
                                     rot_y_deg=scenemod.PILLOW_PROP_DEG)
    robots = [scene["robot_0"], scene["robot_1"]]
    sim_dt = sim.get_physics_dt()
    if cam is not None:
        cam.set_world_poses_from_view(torch.tensor([scenemod.CAM_EYE], device=sim.device),
                                      torch.tensor([scenemod.CAM_TARGET], device=sim.device))

    # ── learned locomotion: Unitree's OFFICIAL unitree_rl_lab velocity-walk policy ──
    # One whole-body (29-joint) velocity-tracking policy per robot serves BOTH the
    # settle-stand (zero command → balances in place) and the approach-walk (tracks a
    # velocity command), so there's no flaky walk→stand handoff between two policies. The
    # MLP runs on the GPU via torch (the Spark has no onnxruntime GPU provider — see
    # locomotion.VelocityWalker / the project's no-CPU-inference rule).
    walk = {i: locomod.VelocityWalker(robots[i], sim.device, SIM_DT * DECIMATION) for i in (0, 1)}
    stance = walk          # the same policy balances in place at zero command
    locos = walk
    # Seat each G1 in the velocity-walk policy's default pose (bent elbows etc.) so its first
    # observation is consistent — ONLY for the legacy walk paths. The RL path deliberately
    # leaves the arms at the spawn default (hands at the SIDES, the bed-reach policy's neutral
    # pose): seating the bent-elbow walking pose and handing that to the reach policy makes it
    # emit a destabilizing action and topples it (verified by eye). Hands at sides → clean start.
    if not RL_PATH:
        for i in (0, 1):
            dpos, dids = walk[i].default_pose()
            robots[i].write_joint_state_to_sim(dpos, torch.zeros_like(dpos), joint_ids=dids)
    mark("locomotion ready (unitree_rl_lab velocity walk-in + stand, GPU/torch)")

    # ── bedsheet: flat, gathered toward the foot (head half of the bed bare) ──
    scene_path = clothmod.find_physics_scene_path(stage)
    clothmod.enable_gpu_dynamics(stage, scene_path)
    sheet = clothmod.build_bedsheet(
        stage, scene_path, "/World/Sheet",
        size=scenemod.SHEET_SIZE, resolution=scenemod.SHEET_RES,
        origin=scenemod.SHEET_ORIGIN, color=(0.86, 0.86, 0.92),
        thickness=scenemod.SHEET_THICKNESS, accordion=True,
        accordion_start=scenemod.SHEET_ACCORDION_START,
        accordion_gather=scenemod.SHEET_ACCORDION_GATHER,
        accordion_waves=scenemod.SHEET_ACCORDION_WAVES,
        accordion_amp=scenemod.SHEET_ACCORDION_AMP,
        particle_mass=scenemod.SHEET_PARTICLE_MASS,
        damping=scenemod.SHEET_DAMPING)   # strong damping so the accordion settles instead of jiggling
    # Give the sheet VISUAL THICKNESS: the particle cloth is a single-layer membrane
    # (renders thin), so we drive a separate closed double-layer slab from the same
    # live particle positions each frame and hide the membrane (physics unchanged).
    shell = clothmod.build_shell_mesh(
        stage, sheet, thickness=scenemod.SHEET_SHELL_THICKNESS, color=(0.86, 0.86, 0.92))
    mark("bedsheet built (flat head edge + accordion ruffle at the foot; thick visual shell)")
    # Sew rigid GRAB TABS around the cover's whole perimeter NOW — before the first physics step — so
    # the new rigid bodies are part of the initial state (creating them mid-run detonates the particle
    # solver, head_v9). A deep grippable band on every edge gives tolerance: the policy's grab point
    # isn't predictable, so wherever a hand closes near an edge there are rigid tabs to hold.
    if RL_PATH:
        def _tabf(name, d):
            v = os.environ.get(name)
            return float(v) if v not in (None, "") else d
        tabs = clothmod.add_perimeter_grab_tabs(
            stage, sheet,
            depth=_tabf("BEDDEMO_TAB_DEPTH", 0.33),     # grippable band depth from each edge
            stride_m=_tabf("BEDDEMO_TAB_STRIDE", 0.35),  # ~21 tabs: the eye-verified-stable head_v12 density.
            # Do NOT pack this toward ~0.28 (~39 tabs): the high tab-count × featherlight-mass combination is
            # numerically unstable — a tab is flung and the particle solver detonates (the cover wads up and
            # the tabs scatter). Verified in ~/.isaac/tab_probe.py; heavier tab mass also fixes it independently.
            mass=_tabf("BEDDEMO_TAB_MASS", 0.001),       # ~cloth-particle mass: heavier tabs settle (don't buzz) + are hidden now
            robot_paths=[f"/World/envs/env_0/Robot_{i}" for i in (0, 1)])  # filter tabs from the robots (no plow-topple)
        mark(f"sewed {len(tabs)} rigid grab tabs along the head edge "
             f"(cloth↔rigid attachment; the hand holds them via the spring peel-off grip)")

    # ── manipulation: TWO-handed Pink IK + finger grips (or optional dataset replay) ──
    # Each G1 drives both arms (the right also owns the 3-DOF waist so it leans into the
    # reach; the left is waist-free so they don't fight) plus a FingerGrip per hand to
    # close the Inspire fingers on a handful of cloth. Two hands gather + pin the cloth,
    # one hand grips it and pulls — a friction grasp, no kinematic attachment.
    manip = {
        "R": {i: PinkArmIK(robots[i], "right", sim.device, SIM_DT * DECIMATION, with_waist=True) for i in (0, 1)},
        "L": {i: PinkArmIK(robots[i], "left", sim.device, SIM_DT * DECIMATION, with_waist=False) for i in (0, 1)},
        "gR": {i: FingerGrip(robots[i], "right", sim.device) for i in (0, 1)},
        "gL": {i: FingerGrip(robots[i], "left", sim.device) for i in (0, 1)},
    } if ARGS.pink else {}
    reps = {i: TrajectoryReplay(robots[i], sim.device) for i in (0, 1)} if ARGS.replay else {}

    # ── whole-body bed-reach RL policy (the default PHASE 2 path) ──
    # One policy per robot owns ALL 29 body joints, so it BALANCES on its own feet while
    # leaning + squatting to reach a commanded hand target — the loco-manipulation skill the
    # walking-balance policy can't hold (issue #2 RL session). Trained in Isaac Lab
    # (rl/bed_reach_env_cfg.py), deployed via the exported TorchScript actor. Unused for the
    # legacy --pink path and the dataset --replay path.
    bedreach = ({i: locomod.BedReachPolicy(robots[i], sim.device) for i in (0, 1)}
                if not (ARGS.pink or ARGS.replay or ARGS.walk_only) else {})

    # ── Device Connect swarm ──
    coord = None
    if not ARGS.no_device_connect:
        from examples.isaac_bed_making.coordination import SwarmCoordinator
        coord = SwarmCoordinator(mode="broker" if ARGS.broker else "loopback", nats_url=ARGS.nats_url)
        print(f"[demo] Device Connect swarm online ({coord.mode}): {coord.start()}")

    state = {"frame": 0, "t": 0.0}
    cloth_view = None
    fabric_points = None

    def step(n=1):
        # Pure physics: advance the simulation. The robots are NEVER kinematically held —
        # no base pin, no teleport, no joint freeze. They stand, walk and reach entirely
        # under their controllers (the velocity policy balancing the legs + Pink IK on the
        # arms), so every motion is one a real G1 could reproduce on hardware (sim-to-real).
        for _ in range(n):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_dt)
            state["t"] += sim_dt

    def capture():
        if cam is None:
            return
        if cloth_view is not None and fabric_points is not None:
            # Drive the thick visual slab from the live particle positions (the thin
            # membrane it replaces is hidden). fabric_points is the SHELL's points.
            clothmod.sync_shell_fabric(cloth_view, fabric_points, shell)
        sim.render()  # updates the live GUI viewport too, when --gui
        cam.update(dt=sim_dt)
        if not ARGS.render:
            return  # GUI-only: shown live, nothing to save
        out = cam.data.output["rgb"]
        if out is None or out.shape[0] == 0:
            return
        rgb = out[0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
        from PIL import Image
        Image.fromarray(rgb).save(frames_dir / f"frame_{state['frame']:04d}.png")
        state["frame"] += 1
        if state["frame"] % 20 == 0:
            print(f"[demo] captured {state['frame']} frames (t={state['t']:.1f}s)", flush=True)

    def control(cmd_fn, n_ctl, cap_every=2, pols=None):
        """Run a locomotion policy for n_ctl control steps (each = DECIMATION sim
        steps). ``cmd_fn(i)`` returns (command, arrived) for robot i. ``pols`` selects
        which policy set drives the legs (walk vs. stance); defaults to the walkers.
        The command must match the chosen policy (``command_to``/``stand`` of the same
        set). Stops early when both robots report arrived."""
        pols = pols or locos
        for c in range(n_ctl):
            arrived = []
            for i in (0, 1):
                cmd, arr = cmd_fn(i)
                pols[i].act(cmd)
                arrived.append(arr)
            step(DECIMATION)
            if c % cap_every == 0:
                capture()
            if all(arrived) or state["t"] > ARGS.max_seconds:
                return c
        return n_ctl

    def balance():
        """Let the velocity policy balance both robots' LEGS in place (zero command,
        legs-only so it leaves the arms + waist to Pink IK). Call once per manipulation
        control step — this is what keeps the robots upright on their own feet while they
        reach over the bed, with no kinematic pinning."""
        for i in (0, 1):
            stance[i].act([0.0, 0.0, 0.0], legs_only=True)

    # Register the cloth in physics, open a tensor view for live positions, and grab
    # the Fabric points handle we blit the deformation into each render.
    step(1)
    cloth_view = clothmod.make_cloth_view("/World/Sheet")
    # Blit into the SHELL's Fabric points (the visible thick slab), not the hidden
    # membrane. Positions still come from the membrane's tensor cloth view.
    fabric_points = clothmod.make_fabric_points(shell.prim_path)
    if fabric_points is None:
        print("[demo] WARNING: cloth not in Fabric — sheet deformation may not render", flush=True)

    # Cloth settle probe (DIAGNOSTIC, opt-in via BEDDEMO_SETTLE_PROBE): the robots are still at their
    # far spawn here, so the cover is UNTOUCHED — this isolates the drape→settle from any robot contact.
    # It reports the max particle frame-to-frame displacement, which decays toward ~0 once the cover
    # comes to rest and stays high if it sloshes (the user-reported jiggle). Off by default (adds settle
    # time); set BEDDEMO_SETTLE_PROBE=1 to measure.
    if os.environ.get("BEDDEMO_SETTLE_PROBE"):
        prev_pts = None
        for s in range(int(2.5 / (DECIMATION * sim_dt))):
            balance()                       # robots just stand at the far spawn; the cloth settles on its own
            step(DECIMATION)
            pts = clothmod.view_positions(cloth_view)
            if prev_pts is not None and pts.shape == prev_pts.shape:
                disp = float(np.max(np.linalg.norm(pts - prev_pts, axis=1)))
                if s % 3 == 0:
                    print(f"[cloth] settle t={state['t']:.2f}s max_step_disp={disp * 100:.2f}cm", flush=True)
            prev_pts = pts
            if s % 2 == 0:
                capture()

    body_ids = {}

    def _bid(i, name):
        key = (i, name)
        if key not in body_ids:
            bid, _ = robots[i].find_bodies([name])
            body_ids[key] = int(bid[0])
        return body_ids[key]

    def hand_pos(i):
        p = robots[i].data.body_pose_w[0, _bid(i, "right_wrist_yaw_link"), :3]
        return float(p[0]), float(p[1]), float(p[2])

    joint_ids_cache = {}

    def arm_q(i):
        """(right shoulder_pitch, right elbow) joint angles in rad — telemetry to check the arms
        hang at the sides (≈0) rather than the velocity policy's bent default (elbow ≈0.97)."""
        for nm in ("right_shoulder_pitch_joint", "right_elbow_joint"):
            if (i, nm) not in joint_ids_cache:
                jid, _ = robots[i].find_joints([nm])
                joint_ids_cache[(i, nm)] = int(jid[0])
        q = robots[i].data.joint_pos[0]
        return (float(q[joint_ids_cache[(i, "right_shoulder_pitch_joint")]]),
                float(q[joint_ids_cache[(i, "right_elbow_joint")]]))

    def foot_z(i):
        zl = float(robots[i].data.body_pose_w[0, _bid(i, "left_ankle_roll_link"), 2])
        zr = float(robots[i].data.body_pose_w[0, _bid(i, "right_ankle_roll_link"), 2])
        return min(zl, zr)

    def diag(label):
        pts = clothmod.view_positions(cloth_view) if cloth_view is not None else np.zeros((0, 3))
        parts = [f"[diag] {label}"]
        if pts.shape[0]:
            cen = pts.mean(axis=0)
            parts.append(f"sheet_cen=({cen[0]:.2f},{cen[1]:.2f},{cen[2]:.2f}) z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
        for i in (0, 1):
            bx, by = locos[i].base_xy()
            hx, hy, hz = hand_pos(i)
            gap = float(np.min(np.linalg.norm(pts - np.array([hx, hy, hz]), axis=1))) if pts.shape[0] else -1.0
            sp, el = arm_q(i)
            # foot_z<0 means the foot is through the floor; arm(sp,el)≈(0,0) = hands at the sides
            parts.append(f"r{i}_base=({bx:.2f},{by:.2f},z{float(robots[i].data.root_pos_w[0,2]):.2f}) "
                         f"hand=({hx:.2f},{hy:.2f},{hz:.2f}) gap={gap:.2f} foot_z={foot_z(i):.2f} "
                         f"arm(sp{sp:.2f},el{el:.2f})")
        print("  ".join(parts), flush=True)

    # ── PHASE 0: find footing (the velocity policy balances in place) + sheet settles ──
    # Legacy walk paths only. The RL path skips this: the bed-reach policy balances the robot
    # (from its hands-at-sides spawn) and lets the sheet settle in run_bedmaking_rl's own
    # opening phase — no velocity walker involved.
    if not RL_PATH:
        print("[demo] robots find their footing; the sheet settles onto the foot of the bed…")
        control(lambda i: (stance[i].stand(), False), n_ctl=int(1.5 / (DECIMATION * sim_dt)),
                cap_every=4, pols=stance)
        diag("after stand-settle")

    # ── PHASE 1: learned approach-walk (hands at the sides) ──
    # Both paths walk in: the velocity policy strides to the bedside while the sheet drapes.
    if not ARGS.no_walk:
        if coord:
            for i in (0, 1):
                coord.invoke(i, "walkToNextCorner", direction="approach")
        print("[demo] both G1s walk to the bed under the learned policy (hands at their sides)…")
        steps = control(lambda i: locos[i].command_to(scenemod.ROBOTS[i]["manip"]),
                        n_ctl=int(8.0 / (DECIMATION * sim_dt)), cap_every=2)
        diag(f"arrived after {steps} control steps")

    # ── PHASE 2: make the bed — the robots BALANCE ON THEIR OWN FEET throughout ──
    # No pin, no teleport, no kinematic freeze — every motion is one a real G1 could do.
    if not ARGS.walk_only:
        if ARGS.replay or ARGS.pink:
            # Legacy paths: the velocity policy balances the legs; settle a moment so both feet
            # are planted before the arms start.
            print("[demo] at the bedside — finding a steady stance before reaching…", flush=True)
            control(lambda i: (stance[i].stand(), False), n_ctl=int(0.8 / (DECIMATION * sim_dt)),
                    cap_every=4, pols=stance)
            capture()
            diag("at the bedside (balancing on its own feet)")
        if ARGS.replay:
            run_replay(reps, robots, balance, coord, step, capture, diag, sim_dt)
        elif ARGS.pink:
            # Legacy Pink-IK stand-and-reach: the velocity policy balances the legs while Pink
            # IK drives arms+waist. Gain-schedule the waist+arms firmer for the IK (still PD
            # dynamics, nothing kinematic), well below the stock rigid values so a sharp arm
            # command can't kick the free base over; legs keep their soft walk gains.
            for i in (0, 1):
                wids, _ = robots[i].find_joints(["waist_.*_joint"])
                aids, _ = robots[i].find_joints([".*_shoulder_.*_joint", ".*_elbow_joint",
                                                 ".*_wrist_.*_joint"])
                robots[i].write_joint_stiffness_to_sim(300.0, joint_ids=wids)
                robots[i].write_joint_damping_to_sim(8.0, joint_ids=wids)
                robots[i].write_joint_stiffness_to_sim(150.0, joint_ids=aids)
                robots[i].write_joint_damping_to_sim(10.0, joint_ids=aids)
            run_bedmaking(manip, balance, coord, step, capture, diag, stage, cloth_view, sheet)
        else:
            # Default RL path: the whole-body reach policy owns legs+waist+arms and balances the
            # robot from its hands-at-sides spawn — no walker, no scripted arm motion.
            run_bedmaking_rl(bedreach, coord, step, capture, diag, stage, cloth_view, sheet, tabs)
    else:
        diag("walk-only: standing at the bedside")

    # ── report ──
    if coord:
        print("\n[demo] Device Connect event history (peer 0):")
        for e in coord.event_history(0, limit=40):
            print(f"   [{e['ts']}] {e['kind']:<14} {e['summary']}")
        goals = [coord.invoke(i, "getGoalState") for i in (0, 1)]
        print(f"[demo] goal state: {goals}")
        coord.stop()

    if ARGS.render and state["frame"] > 0:
        encode(frames_dir, state["frame"])
    print("[demo] done.")
    return 0


def run_replay(reps, robots, balance, coord, step, capture, diag, sim_dt):
    """Replay the real recorded bed-making arm + waist motion while the velocity policy
    balances each robot's legs (``balance()`` every control step — no pin). Only
    waist/arms/fingers are replayed; the legs balance the free base on their own. Robot 1's
    180° spawn makes it a mirrored peer; ``--lag`` desyncs them."""
    N = reps[0].n_frames
    spf = max(1, round((1.0 / sim_dt) / reps[0].fps / max(0.1, ARGS.replay_speed)))
    ctl_per_frame = max(1, round(spf / DECIMATION))
    lag = int(ARGS.lag * reps[0].fps)
    cap_every = max(1, N // 160)

    def hold_steps(n_ctl):
        # The waist/arms/fingers follow the dataset; the legs balance the free base.
        for _ in range(n_ctl):
            balance()
            step(DECIMATION)

    print(f"[demo] arranging the sheet — real motion (episode {reps[0].source_episode}, "
          f"{N} frames, {spf} sim steps/frame)", flush=True)
    if coord:
        coord.invoke(0, "pickUpBedSheet", corner="A")
        coord.invoke(1, "pickUpBedSheet", corner="B")
    # Ease arms from rest into the first recorded pose.
    for w in range(40):
        reps[0].warmup((w + 1) / 40.0)
        reps[1].warmup((w + 1) / 40.0)
        hold_steps(1)
        if w % 8 == 0:
            capture()
    mid_done = False
    for j in range(N + lag):
        reps[0].apply(j)
        reps[1].apply(j - lag)
        hold_steps(ctl_per_frame)
        if j % cap_every == 0:
            capture()
        if coord and not mid_done and j >= N // 2:
            coord.invoke(0, "askForHelp", corner="A", reason="squaring my side")
            coord.invoke(1, "offerHelp", target=coord.peers[0].device_id, corner="A")
            mid_done = True
    if coord:
        coord.invoke(0, "putDownBedSheet", corner="A")
        coord.invoke(1, "putDownBedSheet", corner="B")
    diag("after arranging")
    capture()


def run_bedmaking(manip, balance, coord, step, capture, diag, stage, cloth_view, sheet):
    """Two-handed friction bed-making (no kinematic attachment). The sheet starts flat
    and gathered toward the foot. Each G1, BALANCING ON ITS OWN FEET at the bedside (the
    velocity policy holds the legs via ``balance()`` every control step — no pin), uses
    BOTH hands to win a grip and then pulls with one:

      1. both palms press the sheet's head edge near the forward corner;
      2. the assisting (left) hand slides inboard, bunching a handful of cloth into the
         gripping (right) hand;
      3. the right hand CLOSES its Inspire fingers on the handful (high-friction grip)
         while the left presses it home, then the left opens and lifts away;
      4. the right hand drags the handful toward the head, drawing the sheet up the bed.

    Robot 0 works the −y side, robot 1 the +y side (mirror). The grip is friction +
    finger-cage only — the closed fingers and rubberized palms hold the cloth."""
    TOP = scenemod.BED_TOP_Z
    sign = {0: -1.0, 1: 1.0}
    HEAD_EDGE_X = scenemod.SHEET_ORIGIN[0] - scenemod.SHEET_SIZE[0] / 2.0  # ~0.15
    HEAD_PULL_X = scenemod.HEAD_X + 0.65  # = -0.35
    # The robots BALANCE on their own feet (no pin), so the reach must keep the centre of
    # mass over the feet — a deep lunge across the bed topples the walking balancer. So we
    # grab the sheet's NEAR-SIDE overhang, right at the robot's own side (y≈±0.95, just
    # inboard of the robot at ±1.05), hands at cover height (NO pressing down into the
    # mattress — that reaction force pitches the robot over), and pull gently. It is the
    # honest physical limit: like a person tucking the sheet edge nearest them.
    GY = 0.92           # gripping (right) hand y — the near-side sheet overhang, modest lean
    LY = 1.06           # assisting (left) hand y — out at the robot's own side
    APPROACH_Z = TOP + 0.16   # ≈0.82 — wrist above the cover, no contact yet
    GRIP_Z = TOP + 0.05       # ≈0.71 — wrist at the cover top, gentle contact (no press in)
    LIFT_Z = TOP + 0.22       # left-hand retreat height
    PULL_DX = -0.22           # headward drag (relative to the grip point) — kept short
    R, L, gR, gL = manip["R"], manip["L"], manip["gR"], manip["gL"]

    def drive(n_ctl, rt, lt, rg, lg, cap=4):
        """Hold/seek both arms toward per-robot targets rt[i]/lt[i] (None = leave as-is)
        with finger fractions rg[i]/lg[i], for n_ctl control steps."""
        for i in (0, 1):
            if rt is not None and rt[i] is not None:
                R[i].set_target(rt[i])
            if lt is not None and lt[i] is not None:
                L[i].set_target(lt[i])
        dr = dl = None
        for s in range(n_ctl):
            for i in (0, 1):
                dr = R[i].tick()
                dl = L[i].tick()
                gR[i].set(rg[i])
                gL[i].set(lg[i])
            balance()   # legs keep balancing the free base while the arms reach
            step(DECIMATION)
            if s % cap == 0:
                capture()
        return dr, dl

    def by_side(fn):
        return {i: fn(i) for i in (0, 1)}

    if coord:
        coord.invoke(0, "pickUpBedSheet", corner="A")
        coord.invoke(1, "pickUpBedSheet", corner="B")

    # 1) both palms descend onto the head edge — right where it will grip, left just
    #    inboard to pin the cloth flat — fingers open, soft landing.
    print("[demo] both hands settle on the head edge…", flush=True)
    rt = by_side(lambda i: (HEAD_EDGE_X, sign[i] * GY, APPROACH_Z))
    lt = by_side(lambda i: (HEAD_EDGE_X, sign[i] * LY, APPROACH_Z))
    drive(55, rt, lt, {0: 0.0, 1: 0.0}, {0: 0.0, 1: 0.0})
    diag("hands on the edge")

    # 2) press DOWN hard (wrist into the cover) and close the right fingers on the cloth.
    #    The left presses alongside to PIN the cover flat so the right's press grips it
    #    rather than shoving it away. The left stays open (a flat pin, not a grab).
    print("[demo] pressing in and gripping a handful (right hand)…", flush=True)
    rt = by_side(lambda i: (HEAD_EDGE_X, sign[i] * GY, GRIP_Z))
    lt = by_side(lambda i: (HEAD_EDGE_X, sign[i] * LY, GRIP_Z))
    drive(28, rt, lt, {0: 0.5, 1: 0.5}, {0: 0.0, 1: 0.0})   # press, fingers half-curl
    drive(22, rt, lt, {0: 1.0, 1: 1.0}, {0: 0.0, 1: 0.0})   # fingers close hard on the cloth
    diag("gripped")

    # 3) the left hand lifts clear so only the gripping hand remains on the cloth.
    print("[demo] freeing the assisting hand…", flush=True)
    lt = by_side(lambda i: (HEAD_EDGE_X, sign[i] * LY, LIFT_Z))
    drive(24, rt, lt, {0: 1.0, 1: 1.0}, {0: 0.0, 1: 0.0})
    diag("left hand clear")

    # 4) the gripping hand drags the cover toward the head, fingers held closed and the
    #    wrist held LOW so it keeps pressing as it slides. The target is RELATIVE to where
    #    the hand actually gripped (same y/z, x moved headward) so the reach stays in the
    #    arm's workspace — a far world target sent the arm flying up.
    print("[demo] pulling the cover toward the head with one hand…", flush=True)
    h = {i: R[i].ee_pos() for i in (0, 1)}
    rt = by_side(lambda i: (max(HEAD_PULL_X, h[i][0] + PULL_DX), h[i][1], h[i][2]))
    drive(95, rt, None, {0: 1.0, 1: 1.0}, {0: 0.0, 1: 0.0}, cap=3)
    diag("pulled the sheet toward the head")

    # 5) release + a help exchange over Device Connect, then settle
    drive(10, None, None, {0: 0.0, 1: 0.0}, {0: 0.0, 1: 0.0})
    if coord:
        coord.invoke(0, "askForHelp", corner="A", reason="squaring my side")
        coord.invoke(1, "offerHelp", target=coord.peers[0].device_id, corner="A")
        coord.invoke(0, "putDownBedSheet", corner="A")
        coord.invoke(1, "putDownBedSheet", corner="B")
    for s in range(40):
        balance()
        step(DECIMATION)
        if s % 4 == 0:
            capture()
    diag("after bed-making")
    capture()


def run_bedmaking_rl(bedreach, coord, step, capture, diag, stage, cloth_view, sheet, tabs):
    """Make the bed with the whole-body loco-manipulation RL policy (issue #2 RL session).

    By here the G1s have walked to the bedside (hands at their sides) and the sheet has draped. ONE
    ambidextrous policy per robot owns all 29 body joints, so each BALANCES on its own two feet
    *while* it leans and squats to reach the sheet — the skill a walking-balance policy can't hold
    (it step-recovers on the deep lean and topples). We feed each robot a WORLD hand target each
    control step (it transforms it into its base frame and solves the whole-body motion in one
    forward pass); the policy reaches with whichever hand is on the target's side. No pin, no
    teleport, no joint freeze — physically valid for sim-to-real.

    Each G1, flanking its side of the bed, reaches onto the sheet's near head-side edge, grips it
    (the custom SPRING PEEL-OFF grip: a compliant D6 spring joint from the wrist to a rigid tab sewn
    into the cover — the robot feels the load and balances, and the grip peels off under an excessive
    draw rather than toppling the robot), and draws it toward the head. The target is aimed HEADWARD
    (world −x), which is base +y
    for robot 0 (−y side) and base −y for robot 1 (+y side): so robot 0 leads with its LEFT hand
    and robot 1 with its RIGHT — a natural same-side abduction for both, not the cross-body sweep
    one hand can't balance."""

    def _envf(name, default):   # tuning knobs from env vars (default off → code values); documented config
        v = os.environ.get(name)
        return float(v) if v not in (None, "") else default

    # Trust the reach policy for the MOTION; command WORLD-frame hand targets (which reach the cover
    # correctly — a fixed world point compensates for the base position) and COMMIT each robot to ONE
    # hand via a one-sided base_y clamp (locomotion.BedReachPolicy.hand_lock) so the grab and the
    # headward pull stay on the SAME hand. (The prior failure: the active hand flipped between grab and
    # pull as the base drifted, so the gripping hand went idle while the empty hand swept headward and
    # the cover didn't move.) HEADWARD = world −x; r0 (−y side) leads with its LEFT hand, r1 with RIGHT.
    sign = {0: -1.0, 1: 1.0}                  # world y of each robot's side of the bed
    HAND_LOCK = {0: "left", 1: "right"}
    TOP = scenemod.BED_TOP_Z
    REACH_Y = _envf("BEDDEMO_REACH_Y", 0.80)   # world |y| of the grip point (inboard of the y=±0.9 corner, onto the cover)
    # Grab the cover at its HEAD EDGE, which sits HEADWARD (world −x) of the head-flanking robot
    # (robot at MANIP_X≈-0.20, head edge at ≈-0.35). That is the whole fix for the grab/pull
    # hand-conflict: a headward grab point is base +y for r0 / base −y for r1 — the SAME side as
    # the headward pull — so one hand owns grab+pull (the old default MX put the grab at the
    # robot's own x → base_y≈0 → the pull then flipped to the other hand). hand_lock holds it.
    REACH_X = _envf("BEDDEMO_REACH_X", scenemod.SHEET_HEAD_X)
    APPROACH_Z = _envf("BEDDEMO_APPROACH_Z", TOP + 0.14)   # hover above the cover first (no contact)
    GRIP_Z = _envf("BEDDEMO_GRIP_Z", TOP + 0.0)            # descend ONTO the cover so the hand contacts it
    PULL_DX = _envf("BEDDEMO_PULL_DX", -0.26)  # headward draw (world −x); one-sided clamp caps the hand ~at the pillow line (x≈-0.58)
    PULL_DZ = _envf("BEDDEMO_PULL_DZ", 0.03)   # slight lift as it draws, to ride over the bare mattress
    PULL_SECS = _envf("BEDDEMO_PULL_SECS", 3.0)  # draw
    HOLD_SECS = _envf("BEDDEMO_HOLD_SECS", 1.5)  # dwell at full extension so the gripped cloth catches up
    # The grip is the CUSTOM SPRING PEEL-OFF GRIP (cloth.add_spring_grip). The cover is studded with
    # light rigid grab tabs (built in main via the supported cloth↔rigid attachment, and filtered from
    # the robots so the reaching hand never plows into them). do_grasp creates a compliant D6 spring
    # joint from the active wrist to the nearest tab: PhysX applies the spring as a real bilateral force
    # so the robot feels the cover's load and must balance against it (sim-to-real, NOT a pin), and a
    # break_force PEELS the joint off when the draw load gets dangerous (requirement E, physical let-go).
    # Softened + early-peeling vs spring_v9 (there the draw toppled the balancing bipeds before the
    # cover reached the head): a compliant 280 N/m spring lets the cover follow without a yank, and a
    # 24 N peel-off (the deterministic SOFTWARE peel below — PhysX's break_force is not guaranteed to
    # fire on a DRIVE constraint, so we mirror it ourselves — and PhysX's break_force as a backstop)
    # lets go just under the load that drags a leaning free-base G1 off its feet: progress, no topple.
    GRIP_K = _envf("BEDDEMO_GRIP_STIFFNESS", 280.0)   # spring stiffness (N/m) holding the tab to the wrist
    GRIP_C = _envf("BEDDEMO_GRIP_DAMPING", 22.0)      # spring damping (N·s/m)
    GRIP_BREAK = _envf("BEDDEMO_GRIP_BREAK", 24.0)    # peel-off load (N): the cover snags harder than this -> let go

    # Sensor-driven grasp DECISION (grasp.py): the reach policy is trusted with the motion; this
    # decides WHEN to close/open the physical cloth grip from a short-range hand sensor. We never
    # move the robot or cloth kinematically.
    SENSING_RANGE = _envf("BEDDEMO_SENSE_RANGE", 0.15)   # a real hand sensor sees nothing past a few cm
    CONTACT_GAP = _envf("BEDDEMO_CONTACT_GAP", 0.10)     # within this the cloth is in the hand -> grab
    SLIP_GAP = _envf("BEDDEMO_SLIP_GAP", 0.28)           # gripped but sensor lost it this far -> let go
    SEEK_PRESS = _envf("BEDDEMO_SEEK_PRESS", 0.10)       # press the live-cloth seek target this far BELOW
    #                                                      the cover so the descending hand pins it
    sensors = {i: graspmod.HandClothSensor(SENSING_RANGE) for i in (0, 1)}
    decide = {i: graspmod.GraspDecision(CONTACT_GAP, SLIP_GAP) for i in (0, 1)}
    recovering = {0: False, 1: False}   # set once a robot lets go on balance-loss (requirement E)
    fgrip = {}                          # per-robot FingerGrip on the active hand (closes for the grasp look)
    grip_joint = {0: None, 1: None}    # per-robot spring-grip joint prim path while holding (None = released)
    grip_state = {0: None, 1: None}    # per-robot {vid, lp0, K}: the gripped tab + spring rest, for the load read
    grab_pose = {0: None, 1: None}     # per-robot world hand pose at grasp — the pull ramps FROM here (no yank)
    wrist_paths = {}                   # cache: (i, wrist_link_name) -> full prim path under the robot root
    # NOTE: the seek runs with NO hand_lock (hand_lock stays None) so each hand can reach FREELY to the
    # live cover wherever it settled — committing to the headward hand up front forced the hand outboard,
    # off the bed, away from the cover (eye-verified spring_v8). do_grasp then LOCKS to whichever hand
    # actually grabbed, so the PULL drives the gripping hand headward (not the old empty-hand sweep).

    def wrist_path(i):
        """Full USD prim path of robot i's active wrist link (body0 of the spring-grip joint). Resolved
        by name from the robot subtree so it's robust to however the USD nests the links."""
        link = bedreach[i].active_wrist_link()
        key = (i, link)
        if key not in wrist_paths:
            from pxr import Usd
            root = stage.GetPrimAtPath(f"/World/envs/env_0/Robot_{i}")
            wrist_paths[key] = next(
                (p.GetPath().pathString for p in Usd.PrimRange(root) if p.GetName() == link),
                f"/World/envs/env_0/Robot_{i}/{link}")
        return wrist_paths[key]

    def reach(i, z):                  # world-frame grip target (hand fixed by the one-sided clamp)
        return (REACH_X, sign[i] * REACH_Y, z)

    def seek_target(i):
        # Aim at the LIVE cover nearest the hand (re-evaluated each control step in run()), pressed
        # slightly DOWN so the descending hand pins the cover to the bed instead of shoving the
        # featherlight cloth away. This adapts to wherever the cover actually settled on the robot's
        # side — the fixed grip point misses because the cover doesn't settle where it's authored
        # (and the reach itself nudges it). Falls back to the fixed point until the cloth view is up.
        if cloth_view is None:
            return reach(i, GRIP_Z)
        pts = clothmod.view_positions(cloth_view)
        if pts.shape[0] == 0:
            return reach(i, GRIP_Z)
        h = np.asarray(bedreach[i].ee_pos())
        same_side = pts[pts[:, 1] * sign[i] > -0.10]          # the cover on this robot's half of the bed
        cand = same_side if same_side.shape[0] else pts
        c = cand[int(np.argmin(np.linalg.norm(cand - h, axis=1)))]
        return (float(c[0]), float(c[1]), float(c[2]) - SEEK_PRESS)

    def pull_to(i, frac):
        # World-frame headward draw, RAMPED by ``frac`` (0→1) FROM THE ACTUAL GRAB POSE. The live seek
        # grabs wherever the cover actually settled (footward of the authored head edge — pull_v1 grabbed
        # at x≈0.76, y≈-1.03, not the REACH point at 0.15, ±0.80), so ramping from the authored point
        # jumped the target ~0.65 m and yanked the rigid grip to 76 N → instant peel/topple. Starting at
        # the grab pose and easing headward in x (holding the grab y, slight z lift) means no jump: the
        # policy walks the cover up and the accordion feeds slack instead of lunging at a far point.
        gx, gy, gz = grab_pose[i] if grab_pose[i] is not None else (REACH_X, sign[i] * REACH_Y, GRIP_Z)
        return (gx + frac * PULL_DX, gy, gz + frac * PULL_DZ)

    def torso_up_z(i):
        # World-z component of the torso's own up-axis = R(root_quat)[2,2] = 1 − 2(x²+y²) for a wxyz
        # quat. 1.0 = perfectly upright; cos(θ) at a θ-degree lean. A deep working squat keeps the
        # TORSO upright (knees/hips bend, torso stays near-vertical), so this isolates a topple from
        # the intended deep reach — unlike pelvis height, which conflates the two.
        q = bedreach[i].robot.data.root_quat_w[0]
        return 1.0 - 2.0 * float(q[1] ** 2 + q[2] ** 2)

    def balance_lost(i):
        # Requirement E (sim-to-real): a robot reads its own balance (IMU/proprioception). If the
        # gripped drag is tipping it off its feet — the torso TILTING past a recoverable lean, the
        # pelvis dropping toward a fall, the base launched out of a sane envelope, or a violent base
        # velocity — it LETS GO rather than be dragged over. NOT a kinematic cheat: nothing is moved;
        # we only decide to release the cloth grip.
        d = bedreach[i].robot.data
        pelvis_z = float(d.root_pos_w[0, 2])
        bx, by = bedreach[i].base_xy()
        v = d.root_lin_vel_w[0]
        speed = float((v[0] ** 2 + v[1] ** 2) ** 0.5)
        # up_z < cos(39°)≈0.78 ⇒ torso tilted past a recoverable lean (the earliest reliable topple
        # signal); the load-based SOFTWARE PEEL below should fire first, this is the balance backstop.
        return torso_up_z(i) < 0.78 or pelvis_z < 0.45 or speed > 1.0 or abs(bx) > 1.4 or abs(by) > 1.9

    def spring_load(i, pts=None):
        # Magnitude (N) of the spring-grip force on robot i ≈ K·‖tab − rest‖, where ``rest`` is where
        # the grab-time wrist→tab offset (lp0, in the wrist frame) would place the tab now — so an
        # unloaded grip reads ~0. This is the bilateral load the draw puts on the wrist. We peel
        # deterministically on it (a software mirror of break_force) and log it. 0 if not gripping.
        st = grip_state[i]
        if st is None or cloth_view is None:
            return 0.0
        if pts is None:
            pts = clothmod.view_positions(cloth_view)
        if st["vid"] >= pts.shape[0]:
            return 0.0
        from isaaclab.utils.math import quat_apply
        wp, wq = bedreach[i].active_wrist_pose_w()
        rest_w = wp + quat_apply(wq.unsqueeze(0), st["lp0"].unsqueeze(0))[0]
        tab_w = torch.tensor(pts[st["vid"]], device=bedreach[i].device, dtype=torch.float32)
        return float(st["K"] * torch.linalg.norm(tab_w - rest_w))

    def do_grasp(i, gap):
        # CUSTOM SPRING PEEL-OFF GRIP: create a compliant D6 spring joint from the active wrist to the
        # NEAREST grab tab. Each tab's live position is its bound cloth node (read from the cloth view at
        # the tab's vid), so we pick the tab closest to the hand without a separate rigid-body view. The
        # spring's rest pose is the grab-time wrist→tab offset (local_pos0 in the wrist frame), so it
        # engages with ~no jerk; PhysX applies it as a real force (the robot feels the cover and must
        # balance) and break_force peels it off under a dangerous load. No kinematic attachment.
        from isaaclab.utils.math import quat_rotate_inverse
        pts = clothmod.view_positions(cloth_view)
        ah = bedreach[i].ee_pos()
        h = np.asarray(ah, dtype=float)
        tab_path, vid = min(
            tabs, key=lambda tv: float(np.linalg.norm(pts[tv[1]] - h)) if tv[1] < pts.shape[0] else 1e9)
        wp, wq = bedreach[i].active_wrist_pose_w()
        node_t = torch.tensor(pts[vid], device=bedreach[i].device, dtype=torch.float32)
        lp0 = quat_rotate_inverse(wq.unsqueeze(0), (node_t - wp).unsqueeze(0))[0]
        grip_joint[i] = clothmod.add_spring_grip(
            stage, f"/World/Grip_{i}", wrist_path(i), tab_path,
            (float(lp0[0]), float(lp0[1]), float(lp0[2])),
            stiffness=GRIP_K, damping=GRIP_C, break_force=GRIP_BREAK)
        grip_state[i] = {"vid": vid, "lp0": lp0, "K": GRIP_K}   # the gripped tab + spring rest, for spring_load()
        grab_pose[i] = bedreach[i].ee_pos()   # where the hand actually grabbed → the pull ramps FROM here (no yank)
        fgrip[i].set(1.0)   # close the hand for the grasp LOOK; the spring joint does the holding
        # Now LOCK to the hand that grabbed (the seek ran unlocked, so the active hand = the one that
        # reached the cover). The PULL then drives THIS gripping hand headward, dragging the attached
        # cover — not the empty other hand. Read _active_is_left BEFORE assigning (hand_lock is still None).
        bedreach[i].hand_lock = "left" if bedreach[i]._active_is_left() else "right"
        print(f"[grip] r{i} GRASP — spring joint wrist→{tab_path.split('/')[-1]} at "
              f"hand=({ah[0]:.2f},{ah[1]:.2f},{ah[2]:.2f}) sensor_gap={gap:.3f} "
              f"K={GRIP_K:.0f} break={GRIP_BREAK:.0f}N wrist={bedreach[i].active_wrist_link()}", flush=True)

    def do_release(i, why):
        clothmod.remove_grip(stage, grip_joint[i])   # peel the spring grip (idempotent if PhysX broke it)
        grip_joint[i] = None
        grip_state[i] = None
        decide[i].slipped = True   # no longer holding (holding = gripped and not slipped) → stop re-checking
        fgrip[i].set(0.0)
        print(f"[grip] r{i} RELEASE ({why})", flush=True)

    def sheet_report(label):
        cw = clothmod.corner_world_positions(cloth_view, sheet) if cloth_view is not None else {}
        nw, sw = cw.get("NW"), cw.get("SW")
        pts = clothmod.view_positions(cloth_view) if cloth_view is not None else np.zeros((0, 3))
        hands = " ".join(
            f"r{i}hand=({(h:=bedreach[i].ee_pos())[0]:.2f},{h[1]:.2f},{h[2]:.2f})"
            f"/gap{graspmod.HandClothSensor.true_gap(h, pts):.2f}" for i in (0, 1))
        if nw and sw:
            print(f"[sheet] {label}: NW=({nw[0]:.2f},{nw[1]:.2f},{nw[2]:.2f}) "
                  f"SW=({sw[0]:.2f},{sw[1]:.2f},{sw[2]:.2f})  {hands}", flush=True)
        return cw

    def _aim(i, tgt_fn):
        # A robot that let go on balance-loss holds a neutral near-body reach to re-settle on its
        # feet; everyone else tracks the commanded world target.
        if recovering[i]:
            bedreach[i].set_base_target([0.30, 0.0, -0.05])
        else:
            bedreach[i].set_world_target(tgt_fn(i))

    def run(n_ctl, tgt_fn, cap=3, mode=None):
        """Drive the trusted reach policy to ``tgt_fn(i)`` for n_ctl steps. ``mode``:
        'seek' = grab the instant the hand sensor reports contact; 'pull' = release if the grip
        slipped OR the drag is toppling the robot. The policy owns the whole-body motion throughout."""
        for s in range(n_ctl):
            for i in (0, 1):
                _aim(i, tgt_fn)         # re-aim EACH step so a live (cloth-tracking) target follows the cover
                bedreach[i].act()       # the policy owns the whole body (legs balance, arms reach)
            step(DECIMATION)
            if mode and cloth_view is not None:
                pts = clothmod.view_positions(cloth_view)
                for i in (0, 1):
                    h = bedreach[i].ee_pos()
                    sensed = sensors[i].read(h, pts)
                    true_gap = graspmod.HandClothSensor.true_gap(h, pts)
                    if mode == "seek" and decide[i].on_reach(sensed, true_gap):
                        do_grasp(i, sensed)
                    elif mode == "pull" and decide[i].holding:
                        load = spring_load(i, pts)
                        if not recovering[i] and balance_lost(i):
                            do_release(i, "losing balance — let go (requirement E)")
                            recovering[i] = True
                            bedreach[i].set_base_target([0.30, 0.0, -0.05])
                        elif load > GRIP_BREAK:
                            # Deterministic peel: the draw load exceeded the grip's hold (requirement E,
                            # physical let-go) BEFORE it could drag the robot over — the whole point of a
                            # compliant peel-off grip vs a rigid tether that topples the free-base biped.
                            # The cloth is gone, so re-settle on the feet rather than keep lunging at the
                            # (now empty) pull target — which otherwise drives the post-peel collapse (pull_v1).
                            do_release(i, f"grip peeled — load {load:.0f}N over {GRIP_BREAK:.0f}N (requirement E)")
                            recovering[i] = True
                            bedreach[i].set_base_target([0.30, 0.0, -0.05])
                        elif decide[i].on_pull(sensed, true_gap):
                            do_release(i, "sensor lost the cloth (slipped)")
                if mode == "pull" and s % cap == 0:
                    for i in (0, 1):
                        d = bedreach[i].robot.data
                        bx, by = bedreach[i].base_xy()
                        v = d.root_lin_vel_w[0]
                        sp = float((v[0] ** 2 + v[1] ** 2) ** 0.5)
                        print(f"[pull] r{i} hold={int(decide[i].holding)} load={spring_load(i, pts):4.0f}N "
                              f"up_z={torso_up_z(i):.2f} pelvis_z={float(d.root_pos_w[0, 2]):.2f} "
                              f"base=({bx:+.2f},{by:+.2f}) speed={sp:.2f} recov={int(recovering[i])}", flush=True)
            if s % cap == 0:
                capture()

    def secs(t):
        return max(1, int(t / (DECIMATION * SIM_DT)))   # 50 Hz control

    # 0) Hand off from the walk-in: the reach policy takes over balance (hands already at the sides
    #    from the walk) and eases the leading hand up to a ready height above the draped cover. The
    #    rigid grab tabs are already sewn into the cover (in main, before the first step, via the
    #    SUPPORTED cloth↔rigid attachment — cloth↔articulation-link is unsupported in Sim 5.1, the real
    #    reason a wrist attachment slips) and filtered from the robots. do_grasp later springs the
    #    active wrist to the nearest tab, so the drag load transmits cloth→tab→spring→robot (physical).
    for i in (0, 1):
        bedreach[i].reset()
        fgrip[i] = FingerGrip(bedreach[i].robot, HAND_LOCK[i], bedreach[i].device)
    print("[demo] at the bedside — the reach policy takes over and steadies…", flush=True)
    run(secs(1.8), lambda i: reach(i, APPROACH_Z), cap=4)
    diag("steady at the bedside (balancing on its own feet)")

    # 1) reach DOWN onto the sheet's near head edge; the DECISION layer grips the instant the hand
    #    sensor reports cloth in the hand (not at a fixed time) — the reach itself is the policy's.
    if coord:
        coord.invoke(0, "pickUpBedSheet", corner="A")
        coord.invoke(1, "pickUpBedSheet", corner="B")
    print("[demo] reaching onto the sheet; gripping when the hand sensor feels cloth…", flush=True)
    run(secs(2.6), seek_target, mode="seek")
    for i in (0, 1):
        print(f"[grip] r{i} closest approach this reach = {decide[i].min_true_gap:.3f} m "
              f"(gripped={decide[i].gripped})", flush=True)
    diag("done reaching")
    sheet_report("at grip")

    # 2) draw the cover up toward the pillows (long + held); release if the sensor says it slipped.
    if coord:
        coord.invoke(0, "askForHelp", corner="A", reason="squaring my side")
        coord.invoke(1, "offerHelp", target=coord.peers[0].device_id, corner="A")
    print("[demo] drawing the cover up toward the pillows…", flush=True)
    RAMP = 6                                          # ease the headward target out over the stroke
    for k in range(0, RAMP + 1):                      # start at frac=0 (the grab pose itself) → zero-jump handoff
        run(secs(PULL_SECS / RAMP), lambda i, fr=k / RAMP: pull_to(i, fr), cap=2, mode="pull")
    run(secs(HOLD_SECS), lambda i: pull_to(i, 1.0), cap=3, mode="pull")  # dwell so the gripped cloth catches up
    diag("drew the cover up")
    sheet_report("after pull")

    # 3) release whatever is still held + settle.
    for i in (0, 1):
        if decide[i].holding:
            do_release(i, "task done")
    run(secs(1.2), lambda i: pull_to(i, 1.0), cap=4)
    cw = sheet_report("after settle")

    # 4) Honest, tolerant goal: are the head-side corners drawn up into the pillow zone? Read the
    #    live cloth corners and test each against a pillow-anchored radius (coverage.py) -- no
    #    overseer cam, no trust-based tally. Only claim a corner placed on Device Connect if it
    #    actually landed there.
    report = None
    if cw:
        report = coveragemod.evaluate_bed_made(
            cw, {k: scenemod.BED_CORNERS[k] for k in ("NW", "SW")}, scenemod.PILLOWS)
        print(f"[goal] pillow-anchored bed-made check: {report.as_dict()}", flush=True)
    if coord:
        for i, label, ckey in ((0, "A", "SW"), (1, "B", "NW")):   # r0 grips the -y (SW) head corner
            if report is None or report.placed.get(ckey):
                coord.invoke(i, "putDownBedSheet", corner=label)
    diag("after bed-making (RL)")
    capture()


def encode(frames_dir, n):
    import shutil
    import subprocess
    out = frames_dir / "isaac_bed_making.mp4"
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-framerate", "20", "-pattern_type", "glob",
                        "-i", str(frames_dir / "frame_*.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", str(out)],
                       capture_output=True)
        print(f"[demo] wrote {out} ({n} frames)")
    else:
        print(f"[demo] {n} frames in {frames_dir} (install ffmpeg for mp4)")


if __name__ == "__main__":
    import traceback

    rc = 0
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    if ARGS.gui:
        # Keep the window open so you can orbit/inspect after the run; close it
        # (or Ctrl-C) to exit.
        print("[demo] GUI open — close the window or press Ctrl-C to exit.", flush=True)
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass
        simulation_app.close()
    else:
        # Isaac's replicator orchestrator can hang inside simulation_app.close() on
        # headless shutdown with cameras; artifacts are already written, so hard-exit.
        os._exit(rc)
