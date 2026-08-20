# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene factory for the SO-101 cuRobo synthetic-data demo (issue #67).

Builds a tabletop pick-and-place world: an SO-101 6-DoF arm, a small red cube
to grasp, a "bin" placement target, and a camera trio. The demo is
**backend-agnostic** by design (the executor/collector speak the ``SimEngine``
surface), but today the only runtime present on most boxes is the **MuJoCo**
backend, which already loads a real SO-101. :func:`make_sim` returns a MuJoCo
``Simulation`` by default and lazily attempts the Isaac backend when requested
(``create_simulation("isaac")``), degrading with a clear message if the Isaac
Sim runtime isn't installed.

See ``README.md`` for how this maps onto the issue's T1-T10 task breakdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("so101_curobo.scene")

# SO-101 is the canonical robot; the rest are progressively-more-available
# fallbacks with the same "small arm + gripper" shape so the pick-place script
# still makes sense. (The MuJoCo SO-101 asset doesn't resolve on every box.)
ROBOT_CONFIG_CANDIDATES = ["so101", "so100", "so_arm100", "panda"]

# Workspace layout (metres, world frame; arm base at origin).
# Cube: 3 cm placed head-on in front of the arm (+X, y=0) at x=0.30. The gripper
# FINGER axis is gripper_frame_link +Z (URDF gripper_frame_joint), and the planner
# now points THAT axis straight down for a true top-down grasp (measured ~3.5 deg
# from vertical here). x=0.30 is the sweet spot: close enough that a finger-down
# wrist is within joint limits, far enough that the arm can still reach the
# diagonal bin afterward (x>=0.32 tilts the grasp; x=0.34 makes the bin place
# infeasible once the grasp is finger-down). See planner notes (_FINGER_AXIS_TOOL).
# [0.30, 0.0] sits at the SO-101 5-DOF top-down reach limit (cuRobo reports
# "could not reach" and falls back to scripted). [0.20, 0.20] is within reach
# and is the pose the cuRobo pick-place is validated at (success_rate ~0.3-0.4).
#
# A normal cube (4 cm): the planner now aims the gripper FINGERTIP TCP (not the
# gripper_frame_link, which sits ~6 cm behind the fingers) at the cube, so the
# fingers actually reach it -- no need for a tall box. See planner.tcp_offset.
DEFAULT_CUBE_POSITION = [0.20, 0.20, 0.015]
DEFAULT_CUBE_HALF = [0.015, 0.015, 0.015]
DEFAULT_CUBE_COLOR = [0.85, 0.10, 0.10, 1.0]
# Bin pulled in from the old [0.0, 0.25] edge-of-reach spot. With the fingertip
# TCP grasp the arm ends the pick in a pose from which the far bin is
# unreachable; [0.12, 0.18] keeps the place reliable (3/3) and clearly separated
# from the cube at [0.20, 0.20].
DEFAULT_PLACE_POSITION = [0.12, 0.18, 0.0]
# Bin floor tile half-extents. The bin is an OPEN-TOP container (floor + walls,
# see _add_open_top_bin); this is the floor piece the cube rests ON, so its
# 2*half_z (= floor top) is what the collector uses as the cube's rest surface.
DEFAULT_BIN_HALF = [0.041, 0.041, 0.006]
DEFAULT_BIN_COLOR = [0.15, 0.55, 0.20, 1.0]


@dataclass
class SceneInfo:
    """What :func:`build_pick_place_scene` actually created."""

    robot_name: str
    robot_config: str
    joint_names: list[str]
    gripper_joint: str | None
    cube_name: str
    cube_position: list[float]
    place_position: list[float]
    cube_half: list[float] = field(default_factory=lambda: list(DEFAULT_CUBE_HALF))
    bin_half: list[float] = field(default_factory=lambda: list(DEFAULT_BIN_HALF))
    cameras: list[str] = field(default_factory=list)
    backend: str = "mujoco"
    actuated: bool = False

    def pretty(self) -> str:
        return (
            f"{self.robot_config} arm '{self.robot_name}' ({len(self.joint_names)} joints) "
            f"+ red cube at {[round(x, 2) for x in self.cube_position]} "
            f"+ bin at {[round(x, 2) for x in self.place_position]}; "
            f"cameras={self.cameras}; backend={self.backend}"
        )


def make_sim(backend: str = "mujoco", **isaac_kwargs: Any):
    """Return a ``SimEngine``-style simulation for the requested backend.

    ``mujoco`` (default) uses ``strands_robots.simulation.Simulation`` and loads
    a real SO-101. ``isaac`` lazily tries ``create_simulation("isaac")``; if the
    Isaac Sim runtime isn't installed this raises a clear, actionable error
    instead of a cryptic ImportError (the demo's app catches it and falls back
    to MuJoCo so the planning + collection loop is still demonstrable).
    """
    backend = (backend or "mujoco").lower()
    if backend in ("mujoco", "mj"):
        from strands_robots.simulation import Simulation

        return Simulation(tool_name="sim", mesh=False)

    if backend in ("isaac", "isaacsim", "isaac_sim"):
        # The Isaac backend now ships in-tree as the ``isaac`` builtin backend
        # (``strands_robots.simulation.isaac.simulation:IsaacSimulation``), so
        # ``create_simulation("isaac")`` resolves it directly on a stock
        # ``strands-robots[sim-isaac]`` install -- no manual ``register_backend``
        # is needed. Install the Isaac Sim runtime (Python 3.12 venv with
        # ``isaacsim``) via the NGC docker image or NVIDIA Omniverse Launcher.
        try:
            from strands_robots.simulation import create_simulation
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Isaac backend requested but create_simulation() is unavailable. "
                "Use backend='mujoco' (default), or install a strands-robots build "
                "that ships the SimEngine factory."
            ) from exc
        # Isaac Sim's Kit crashes during startup on some GPU/driver setups inside
        # ``omni.kit.raycast.query`` (a ray-query plugin we don't need for this
        # pick-place demo). SimulationApp reads Kit flags from ``sys.argv``, so
        # exclude that extension before the app boots. Verified: with it excluded
        # (+ a single NVIDIA Vulkan ICD via VK_ICD_FILENAMES) the kit boots
        # cleanly headless on this box. Idempotent.
        import sys as _sys

        _excl = "--/app/extensions/excluded/0=omni.kit.raycast.query"
        if _excl not in _sys.argv:
            _sys.argv.append(_excl)
        try:
            return create_simulation(
                "isaac",
                headless=isaac_kwargs.pop("headless", True),
                # Render with the fast rasterization pipeline so camera previews
                # and recorded episode videos are real frames (the default
                # "headless" render_mode returns blank frames -> no video in the
                # UI). Override via isaac_kwargs["render_mode"] or
                # STRANDS_ISAAC_RTX_PATHTRACING for photoreal.
                render_mode=isaac_kwargs.pop("render_mode", "rtx_realtime"),
                # Match rendering_dt to the physics step so EVERY ``step(render=
                # True)`` actually re-renders the cameras. The config default
                # (rendering_dt=1/30 vs physics_dt=0.002) only renders ~every 17
                # steps, so the recorded camera frames came out near-static
                # (and partly unrendered/black). 0.002 renders each step.
                rendering_dt=isaac_kwargs.pop("rendering_dt", 0.002),
                physics_dt=isaac_kwargs.pop("physics_dt", 0.002),
                **isaac_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - runtime missing / not wired
            raise RuntimeError(
                f"Could not create the Isaac Sim backend ({type(exc).__name__}: {exc}). "
                "The Isaac Sim runtime (Python 3.12 venv with `isaacsim`) is required; "
                "install it via the NGC docker image or NVIDIA Omniverse Launcher. "
                "Falling back to MuJoCo is recommended on boxes without it."
            ) from exc

    raise ValueError(f"Unknown backend {backend!r}. Use 'mujoco' or 'isaac'.")


def _status(result: Any) -> str:
    return str(result.get("status", "unknown")) if isinstance(result, dict) else "unknown"


def _add_robot_with_fallback(sim, name: str, candidates: list[str]) -> str:
    errors = []
    for cfg in candidates:
        if _status(sim.add_robot(name=name, data_config=cfg, position=[0.0, 0.0, 0.0])) == "success":
            if cfg != candidates[0]:
                logger.warning("Robot %r unavailable; fell back to %r.", candidates[0], cfg)
            return cfg
        errors.append(cfg)
    raise RuntimeError(
        f"Could not load any SO-101-class arm. Tried {candidates}. "
        "Install a MuJoCo Menagerie SO-101/SO-100 model or pass a resolvable config."
    )


def _add_fingertip_pads(sim, robot_name: str) -> bool:
    """Add small box collision pads at the gripper fingertips (blog / MuJoCo #239).

    The SO-101 gripper meshes make unstable single-point mesh-mesh contact, so a
    physically grasped cube jitters/slips. Adding small box primitives that
    protrude slightly inward from each finger gives stable multi-point contact
    for a real friction grip. Positions are in each finger body's frame (from the
    URDF geom placements); box ``size`` is full extents (the backend halves it).
    Best-effort: a no-op if the patch API or bodies are missing.
    """
    patch = getattr(sim, "patch_scene_mjcf", None)
    if not callable(patch):
        return False
    ns = f"{robot_name}/"
    ops = [
        {
            "op": "add_geom",
            "body": f"{ns}gripper_link",
            "type": "box",
            "size": [0.016, 0.016, 0.016],
            "pos": [-0.012, -0.0002, -0.075],
            "name": "static_finger_pad",
            "rgba": [1.0, 0.5, 0.5, 0.6],
        },
        {
            "op": "add_geom",
            "body": f"{ns}moving_jaw_so101_v1_link",
            "type": "box",
            "size": [0.016, 0.016, 0.016],
            "pos": [-0.0014, -0.05, 0.019],
            "name": "moving_finger_pad",
            "rgba": [0.5, 0.5, 1.0, 0.6],
        },
    ]
    try:
        res = sim.patch_scene_mjcf(ops)
        ok = _status(res) == "success"
        if ok:
            _set_pad_friction(sim, ["static_finger_pad", "moving_finger_pad"])
        logger.info("fingertip pads added: %s", ok)
        return ok
    except Exception:  # noqa: BLE001 - non-fatal; kinematic carry still works
        logger.debug("fingertip pad add failed (non-fatal)", exc_info=True)
        return False


def _set_pad_friction(sim, pad_names) -> None:
    """Set high tangential friction on the fingertip pads (best-effort)."""
    fn = getattr(sim, "set_geom_properties", None)
    if not callable(fn):
        return
    for name in pad_names:
        try:
            fn(name, friction=[4.0, 0.2, 0.05])
        except Exception:  # noqa: BLE001
            logger.debug("pad friction set failed for %s", name, exc_info=True)


def _actuate_arm(sim, robot_name: str) -> bool:
    """Make the SO-101 a FORCE-CONTROLLED arm for a physical (contact) grasp.

    The URDF loads with no actuators (nu=0) and was driven kinematically (teleport
    qpos), which is incompatible with real contact physics (the teleporting arm
    flings a collidable cube). This converts it to a position-servo arm so it can
    be driven via ``ctrl`` and grip the cube by friction:

    * position actuators on every arm joint + gripper (PD via gain + bias damping),
    * the stable ``implicitfast`` integrator + per-joint damping/armature (the bare
      URDF has none -> Euler blows up),
    * gravity compensation on the arm bodies (so modest gains track tightly),
    * **self-collision disabled between the arm's own links** -- the crucial fix:
      adjacent links (e.g. shoulder vs base) self-collide in MuJoCo and BLOCK the
      motion cuRobo planned (cuRobo ignores adjacent-link collisions), which made
      the base joint unable to rotate the arm to the cube. Fingertip pads, cube and
      ground keep colliding so the grasp still makes contact.

    Returns True if the arm became actuated (nu>0). Best-effort; on failure the
    caller keeps the kinematic path.
    """
    try:
        import mujoco

        w = getattr(sim, "_world", None)
        spec = w._backend_state.get("spec") if w is not None else None
        if spec is None:
            return False
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for b in spec.bodies:
            if b.name.startswith(f"{robot_name}/"):
                b.gravcomp = 1.0
        for j in spec.joints:
            if j.name.startswith(f"{robot_name}/"):
                j.damping[0] = 2.0
                j.armature = 0.01
        # Per-joint PD gains (bigger joints need more authority); bias[2] = kd for
        # ~critical damping. gripper range allows open(0.3)->close(-0.15).
        kps = {
            "shoulder_pan": 200.0,
            "shoulder_lift": 300.0,
            "elbow_flex": 200.0,
            "wrist_flex": 100.0,
            "wrist_roll": 60.0,
            "gripper": 600.0,
        }
        for jn, kp in kps.items():
            a = spec.add_actuator()
            a.name = f"{robot_name}_act_{jn}"
            a.target = f"{robot_name}/{jn}"
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0] = kp
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[1] = -kp
            a.biasprm[2] = -2.0 * (kp**0.5)
            a.ctrllimited = 1
            a.ctrlrange[0] = -3.2
            a.ctrlrange[1] = 3.2
        # Disable collision on the arm LINK geoms IN THE SPEC (so it survives this
        # and any later recompile), keeping fingertip pads + cube + ground
        # colliding. Adjacent links (base<->shoulder) otherwise self-collide and
        # block the planned base rotation (cuRobo ignores adjacent-link contacts).
        for gm in spec.geoms:
            nm = gm.name or ""
            if "pad" in nm:
                continue
            # spec geoms don't carry body name directly; match by the robot's mesh
            # geoms (all arm link visual/collision geoms). Pads are named; cube and
            # bin are separate bodies added via add_object (names cube_geom/bin).
            if nm in ("cube_geom", "bin_geom") or "cube" in nm or "bin" in nm or "ground" in nm:
                continue
            gm.contype = 0
            gm.conaffinity = 0
        ret = spec.recompile(w._model, w._data)
        if isinstance(ret, tuple) and len(ret) == 2:
            w._model, w._data = ret
        elif ret is not None:
            w._model = ret
        m = w._model
        # Also enforce on the live model (belt-and-suspenders for any geom the spec
        # name match missed); keep fingertip pads + cube + ground colliding.
        for g in range(m.ngeom):
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if bn.startswith(f"{robot_name}/") and "pad" not in gn:
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        logger.info("arm actuated: nu=%d (force-controlled, self-collision off)", m.nu)
        return m.nu > 0
    except Exception:  # noqa: BLE001 - non-fatal; kinematic path still works
        logger.debug("arm actuation failed (non-fatal)", exc_info=True)
        return False


def _enforce_grip_params(sim, robot_name: str) -> None:
    """Re-apply strong gripper force + high grasp friction on the LIVE model.

    Run AFTER all recompiles (robot, pads, cube/bin, cameras) -- each recompile
    resets live-model edits to the spec defaults. Sets a high gripper actuator
    gain so the jaw clamps hard, and high tangential friction on the fingertip
    pads + cube so the grip holds the cube through the lift and place. Without
    this the physical grasp slips and drops the cube.
    """
    try:
        import mujoco

        m = getattr(sim, "mj_model", None)
        if m is None:
            return
        gact = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot_name}_act_gripper")
        if gact >= 0:
            m.actuator_gainprm[gact][0] = 600.0
            m.actuator_biasprm[gact][1] = -600.0
            m.actuator_biasprm[gact][2] = -40.0
        for g in range(m.ngeom):
            gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if "pad" in gn or "cube" in (gn + bn).lower():
                m.geom_friction[g] = [4.0, 0.2, 0.05]
    except Exception:  # noqa: BLE001
        logger.debug("grip param enforcement failed (non-fatal)", exc_info=True)


def _add_open_top_bin(sim, place_position, is_isaac: bool) -> None:
    """Build an OPEN-TOP bin (floor tile + 4 short walls) at ``place_position``.

    A single flat plate was ambiguous: the cube appeared to sink into / sit
    beside it (the rendered plate vs the cube's placed XY didn't read as
    "on top"). An open-top container removes the ambiguity -- the cube is
    dropped INTO the cavity and is visibly contained by the walls, regardless
    of small placement offsets.

    Geometry (centered on ``place_position``): a square floor tile and four
    walls enclosing a ~7 cm inner cavity, comfortably larger than the 3 cm cube.
    Static, green. ``size`` is full-extents on Isaac (the backend converts;
    MuJoCo takes half-extents) -- matching the convention used by the caller.
    The primary floor piece keeps the name ``bin`` so the collector's
    place-distance check (which reads ``scene.place_position``) is unaffected.
    """
    px, py = float(place_position[0]), float(place_position[1])
    g = DEFAULT_BIN_COLOR
    inner = 0.035  # inner half-width of the cavity (7 cm cavity)
    t = 0.006  # wall / floor half-thickness
    wall_h = 0.018  # wall half-height (~3.6 cm tall walls)
    floor_top = 2.0 * t  # top surface of the floor tile (sits on the ground)

    def _box(name, pos, half):
        size = [2.0 * h for h in half] if is_isaac else list(half)
        r = sim.add_object(
            name=name,
            shape="box",
            position=pos,
            size=size,
            color=g,
            mass=1.0,
            is_static=True,
        )
        if _status(r) != "success":
            logger.info("bin part %r not added (non-fatal): %s", name, r)

    # Floor tile (the cube rests on this); keep the name "bin".
    _box("bin", [px, py, t], [inner + t, inner + t, t])
    # Four walls standing on the floor tile, enclosing the cavity.
    off = inner + t  # wall centerline distance from center
    wz = floor_top + wall_h  # wall center z
    _box("bin_wall_px", [px + off, py, wz], [t, inner + t, wall_h])
    _box("bin_wall_nx", [px - off, py, wz], [t, inner + t, wall_h])
    _box("bin_wall_py", [px, py + off, wz], [inner + t, t, wall_h])
    _box("bin_wall_ny", [px, py - off, wz], [inner + t, t, wall_h])
    logger.info("Open-top bin added at [%.2f, %.2f] (floor + 4 walls)", px, py)


def _add_soft_lighting(sim) -> bool:
    """Replace Isaac's hard default light with soft, low-shadow lighting.

    Isaac's ``add_default_ground_plane()`` ships a single hard distant light
    (1deg angular size), which casts a crisp, dark shadow under the red cube and
    the arm. This authors instead:

    * a bright ``DomeLight`` -- omnidirectional ambient fill that casts no hard
      shadow and lifts the dark undersides;
    * two WIDE-angle ``DistantLight``s (key + fill) from opposite directions. A
      large ``AngleAttr`` (the light's apparent angular size) gives a big
      penumbra -> soft, faint shadow edges instead of a hard dark blob; the fill
      from the opposite side further washes out the remaining shadow.

    The existing default light (under the ground-plane subtree) is dimmed so it
    doesn't re-introduce a hard shadow. Isaac-only, best-effort (no-op if the USD
    stage / ``pxr`` aren't available).
    """
    try:
        import omni.usd  # type: ignore[import-not-found]
        from pxr import Gf, Sdf, UsdGeom, UsdLux  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        logger.debug("soft lighting skipped (pxr/omni.usd unavailable)", exc_info=True)
        return False
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False

        # Bright omnidirectional fill -- the main shadow-softener.
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/SoftLights/dome"))
        dome.CreateIntensityAttr(1200.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        # Wide-angle key light (soft-edged shadow) from the front-top.
        key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SoftLights/key"))
        key.CreateIntensityAttr(1100.0)
        key.CreateAngleAttr(18.0)  # large angular size -> wide penumbra -> soft shadow
        key.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
        UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 20.0))

        # Wide-angle fill from the opposite side to wash out the remaining shadow.
        fill = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SoftLights/fill"))
        fill.CreateIntensityAttr(900.0)
        fill.CreateAngleAttr(25.0)
        fill.CreateColorAttr(Gf.Vec3f(0.95, 0.97, 1.0))
        UsdGeom.Xformable(fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 0.0, 200.0))

        # Dim Isaac's default hard distant light so it doesn't re-cast a sharp
        # shadow on top of the soft rig (best-effort -- name varies by version).
        for cand in (
            "/World/defaultGroundPlane/SphereLight",
            "/World/defaultGroundPlane/DistantLight",
            "/World/defaultDistantLight",
            "/World/DistantLight",
        ):
            p = stage.GetPrimAtPath(cand)
            if p and p.IsValid():
                try:
                    UsdLux.LightAPI(p).GetIntensityAttr().Set(150.0)
                except Exception:  # noqa: BLE001
                    try:
                        p.GetAttribute("inputs:intensity").Set(150.0)
                    except Exception:  # noqa: BLE001
                        pass
        logger.info("Isaac soft lighting added (dome + wide-angle key/fill; default light dimmed)")
        return True
    except Exception:  # noqa: BLE001
        logger.debug("soft lighting failed (non-fatal)", exc_info=True)
        return False


def _apply_isaac_grip_friction(sim, robot_name: str) -> bool:
    """Bind a high-friction PhysX material to the cube + gripper finger prims.

    The SO-101 physical grasp needs friction to hold the cube while lifting; the
    PhysX default material is too slippery (the cube squeezes out of the closed
    jaws). Creates one high-friction ``PhysicsMaterial`` and applies it to the
    cube and the gripper/moving-jaw collision prims via the USD stage. Isaac-only,
    best-effort.
    """
    try:
        from isaacsim.core.api.materials import PhysicsMaterial  # type: ignore[import-not-found]
        from isaacsim.core.utils.prims import get_prim_at_path  # type: ignore[import-not-found]
        from pxr import Usd, UsdShade  # type: ignore[import-not-found]

        mat = PhysicsMaterial(
            prim_path="/World/Looks/grip_friction",
            static_friction=3.5,
            dynamic_friction=3.5,
            restitution=0.0,
        )

        def _bind(prim_path: str) -> None:
            prim = get_prim_at_path(prim_path)
            if not prim or not prim.IsValid():
                return
            # Bind the material to every collision-bearing geom in the subtree.
            for p in Usd.PrimRange(prim):
                try:
                    UsdShade.MaterialBindingAPI(p).Bind(
                        UsdShade.Material(mat.prim),
                        UsdShade.Tokens.weakerThanDescendants,
                        "physics",
                    )
                except Exception:  # noqa: BLE001
                    pass

        _bind("/World/Objects/cube")
        # Gripper + moving jaw links under the robot prim.
        for link in ("gripper_link", "moving_jaw_so101_v1_link", "gripper_frame_link"):
            _bind(f"/World/Robots/{robot_name}/{link}")
        logger.info("Isaac grip friction material bound (cube + gripper)")
        return True
    except Exception:  # noqa: BLE001 - friction binding is best-effort
        logger.debug("Isaac grip friction binding failed (non-fatal)", exc_info=True)
        return False


def _erect_arm(sim, robot_name: str) -> bool:
    """Stand the arm in its model's ``home`` keyframe (zero pose sprawls flat).

    Sets qpos + actuator targets so the pose holds when stepped. MuJoCo-only;
    a no-op (returns False) on backends without ``mj_model``/``mj_data`` or a
    home keyframe.
    """
    try:
        import mujoco

        m = getattr(sim, "mj_model", None)
        d = getattr(sim, "mj_data", None)
        if m is None or d is None:
            return False
        key_id = next(
            (k for k in range(m.nkey) if "home" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) or "").lower()),
            -1,
        )
        if key_id < 0:
            return False
        kq = m.key_qpos[key_id]
        ns = f"{robot_name}/"
        # Joint types are compared BY VALUE: ``m.jnt_type[j]`` is a numpy integer and
        # ``x in (enum, ...)`` puts the enum on the left of ``==``, where a pybind11
        # enum does not match a numpy integer on every mujoco build.
        hinge_slide = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        for j in range(m.njnt):
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if jn.startswith(ns) and int(m.jnt_type[j]) in hinge_slide:
                d.qpos[m.jnt_qposadr[j]] = kq[m.jnt_qposadr[j]]
        for a in range(m.nu):
            jid = int(m.actuator_trnid[a, 0])
            if jid >= 0 and int(m.jnt_type[jid]) in hinge_slide:
                d.ctrl[a] = kq[m.jnt_qposadr[jid]]
        mujoco.mj_forward(m, d)
        return True
    except Exception:  # noqa: BLE001 - non-fatal pose nicety
        logger.debug("Could not set home pose.", exc_info=True)
        return False


def build_pick_place_scene(
    sim,
    cube_position: list[float] | None = None,
    place_position: list[float] | None = None,
    robot_candidates: list[str] | None = None,
    add_bin: bool = True,
    camera_size: tuple[int, int] = (640, 480),
    backend: str = "mujoco",
    robot_urdf: str | None = None,
) -> SceneInfo:
    """Populate ``sim`` with the SO-101 pick-and-place world. Returns a SceneInfo.

    Assumes a fresh ``sim`` (``create_world`` is called here). If ``robot_urdf``
    is given, the arm is loaded from that URDF so the sim shares the EXACT model
    cuRobo plans with (identical joint conventions + EE frame -> the plan
    executes correctly); otherwise a MuJoCo ``data_config`` SO-101 is used.
    """
    cube_position = list(cube_position or DEFAULT_CUBE_POSITION)
    place_position = list(place_position or DEFAULT_PLACE_POSITION)
    candidates = robot_candidates or ROBOT_CONFIG_CANDIDATES
    cw, ch = camera_size
    # Isaac's Kit renders RTX at a 16:9 render product (1280x720). A 4:3 camera
    # (640x480) captures that 16:9 render into a 4:3 buffer, leaving a hard black
    # band (the unrendered region). Use a 16:9 camera resolution on Isaac so the
    # whole frame is rendered (no black band).
    if backend in ("isaac", "isaacsim", "isaac_sim"):
        cw, ch = 640, 360

    cw_res = sim.create_world(timestep=0.002, gravity=[0.0, 0.0, -9.81], ground_plane=True)
    if _status(cw_res) != "success":
        raise RuntimeError(f"create_world failed: {cw_res}")

    # Soft lighting so the red cube + arm don't cast a harsh, dark shadow.
    # Isaac's default ground-plane rig is a single hard distant light (1deg
    # angular size -> crisp, dark shadows). Override it with a bright dome
    # (omnidirectional fill -- casts no hard shadow) plus WIDE-angle distant
    # lights (large AngleAttr -> big penumbra -> soft shadow edges) from two
    # directions so undersides are filled in. Isaac-only; best-effort.
    if backend in ("isaac", "isaacsim", "isaac_sim"):
        _add_soft_lighting(sim)

    if robot_urdf:
        rr = sim.add_robot(name="arm", urdf_path=robot_urdf, position=[0.0, 0.0, 0.0])
        if _status(rr) != "success":
            raise RuntimeError(f"add_robot(urdf_path={robot_urdf!r}) failed: {rr}")
        robot_config = "so101 (URDF, cuRobo-matched)"
        # MuJoCo physical (contact) grasp scaffolding: fingertip box pads + a
        # force-controlled position-servo arm. MuJoCo-ONLY -- it mutates the
        # MuJoCo MjSpec. The Isaac backend has its own PhysX articulation +
        # grasp path, so skip it there. Set SO101_PHYSICS_GRIP=0 to also skip on
        # MuJoCo (legacy kinematic carry).
        import os as _os

        arm_actuated = False
        is_mujoco = backend in ("mujoco", "mj")
        if is_mujoco and _os.environ.get("SO101_PHYSICS_GRIP", "1") != "0":
            _add_fingertip_pads(sim, robot_name="arm")
            arm_actuated = _actuate_arm(sim, robot_name="arm")
    else:
        robot_config = _add_robot_with_fallback(sim, name="arm", candidates=candidates)
        arm_actuated = False

    # Cube body type: the URDF/Isaac path drives a KINEMATIC grasp (the actuator-
    # less arm can't grip via friction, so the collector teleport-follows the cube
    # to the gripper). A free dynamic cube there only adds liability -- it drifts a
    # few mm each episode as the arm nudges it and occasionally gets flung, which
    # breaks multi-episode determinism. A static (FixedCuboid) cube still moves via
    # set_world_pose (so the kinematic carry works) but never drifts or flings, so
    # every episode resets to the identical pose. MuJoCo (dynamic actuated grasp)
    # keeps a dynamic cube.
    #
    # ``size`` convention divergence between backends: MuJoCo's ``add_object``
    # takes half-extents (matching ``mjcf`` ``geom size``); the in-tree
    # ``strands_robots.simulation.isaac`` backend takes full extents. Convert
    # here based on the backend so the physical cube/bin sizes match across
    # both runs.
    is_isaac = backend in ("isaac", "isaacsim", "isaac_sim")
    cube_size = [2.0 * h for h in DEFAULT_CUBE_HALF] if is_isaac else list(DEFAULT_CUBE_HALF)
    # Cube is DYNAMIC: the Isaac PhysX arm physically grips it (the gripper
    # closes on it with force and friction holds it during lift/place), so it
    # must be a free rigid body, not a static teleport-carried one. (MuJoCo also
    # uses a dynamic cube for its actuated grasp.)
    cube_static = False
    sim.add_object(
        name="cube",
        shape="box",
        position=cube_position,
        size=cube_size,
        color=DEFAULT_CUBE_COLOR,
        mass=0.04,
        is_static=cube_static,
    )
    # High tangential friction on the cube so the gripper pads hold it during the
    # lift (physical grasp). Best-effort.
    try:
        _fn = getattr(sim, "set_geom_properties", None)
        if callable(_fn):
            _fn("cube", friction=[4.0, 0.2, 0.05])
    except Exception:  # noqa: BLE001
        logger.debug("cube friction set failed (non-fatal)", exc_info=True)
    if add_bin:
        _add_open_top_bin(sim, place_position, is_isaac)

    # Isaac physical grasp: give the cube + gripper high friction so the closed
    # fingers actually hold the cube against gravity during the lift (PhysX
    # default material friction is too low -> the cube slips out).
    if is_isaac and robot_urdf:
        _apply_isaac_grip_friction(sim, robot_name="arm")

    cams = []
    # Camera rig framed on the actual workspace: arm at origin, cube at
    # [0.20, 0.20], bin at [0.0, 0.25]. Cameras sit CLOSE so the arm + cube +
    # bin fill the frame (the old rig sat ~1.5 m out aimed at a stale [0.30, 0]
    # cube, so everything looked tiny and off-center). The aim point is the
    # midpoint of the action (~[0.12, 0.18, 0.06]). FRONT: head-on from +X
    # Camera rig. Poses are pulled back and only mildly tilted: in headless
    # Isaac, steep/close camera angles produced static, half-black RTX frames
    # (validated -- a far, near-level camera renders full moving frames; a steep
    # oblique one does not). These viewpoints frame arm + cube [0.20,0.20] + bin
    # [0.12,0.18] while staying in the render-friendly regime.
    #
    # Each camera frames the WHOLE arm (base near [0,0]) plus the cube
    # ([0.20,0.20]) and bin ([0.12,0.18]). The action spans ~x[0,0.25] y[0,0.25],
    # so the front/topdown views aim at the mid-span (~[0.10,0.12]) and are
    # pulled back with a wider FOV -- earlier they aimed at the cube and only
    # caught the gripper end, cropping off the arm body/base.
    _all_specs = (
        ("front", [1.15, 0.10, 0.62], [0.06, 0.12, 0.06], 60.0),
        ("topdown", [0.10, 0.12, 1.35], [0.10, 0.12, 0.0], 62.0),
        ("oblique", [0.92, -0.55, 0.78], [0.13, 0.18, 0.05], 52.0),
    )
    _cam_specs = _all_specs
    for name, pos, tgt, fov in _cam_specs:
        if _status(sim.add_camera(name=name, position=pos, target=tgt, fov=fov, width=cw, height=ch)) == "success":
            cams.append(name)

    _erect_arm(sim, robot_name="arm")
    sim.step(20)  # settle into the home pose

    # Final enforcement (after ALL recompiles -- robot, pads, cube/bin, cameras):
    # re-apply the strong gripper force + high pad/cube friction directly on the
    # live model, because intervening recompiles reset live-model edits to the
    # spec defaults. This is what makes the physical grip actually hold the cube
    # through the lift+place (5/6) rather than slip (0/6).
    if arm_actuated:
        _enforce_grip_params(sim, robot_name="arm")

    jn = list(sim.robot_joint_names("arm"))
    gripper = jn[-1] if jn else None  # SO-101/SO-100: last joint is the gripper jaw
    info = SceneInfo(
        robot_name="arm",
        robot_config=robot_config,
        joint_names=jn,
        gripper_joint=gripper,
        cube_name="cube",
        cube_position=cube_position,
        place_position=place_position,
        cube_half=list(DEFAULT_CUBE_HALF),
        bin_half=list(DEFAULT_BIN_HALF),
        cameras=cams,
        backend=backend,
        actuated=bool(arm_actuated),
    )
    logger.info("SO-101 cuRobo scene ready: %s", info.pretty())
    return info
