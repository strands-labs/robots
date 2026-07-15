# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Motion planning for the SO-101 pick-and-place demo (issue #67).

Two planners, one interface — both emit a backend-agnostic
:class:`JointTrajectory` (a list of joint-target waypoints) that the collector
streams via ``send_action``:

* :class:`CuroboMotionPlanner` — the real, collision-aware path (issue #67
  T4/T5). It **lazy-imports cuRobo**; cuRobo isn't installed on most boxes (and
  the latest release wants driver >= 580 vs our 550), so every entry point
  raises a clear, actionable install hint rather than a cryptic ImportError.
* :class:`ScriptedPlanner` — the dependency-free fallback (#67 graceful
  degradation). It interpolates a joint-space pick -> lift -> place -> release
  demonstration so the planning + collection loop is fully demonstrable on the
  MuJoCo backend today, without cuRobo/IK.

:func:`make_planner` returns cuRobo when available, else the scripted planner.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger("so101_curobo.planner")

CUROBO_INSTALL_HINT = (
    "cuRobo is not installed. VALIDATED install on driver 550 / CUDA 12.4 / L4 "
    "(issue #67 T3 -- the docs' driver>=580 is conservative; CUDA 12.x kernels "
    "run on a 12.4 driver):\n"
    "  export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9\n"
    "  python -m venv --system-site-packages .venv && source .venv/bin/activate\n"
    "  pip install -U pip setuptools wheel ninja\n"
    "  git clone --depth 1 https://github.com/NVlabs/curobo && cd curobo\n"
    "  sed -i '/Topic :: Scientific.Engineering :: Robotics/d' pyproject.toml\n"
    "  pip install -e . --no-build-isolation\n"
    "  pip install 'cuda-core[cu12]'   # runtime kernel backend (required by the\n"
    "                                  #  refactored cuRobo; no precompile needed)\n"
    "Until installed, the demo uses the ScriptedPlanner fallback."
)


def curobo_available() -> bool:
    """True if the ``curobo`` package is importable."""
    try:
        return importlib.util.find_spec("curobo") is not None
    except Exception:  # noqa: BLE001
        return False


def _so101_cache_urdf() -> tuple[str | None, str]:
    """Best-effort: resolve the SO-101 URDF + mesh dir from the strands-robots cache.

    The default MuJoCo demo already auto-downloads the SO-101 model into the
    external ``~/.strands_robots`` cache (via ``strands_robots.assets``), and
    that download ships a URDF (``so101_new_calib.urdf``) and a ``assets/``
    mesh dir alongside the MJCF. The cuRobo / URDF-sim paths previously
    *ignored* that and forced the user to supply ``--curobo-urdf`` /
    ``SO101_URDF`` by hand even though a perfectly good URDF was already on
    disk. This resolves that cached URDF so the cuRobo path works
    out-of-the-box on the same machine the MuJoCo demo already ran on.

    Returns ``(urdf_path, asset_path)``; ``urdf_path`` is ``None`` when the
    cache / package isn't available (so callers keep their existing
    "raise an actionable error" behaviour). ``resolve_model_dir`` triggers the
    same auto-download the MuJoCo path uses, so a clean box with internet
    populates the cache here too.
    """
    try:
        import glob

        from strands_robots.assets import resolve_model_dir  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None, ""
    try:
        model_dir = resolve_model_dir("so101")
    except Exception:  # noqa: BLE001
        return None, ""
    if not model_dir:
        return None, ""
    model_dir = str(model_dir)
    # Prefer the new-calibration URDF; fall back to any *.urdf in the dir.
    preferred = os.path.join(model_dir, "so101_new_calib.urdf")
    if os.path.exists(preferred):
        urdf = preferred
    else:
        matches = sorted(glob.glob(os.path.join(model_dir, "*.urdf")))
        urdf = matches[0] if matches else None
    if not urdf:
        return None, ""
    # Meshes live in the cache dir's ``assets/`` subdir when present; cuRobo's
    # RobotBuilder resolves ``package://`` / relative mesh refs against it.
    asset_dir = os.path.join(model_dir, "assets")
    asset_path = asset_dir if os.path.isdir(asset_dir) else model_dir
    return urdf, asset_path


def resolve_so101_urdf(urdf_path: str | None = None) -> str | None:
    """Resolve the SO-101 URDF with the documented precedence.

    1. explicit ``urdf_path`` argument,
    2. the ``SO101_URDF`` env var,
    3. the auto-downloaded ``strands_robots`` SO-101 cache URDF
       (see :func:`_so101_cache_urdf`).

    Returns ``None`` only when none of the three resolve, so callers keep
    their existing fail-with-hint behaviour for a box that has neither an
    explicit URDF nor the cache.
    """
    if urdf_path:
        return urdf_path
    env = os.environ.get("SO101_URDF")
    if env:
        return env
    cached, _ = _so101_cache_urdf()
    return cached


def resolve_so101_asset(asset_path: str = "") -> str:
    """Resolve the SO-101 mesh dir (for cuRobo) with the same precedence.

    explicit ``asset_path`` -> ``SO101_ASSET`` env -> the cache mesh dir.
    Returns ``""`` when none resolve (cuRobo treats that as "no extra mesh
    search path", unchanged from before).
    """
    if asset_path:
        return asset_path
    env = os.environ.get("SO101_ASSET")
    if env:
        return env
    _, cached_assets = _so101_cache_urdf()
    return cached_assets


CUROBO_AVAILABLE = curobo_available()

# The SO-101 gripper's fingers extend along the gripper_frame_link +Z axis. From
# the URDF gripper_frame_joint (gripper_link -> gripper_frame_link, xyz=
# (-0.0079, -0.0002, -0.0981) rpy=(0, pi, 0)), the wrist->tool-point offset
# expressed in the tool frame is ~[0.08, 0, 0.997] -- i.e. essentially +Z. The
# earlier code assumed the +x axis was the approach axis; that was wrong and
# produced a horizontal side-grasp (the tool +x can point straight down while the
# fingers point sideways). Measure/aim THIS axis for a true top-down grasp.
_FINGER_AXIS_TOOL = (0.08, -0.002, 0.997)


def _approach_angle_deg(quat_wxyz: Sequence[float]) -> float:
    """Angle in degrees between the gripper's FINGER axis and straight down (-Z).

    The SO-101 fingers extend along ``gripper_frame_link`` +Z (``_FINGER_AXIS_TOOL``;
    see the URDF gripper_frame_joint), NOT +x. For a unit quaternion (w, x, y, z)
    the world Z-component of a tool-frame vector ``f`` is
    ``R[2,:] . f = (2(xz-wy), 2(yz+wx), 1-2(x^2+y^2)) . f``; the finger axis angle
    from straight-down (0, 0, -1) is ``acos(-worldZ)``. Returns 0 for fingers
    pointing straight down (true top-down grasp), 90 for a horizontal side-grasp.
    """
    w, x, y, z = quat_wxyz
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    fx, fy, fz = _FINGER_AXIS_TOOL
    fn = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    world_z = (2.0 * (x * z - w * y)) * fx + (2.0 * (y * z + w * x)) * fy + (1.0 - 2.0 * (x * x + y * y)) * fz
    return math.degrees(math.acos(max(-1.0, min(1.0, -world_z / fn))))


@dataclass
class JointTrajectory:
    """A planned joint-space trajectory: dense ``{joint_name: target}`` waypoints."""

    joint_names: list[str]
    waypoints: list[dict[str, float]]
    phases: list[str]
    planner: str = "scripted"
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.waypoints)

    def phase_of(self, i: int) -> str:
        return self.phases[i] if 0 <= i < len(self.phases) else ""


def _merge(base: dict[str, float], updates: dict[str, float]) -> dict[str, float]:
    out = dict(base)
    for k, v in updates.items():
        if k is not None:
            out[k] = v
    return out


class PrecomputedPlanner:
    """Replay a trajectory planned **offline** (e.g. by cuRobo) from a JSON file.

    cuRobo and Isaac Sim 4.5 can't share a process: cuRobo's collision kernels
    need ``warp-lang >= 1.14`` (``wp.func(module=)``) while the Isaac kit bundles
    warp 1.5 which lacks it, so an in-kit ``import curobo`` collision path raises
    and the demo silently falls back to the scripted planner. To still execute a
    *real* cuRobo plan on the Isaac backend, plan offline (``plan_curobo_offline.py``
    in a cuRobo-capable venv), dump the :class:`JointTrajectory` to JSON, and
    replay it here -- this class imports neither cuRobo nor warp, so it runs
    inside the kit.

    The file is resolved from ``traj_path`` or the ``SO101_CUROBO_TRAJ`` env var.
    ``plan_pick_place`` ignores cube/place args (the geometry is baked into the
    saved plan) but validates them against the plan's recorded ``plan_for`` so a
    mismatched scene fails loudly instead of replaying a stale trajectory.
    """

    name = "precomputed"

    def __init__(self, traj_path: str | None = None, **_ignored):
        self.traj_path = traj_path or os.environ.get("SO101_CUROBO_TRAJ")

    def available(self) -> bool:
        return bool(self.traj_path and os.path.exists(self.traj_path))

    def plan_pick_place(
        self,
        joint_names: Sequence[str],
        start_q: Sequence[float],
        gripper_joint: str | None = None,
        cube_xy: Sequence[float] | None = None,
        place_xy: Sequence[float] | None = None,
        **_ignored,
    ) -> JointTrajectory:
        import json

        if not self.available():
            raise RuntimeError(
                "PrecomputedPlanner needs a saved trajectory JSON. Pass traj_path=... or set "
                "SO101_CUROBO_TRAJ. Generate one with examples/so101_curobo/plan_curobo_offline.py "
                "in a cuRobo-capable venv."
            )
        with open(self.traj_path) as f:
            d = json.load(f)
        wps = [dict(wp) for wp in d["waypoints"]]
        if not wps:
            raise RuntimeError(f"Precomputed trajectory {self.traj_path!r} has no waypoints.")
        # Sanity-check the plan targets the scene we're about to execute (so a
        # stale file isn't replayed against a moved cube). Tolerate small float
        # drift; only error on a real mismatch.
        #
        # Only the PICK (cube_xy) is validated. The place_xy in the plan is a
        # *control target*, not the bin's location: the 5-DOF arm can't reach the
        # full fingertip-TCP place pose, so the place segment falls back to a
        # heuristic interpolation that systematically lands the carried cube
        # ~10 cm in +X of the commanded target. The offline plan therefore aims
        # the place target OFF the bin (e.g. x=0.02 to land the cube on a bin at
        # x=0.12) to compensate. Validating place_xy against the scene's bin
        # position would wrongly reject this intentional offset, so skip it.
        pf = d.get("plan_for") or {}
        for label, want, key in (("cube", cube_xy, "cube_xy"),):
            have = pf.get(key)
            if want is not None and have is not None:
                if max(abs(float(a) - float(b)) for a, b in zip(want[:2], have[:2])) > 1e-3:
                    raise RuntimeError(
                        f"Precomputed trajectory was planned for {label}_xy={have} but the scene has "
                        f"{list(want[:2])}. Re-plan with plan_curobo_offline.py for this scene."
                    )
        phases = list(d.get("phases") or [""] * len(wps))
        if len(phases) < len(wps):
            phases += [phases[-1] if phases else ""] * (len(wps) - len(phases))
        # Deepen the gripper CLAMP on the gripping phases. The offline plan closes
        # to gripper_close (~-0.15), which on the force-driven Isaac arm leaves the
        # grip *marginal* -- it holds the cube on some PhysX contact rolls and
        # slips on others (non-deterministic max-lift z), so a 5-episode batch
        # scores ~0-20% even though a single run can succeed. Driving the gripper
        # target to the joint's hard min (~-0.174) squeezes the jaws fully closed,
        # generating consistent normal force so the friction grip holds every
        # episode. Tunable via SO101_GRIP_CLOSE; applied only to the closed
        # phases (close/lift/place/place_down) so reach/grasp still open to clear
        # the cube on descent.
        grip_jn = gripper_joint or (list(d.get("joint_names") or joint_names) or [None])[-1]
        try:
            deep_close = float(os.environ.get("SO101_GRIP_CLOSE", "-0.174"))
        except ValueError:
            deep_close = -0.174
        if grip_jn is not None:
            closed_phases = {"close", "lift", "place", "place_down"}
            for wp, ph in zip(wps, phases):
                if ph in closed_phases and grip_jn in wp:
                    # Only deepen (never weaken) the existing closed target.
                    wp[grip_jn] = min(float(wp[grip_jn]), deep_close)
        return JointTrajectory(
            joint_names=list(d.get("joint_names") or joint_names),
            waypoints=wps,
            phases=phases,
            planner=f"precomputed({d.get('planner', 'curobo')})",
            meta=dict(d.get("meta") or {}),
        )


class ScriptedPlanner:
    """Dependency-free joint-space pick-and-place demonstration.

    No IK: it sweeps the arm through scripted keyframes (aim -> reach -> grasp
    -> lift -> traverse -> lower -> release) using heuristic joint deltas from
    the home pose, then linearly interpolates. Produces coherent, recordable
    motion; it is **not** guaranteed to achieve a physical grasp (that's what
    the cuRobo path is for). Success is judged empirically by the collector.
    """

    name = "scripted"

    def __init__(self, gripper_open: float = 0.0, gripper_close: float = 1.5, steps_per_phase: int = 8):
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.steps_per_phase = steps_per_phase

    def plan_pick_place(
        self,
        joint_names: Sequence[str],
        start_q: Sequence[float],
        gripper_joint: str | None = None,
        cube_xy: Sequence[float] | None = None,
        place_xy: Sequence[float] | None = None,
        steps_per_phase: int | None = None,
        base_sign: float = 1.0,
    ) -> JointTrajectory:
        jn = list(joint_names)
        n = len(jn)
        steps = steps_per_phase or self.steps_per_phase
        home = {j: float(q) for j, q in zip(jn, start_q)}

        base = jn[0] if n > 0 else None
        lift = jn[1] if n > 1 else None
        elbow = jn[2] if n > 2 else None
        wristf = jn[3] if n > 3 else None
        grip = gripper_joint or (jn[5] if n > 5 else (jn[-1] if jn else None))

        OPEN, CLOSE = self.gripper_open, self.gripper_close

        def aim(xy) -> float:
            # Base-pan angle to face a world XY target. ``base_sign`` flips the
            # joint sign for robots whose shoulder_pan rotates opposite to world
            # +Z (the SO-101 URDF on Isaac: commanding +pan swings the arm to
            # -Y, so a +Y target needs a negative pan). MuJoCo's model uses the
            # default +1 convention. See controller._plan.
            if not (base and xy):
                return home.get(base, 0.0) if base else 0.0
            return float(base_sign * math.atan2(xy[1], xy[0]))

        reach = dict(home)
        if base:
            reach[base] = aim(cube_xy)
        if lift:
            reach[lift] = home[lift] - 0.8
        if elbow:
            reach[elbow] = home[elbow] + 1.0
        if wristf:
            reach[wristf] = home[wristf] + 0.5
        if grip:
            reach[grip] = OPEN

        liftq = _merge(reach, {grip: CLOSE})
        if lift:
            liftq[lift] = home[lift] - 0.2
        if elbow:
            liftq[elbow] = home[elbow] + 0.4

        traverse = _merge(liftq, {base: aim(place_xy)})
        lower = dict(traverse)
        if lift:
            lower[lift] = home[lift] - 0.6
        if elbow:
            lower[elbow] = home[elbow] + 0.8

        keyframes = [
            ("home", _merge(home, {grip: OPEN})),
            ("aim_pick", _merge(home, {base: aim(cube_xy), grip: OPEN})),
            ("reach", reach),
            ("grasp", _merge(reach, {grip: CLOSE})),
            ("lift", liftq),
            ("traverse", traverse),
            ("lower", lower),
            ("release", _merge(lower, {grip: OPEN})),
            ("retreat", _merge(home, {base: aim(place_xy), grip: OPEN})),
        ]

        waypoints: list[dict[str, float]] = []
        phases: list[str] = []
        for (_, q0), (p1, q1) in zip(keyframes[:-1], keyframes[1:]):
            for s in range(1, steps + 1):
                t = s / steps
                waypoints.append({j: (1.0 - t) * q0.get(j, home[j]) + t * q1.get(j, home[j]) for j in jn})
                phases.append(p1)

        return JointTrajectory(
            joint_names=jn,
            waypoints=waypoints,
            phases=phases,
            planner=self.name,
            meta={"gripper_joint": grip, "cube_xy": list(cube_xy or []), "place_xy": list(place_xy or [])},
        )


class CuroboMotionPlanner:
    """cuRobo collision-aware motion planner for the SO-101 (issue #67 T4/T5).

    Builds the SO-101 kinematic model from a URDF with cuRobo's new
    ``RobotBuilder`` (auto-derives the 5-DOF arm chain to the tool frame,
    excluding the gripper jaw), then plans EEF pose-to-pose trajectories with
    ``MotionPlanner`` and chains them into reach -> grasp -> lift -> place ->
    release. The arm is collision-aware; the gripper is scripted. Validated on
    driver 550 / CUDA 12.4 / L4 (see CUROBO_INSTALL_HINT).

    Lazy: imports cuRobo only in :meth:`_ensure`. Needs an SO-101 URDF
    (``urdf_path`` or env ``SO101_URDF``, plus meshes via ``asset_path`` /
    ``SO101_ASSET``); otherwise raises an actionable error so callers fall back
    to :class:`ScriptedPlanner`.
    """

    name = "curobo"

    def __init__(
        self,
        urdf_path: str | None = None,
        asset_path: str = "",
        tool_frame: str = "gripper_frame_link",
        self_collision: bool = False,
        device: str = "cuda",
        grasp_quaternion: Sequence[float] | None = None,
        position_only: bool = True,
        top_down_grasp: bool = True,
        top_down_weight: float = 0.05,
        orientation_tolerance: float = 1.6,
        top_down_attempts: int = 6,
        gripper_open: float = 0.3,
        gripper_close: float = -0.15,
        grasp_z: float = 0.05,
        fingertip_offset: Sequence[float] | None = None,
        **_ignored,
    ):
        self.urdf_path = resolve_so101_urdf(urdf_path)
        self.asset_path = resolve_so101_asset(asset_path)
        self.tool_frame = tool_frame
        self.self_collision = self_collision
        self.device = device
        self.grasp_quaternion = list(grasp_quaternion) if grasp_quaternion else None
        # The SO-101 is a 5-DOF arm and cannot achieve arbitrary 6-DOF
        # orientations, so a fully-constrained pose goal is usually infeasible.
        # Default to POSITION-ONLY tracking (orientation is left free) so the
        # arm can actually reach tabletop pick/place positions.
        self.position_only = position_only
        # Top-down grasp (issue #67 T5 "further-tuning path"): a *soft* downward
        # orientation bias on the pick/approach/lift segments so the gripper
        # descends onto the cube near-vertically instead of grazing it sideways.
        # Strict vertical is infeasible for this 5-DOF arm and a high weight
        # makes IK finicky; calibration on the SO-101 URDF (driver 550 / L4)
        # found rpy weight ~0.05 + a relaxed orientation success tolerance
        # (~1.6 rad) + best-of-N attempts yields a consistently near-vertical
        # (<= ~13 deg from straight down) grasp at 8/8 cube targets, vs ~84 deg
        # (horizontal, high-variance) with free orientation. The place segments
        # stay position-only (the bin pose is not vertical-reachable, and a
        # top-down drop into a bin is not required). Unreachable oriented solves
        # fall back to position-only, so this never regresses below the
        # position-only path.
        self.top_down_grasp = top_down_grasp
        self.top_down_weight = float(top_down_weight)
        self.orientation_tolerance = float(orientation_tolerance)
        self.top_down_attempts = max(1, int(top_down_attempts))
        # Gripper joint targets (revolute, range ~[-0.174, 1.745]). IMPORTANT:
        # the jaw CLOSES toward the LOW/negative end and OPENS toward the high end
        # (confirmed: commanding 1.74 swings the jaw ~90 deg wide open). So the
        # grip value must be LOW and the approach-open value just above it:
        #   gripper_close ~ -0.15  -> jaws together to clamp the cube,
        #   gripper_open  ~  0.3   -> jaws open just enough to clear the 3 cm cube
        #                             on the top-down descent (the old code had
        #                             these inverted -- close=1.74 -> jaws flew open).
        # (Values are approximate; fine-tune against a render of the grip.)
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.grasp_z_cfg = float(grasp_z)
        # TCP (fingertip) offset expressed in the cuRobo tool frame
        # (gripper_frame_link). The SO-101 URDF's gripper_frame_link is the
        # "graspframe" ~6 cm BEHIND the fingertips along its -Z axis (plus a small
        # -X for the one-fixed-one-moving jaw asymmetry); the actual fingertip TCP
        # (where a grasped object sits) is here. cuRobo's GoalToolPose drives the
        # *tool frame* to the goal, so to put the FINGERTIPS on the target we
        # offset the goal by R_goal @ tcp_offset (the tool-local offset rotated
        # into the world). Measured from the jaw geoms; matches the ggando.com
        # SO-101 blog's gripperframe-vs-graspframe distinction.
        self.tcp_offset = [0.0, 0.0, -0.04]
        self.fingertip_offset = list(fingertip_offset) if fingertip_offset else [0.0, 0.0, 0.0]
        self._planner = None
        self._arm_joint_names: list[str] = []
        self._default_quat: list[float] | None = None

    @staticmethod
    def available() -> bool:
        return curobo_available()

    def _ensure(self):
        if not self.available():
            raise RuntimeError(CUROBO_INSTALL_HINT)
        if self._planner is not None:
            return
        import os
        import tempfile

        import yaml

        if not self.urdf_path or not os.path.exists(self.urdf_path):
            raise RuntimeError(
                "CuroboMotionPlanner needs an SO-101 URDF. It auto-resolves the "
                "strands-robots SO-101 cache URDF (the one the MuJoCo demo "
                "downloads), but that wasn't found here -- pass urdf_path=... or "
                "set SO101_URDF (+ SO101_ASSET for meshes). See README (#67 T2/T4)."
            )
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.robot_builder import RobotBuilder
        from curobo.types import JointState

        builder = RobotBuilder(urdf_path=self.urdf_path, asset_path=self.asset_path, tool_frames=[self.tool_frame])
        tmp_yml = os.path.join(tempfile.mkdtemp(prefix="curobo_so101_"), "so101_curobo.yml")
        builder.save(builder.build(), tmp_yml, include_cspace=True)
        with open(tmp_yml) as f:
            robot = yaml.safe_load(f)
        # When top-down grasping is on, relax the *success* orientation tolerance
        # so the solver accepts the most-vertical pose the 5-DOF arm can reach
        # (strict vertical is infeasible). Position-only segments set the
        # orientation weight to 0, so this relaxation does not affect them
        # (verified: free-orientation reach is identical at tol 0.05 vs 1.6).
        ori_tol = self.orientation_tolerance if self.top_down_grasp else 0.05
        self._planner = MotionPlanner(
            MotionPlannerCfg.create(
                robot=robot,
                self_collision_check=self.self_collision,
                use_cuda_graph=False,
                orientation_tolerance=ori_tol,
            )
        )
        self._arm_joint_names = list(self._planner.joint_names)
        # 5-DOF arm: default to POSITION-ONLY tracking so tabletop targets are
        # reachable (orientation free; a fully-constrained 6-DOF goal is
        # infeasible). This is also the per-segment fallback and the criteria
        # used for the place segments; the pick/approach/lift segments override
        # it with a soft top-down orientation bias in :meth:`_plan_segment`.
        if self.position_only or self.top_down_grasp:
            from curobo.types import ToolPoseCriteria

            self._planner.update_tool_pose_criteria(
                {self.tool_frame: ToolPoseCriteria.track_position(xyz=[1.0, 1.0, 1.0])}
            )
        # Grasp orientation target: point the gripper's FINGER axis straight DOWN.
        # The fingers extend along gripper_frame_link +Z (_FINGER_AXIS_TOOL). A
        # 180-deg rotation about the base X axis maps the tool's +Z to world -Z
        # (fingers down) with +x facing forward, so it is the top-down grasp target.
        # (The previous code used the home EE quaternion, which aligns the tool +x
        # with -Z -> a horizontal side-grasp, because +x is NOT the finger axis.)
        # cuRobo biases toward this softly (the 5-DOF arm can't hit it exactly) and
        # best-of-N keeps the most finger-vertical solve. Override via grasp_quaternion.
        self._default_quat = [0.0, 1.0, 0.0, 0.0]
        # Log the home tool pose too (confirms FK/kinematics is live).
        q0 = self._planner.default_joint_state.position
        q0 = q0.unsqueeze(0) if q0.dim() == 1 else q0
        st = self._planner.compute_kinematics(JointState.from_position(q0, joint_names=self._arm_joint_names))
        tp = st.tool_poses
        if isinstance(tp, dict):
            tp = tp.get(self.tool_frame, list(tp.values())[0])
        home_quat = [round(float(x), 3) for x in tp.quaternion.reshape(-1, 4)[0].detach().cpu().tolist()]
        logger.info(
            "cuRobo SO-101 planner ready: arm joints=%s home_quat=%s grasp_quat(finger-down)=%s",
            self._arm_joint_names,
            home_quat,
            self._default_quat,
        )

    @staticmethod
    def _rotate_by_quat(v, q_wxyz):
        """Rotate 3-vector ``v`` by unit quaternion ``q`` (w, x, y, z)."""
        w, x, y, z = q_wxyz
        vx, vy, vz = v
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        return [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ]

    def _heuristic_place_arm(self, cur_arm, place_xy, home_arm) -> list[float]:
        """A best-effort joint-space place pose when cuRobo can't solve the bin.

        Pan the base (joint 0) to face the bin XY and partly retract the
        shoulder/elbow toward home so the carried cube is lifted clear of the
        ground and brought OVER the bin (pure base-pan kept the arm folded low,
        dragging/flinging the cube). Blended modestly to avoid a big posture jump.
        Used only as a graceful continuation of the real cuRobo pick so the whole
        episode isn't thrown to the scripted fallback.
        """
        import math

        out = list(cur_arm)
        if len(out) >= 1 and place_xy:
            out[0] = float(math.atan2(place_xy[1], place_xy[0]))
        # Retract shoulder (j1) and elbow (j2) partway toward home so the held
        # cube rises and the arm reaches outward over the bin rather than staying
        # folded over the pick spot.
        for j in (1, 2):
            if len(out) > j and j < len(home_arm):
                out[j] = 0.65 * home_arm[j] + 0.35 * out[j]
        return out

    def _plan_segment(
        self, start_arm_q, goal_xyz, goal_quat, orient: bool = False, apply_tcp: bool = True
    ) -> list[list[float]]:
        import torch
        from curobo.types import GoalToolPose, JointState, ToolPoseCriteria

        jn = self._arm_joint_names
        start = JointState.from_position(
            torch.tensor([start_arm_q], dtype=torch.float32, device=self.device), joint_names=jn
        )
        # Offset the goal so the FINGERTIP TCP (not the gripper_frame_link origin,
        # which sits ~6 cm behind the fingers) lands at goal_xyz. The tool-local
        # tcp_offset is rotated by the goal orientation into the world and
        # subtracted: tool_goal = goal - R(goal_quat) @ tcp_offset. This makes
        # reach/grasp/lift/place all track the fingertips consistently (the
        # gripperframe-vs-graspframe fix from the SO-101 RL blog).
        gx = list(goal_xyz)
        if apply_tcp and any(self.tcp_offset):
            world_off = self._rotate_by_quat(self.tcp_offset, goal_quat)
            gx = [goal_xyz[i] - world_off[i] for i in range(3)]
        pos = torch.tensor(gx, dtype=torch.float32, device=self.device).reshape(1, 1, 1, 1, 3)
        quat = torch.tensor(goal_quat, dtype=torch.float32, device=self.device).reshape(1, 1, 1, 1, 4)
        goal = GoalToolPose(tool_frames=[self.tool_frame], position=pos, quaternion=quat)

        def _solve():
            res = self._planner.plan_pose(goal, start)
            if res is None or not bool(res.success.any()):
                return None
            return res.get_interpolated_plan().position.reshape(-1, len(jn)).detach().cpu().tolist()

        # Top-down segments: bias orientation toward the (downward) goal quat with
        # a soft weight, and keep the most-vertical of N attempts (a single solve
        # varies ~10-86 deg from vertical due to cuRobo nondeterminism; best-of-N
        # reliably lands <= ~13 deg). Restore position-only afterwards so the
        # place segments and any fallback are unconstrained in orientation.
        if orient and self.top_down_grasp:
            w = self.top_down_weight
            self._planner.update_tool_pose_criteria(
                {self.tool_frame: ToolPoseCriteria.track_position_and_orientation(xyz=[1.0, 1.0, 1.0], rpy=[w, w, w])}
            )
            best_traj, best_err = None, None
            try:
                for _ in range(self.top_down_attempts):
                    traj = _solve()
                    if traj is None:
                        continue
                    err = self._approach_deg(traj[-1])
                    if best_err is None or err < best_err:
                        best_traj, best_err = traj, err
            finally:
                self._planner.update_tool_pose_criteria(
                    {self.tool_frame: ToolPoseCriteria.track_position(xyz=[1.0, 1.0, 1.0])}
                )
            if best_traj is not None:
                logger.info(
                    "cuRobo top-down reach %s: %.1f deg from vertical (best of %d)",
                    [round(float(x), 3) for x in goal_xyz],
                    best_err,
                    self.top_down_attempts,
                )
                return best_traj
            logger.info(
                "cuRobo top-down unreachable at %s; falling back to position-only",
                [round(float(x), 3) for x in goal_xyz],
            )

        traj = _solve()
        if traj is None:
            raise RuntimeError(f"cuRobo could not reach {[round(float(x), 3) for x in goal_xyz]}")
        return traj

    def _approach_deg(self, arm_q) -> float:
        """Angle (deg) of the tool approach axis from straight down (world -Z).

        The SO-101 ``gripper_frame_link`` local +x is the grasp approach axis
        (FK-verified: at home it maps to world -Z). 0 deg == perfectly top-down,
        90 deg == horizontal.
        """
        import torch
        from curobo.types import JointState

        q = torch.tensor([arm_q], dtype=torch.float32, device=self.device)
        st = self._planner.compute_kinematics(JointState.from_position(q, joint_names=self._arm_joint_names))
        tp = st.tool_poses
        if isinstance(tp, dict):
            tp = tp.get(self.tool_frame, list(tp.values())[0])
        quat = [float(v) for v in tp.quaternion.reshape(-1, 4)[0].detach().cpu().tolist()]
        return _approach_angle_deg(quat)

    def plan_pick_place(
        self,
        joint_names: Sequence[str],
        start_q: Sequence[float],
        gripper_joint: str | None = None,
        cube_xy: Sequence[float] | None = None,
        place_xy: Sequence[float] | None = None,
        table_z: float = 0.0,
        approach: float = 0.12,
        grasp_z: float | None = None,
        hold_steps: int = 5,
        **_ignored,
    ) -> JointTrajectory:
        """Collision-aware reach -> grasp -> lift -> place -> release.

        cuRobo plans the arm DOFs (mapped to ``joint_names`` by kinematic order);
        the trailing gripper joint is opened/closed per phase. Raises
        ``RuntimeError`` (caught by the caller -> ScriptedPlanner fallback) if a
        segment is unreachable.
        """
        self._ensure()
        if grasp_z is None:
            grasp_z = self.grasp_z_cfg
        jn = list(joint_names)
        n_arm = len(self._arm_joint_names)
        if len(jn) < n_arm:
            raise RuntimeError(f"scene has {len(jn)} joints; SO-101 arm needs {n_arm}.")
        arm_names = jn[:n_arm]
        grip = gripper_joint or (jn[-1] if len(jn) > n_arm else None)
        start_arm = [float(x) for x in start_q[:n_arm]]
        quat = self.grasp_quaternion or self._default_quat
        cx, cy = (list(cube_xy) if cube_xy else [0.2, 0.2])[:2]
        px, py = (list(place_xy) if place_xy else [-0.2, 0.2])[:2]
        OPEN, CLOSE = self.gripper_open, self.gripper_close
        home = {j: float(q) for j, q in zip(jn, start_q)}

        waypoints: list[dict[str, float]] = []
        phases: list[str] = []
        cur_arm = start_arm
        cur_grip = OPEN  # current gripper target; ramped on in-place close/release

        def emit(arm_q, gripval, phase):
            wp = dict(home)
            for i, a in enumerate(arm_names):
                wp[a] = float(arm_q[i])
            if grip is not None:
                wp[grip] = gripval
            waypoints.append(wp)
            phases.append(phase)

        # (phase, goal_xyz | None for an in-place gripper event, gripper value,
        #  orient): reach + grasp descend onto the cube FINGER-DOWN (top-down
        #  pickup -- the part the user sees). lift + place stay position-only: the
        #  5-DOF arm can't both hold the gripper finger-down AND reach the diagonal
        #  bin, so forcing a finger-down lift leaves the place segment unreachable
        #  ("start state in collision"). Freeing the lift orientation lets the arm
        #  re-pose for the bin while the cube rides along (rigid kinematic grasp).
        # Descend the grasp deeper toward the cube (lower tool-z) so the fingers
        # come down near the cube instead of hovering well above it, while
        # keeping reach/lift at the clearance height the arm re-uses for the bin.
        # Per-segment: (phase, goal_xyz|None, gripper, orient, apply_tcp). The
        # FINGERTIP TCP offset is applied to the PICK segments (reach/grasp/lift)
        # so the fingers actually land on the cube; the PLACE segments use the
        # plain tool frame because the full TCP offset pushes the bin pose out of
        # the 5-DOF arm's reach (-> scripted fallback). The cube is carried at the
        # fingertip either way, so it still drops near the bin.
        grasp_goal = [cx, cy, table_z + grasp_z]
        segments = [
            ("reach", [cx, cy, table_z + approach], OPEN, True, True),
            ("grasp", grasp_goal, OPEN, True, True),
            ("close", None, CLOSE, False, True),
            ("lift", [cx, cy, table_z + approach], CLOSE, False, True),
            # place/place_down: keep the top-down orientation bias (orient=True)
            # so cuRobo holds the gripper finger-down and does NOT flip to a
            # mirrored IK branch -- a position-only place lets the base joint snap
            # ~180 deg to the opposite side at place_down, flinging the cube far
            # from the bin (validated offline: place_down pan jumped -0.83 -> +0.98).
            ("place", [px, py, table_z + approach], CLOSE, True, True),
            ("place_down", [px, py, table_z + grasp_z + 0.01], CLOSE, True, True),
            ("release", None, OPEN, False, False),
        ]
        for phase, xyz, gripval, orient, apply_tcp in segments:
            if xyz is None:
                # In-place gripper event (close/release). Ramp the gripper from
                # its current value to the target across hold_steps (smooth, no
                # violent flick), THEN hold at the target for extra frames so the
                # gripper FULLY closes/clamps the cube before the arm moves on --
                # without this the close-phase ends with the jaw only partly shut
                # and the cube slips out during the lift.
                ramp = max(1, hold_steps)
                for s in range(1, ramp + 1):
                    t = s / ramp
                    emit(cur_arm, (1.0 - t) * cur_grip + t * gripval, phase)
                # Extra clamp/settle frames at the fully-commanded gripper value.
                clamp_frames = 18 if phase == "close" else hold_steps
                for _ in range(clamp_frames):
                    emit(cur_arm, gripval, phase)
                cur_grip = gripval
                continue
            try:
                seg = self._plan_segment(cur_arm, xyz, quat, orient=orient, apply_tcp=apply_tcp)
            except RuntimeError:
                # The PICK segments (reach/grasp/lift) must succeed for a real
                # cuRobo demonstration -> re-raise so the caller's ScriptedPlanner
                # fallback handles the whole episode. The PLACE segments are at the
                # 5-DOF arm's reach edge and often unsolvable from the fingertip
                # grasp pose; rather than throwing away the (good) cuRobo pick, do
                # a joint-space interpolation toward a heuristic place pose so the
                # episode stays a cuRobo grasp + a smooth scripted place.
                if phase in ("reach", "grasp", "lift"):
                    raise
                target_arm = self._heuristic_place_arm(cur_arm, [px, py], home_arm=start_arm)
                # Many small steps so the kinematically-carried cube moves smoothly
                # to over the bin instead of being teleported in big jumps (which
                # injects velocity and flings it). 40 steps ~ the cuRobo segment
                # density, keeping the place at real-time speed.
                steps = 40
                seg = [
                    [cur_arm[k] + (target_arm[k] - cur_arm[k]) * (s / steps) for k in range(len(cur_arm))]
                    for s in range(1, steps + 1)
                ]
            for arm_q in seg:
                emit(arm_q, gripval, phase)
            cur_arm = seg[-1]
            cur_grip = gripval

        return JointTrajectory(
            joint_names=jn,
            waypoints=waypoints,
            phases=phases,
            planner="curobo",
            meta={
                "arm_joints": arm_names,
                "gripper_joint": grip,
                "tool_frame": self.tool_frame,
                "cube_xy": [cx, cy],
                "place_xy": [px, py],
            },
        )


_SCRIPTED_KEYS = ("gripper_open", "gripper_close", "steps_per_phase")
_PRECOMPUTED_KEYS = ("traj_path",)
_CUROBO_KEYS = (
    "urdf_path",
    "asset_path",
    "tool_frame",
    "self_collision",
    "device",
    "grasp_quaternion",
    "top_down_grasp",
    "top_down_weight",
    "orientation_tolerance",
    "top_down_attempts",
    "gripper_open",
    "gripper_close",
)


def _curobo_usable(kwargs: dict) -> bool:
    """cuRobo is usable only if installed AND an SO-101 URDF is resolvable.

    URDF resolution follows :func:`resolve_so101_urdf` (explicit kwarg ->
    ``SO101_URDF`` -> the auto-downloaded strands-robots cache URDF), so the
    ``auto`` path now selects cuRobo out-of-the-box on a box that already ran
    the MuJoCo demo, instead of silently falling back to scripted just because
    no URDF flag was passed.
    """
    if not CUROBO_AVAILABLE:
        return False
    return bool(resolve_so101_urdf(kwargs.get("urdf_path")))


def make_planner(prefer: str = "auto", robot_cfg: str = "so101", **kwargs):
    """Return the best available planner.

    ``prefer``: ``"auto"`` (precomputed cuRobo replay if a trajectory file is
    set, else in-process cuRobo if installed + a URDF is resolvable, else
    scripted), ``"curobo"`` (force cuRobo -- but if a precomputed trajectory is
    available, replay that instead, since in-kit cuRobo can't run on the Isaac
    backend due to the warp conflict), ``"precomputed"`` (force the offline
    cuRobo replay), or ``"scripted"``.
    """
    prefer = (prefer or "auto").lower()

    def _scripted():
        return ScriptedPlanner(**{k: v for k, v in kwargs.items() if k in _SCRIPTED_KEYS})

    def _precomputed():
        return PrecomputedPlanner(**{k: v for k, v in kwargs.items() if k in _PRECOMPUTED_KEYS})

    def _precomputed_available() -> bool:
        return PrecomputedPlanner(**{k: v for k, v in kwargs.items() if k in _PRECOMPUTED_KEYS}).available()

    if prefer == "scripted":
        return _scripted()
    if prefer == "precomputed":
        return _precomputed()
    if prefer == "curobo":
        # A pre-planned cuRobo trajectory replays even where in-process cuRobo
        # can't run (Isaac kit's warp 1.5 vs cuRobo's warp 1.14). Prefer it.
        if _precomputed_available():
            logger.info("Using precomputed cuRobo trajectory (offline plan replay).")
            return _precomputed()
        if not CUROBO_AVAILABLE:
            logger.warning(
                "cuRobo requested but not installed; using ScriptedPlanner. %s", CUROBO_INSTALL_HINT.splitlines()[0]
            )
            return _scripted()
        return CuroboMotionPlanner(**{k: v for k, v in kwargs.items() if k in _CUROBO_KEYS})
    # auto
    if _precomputed_available():
        return _precomputed()
    if _curobo_usable(kwargs):
        return CuroboMotionPlanner(**{k: v for k, v in kwargs.items() if k in _CUROBO_KEYS})
    return _scripted()
