#!/usr/bin/env python3
"""Three-phase emergency evacuation protocol for a robot fleet, benchmark-scored.

Goal: During an emergency, robots must immediately clear the evacuation path
and never block personnel. A plain e-stop is NOT sufficient - a frozen robot
in a corridor is itself the blocking hazard, and mesh lockout refuses every
action except ``status``/``resume`` (strands_robots.mesh.core). So the
protocol layers three phases on the primitives that already exist, with the
LLM outside the safety path at every step:

- **Phase 1 - abort**: an injected alarm broadcast (a mesh event, not a
  simulated sensor) stops every policy rollout fleet-wide. Rate-limited (an
  alarm flood cannot re-trigger the protocol) and audited.
- **Phase 2 - clear path**: each robot runs a pre-validated deterministic
  retreat - scripted base/joint setpoints, never a learned policy - to its
  muster pose. Priority conflicts resolve by deterministic corridor-distance
  ordering: closest to the path moves first, the others hold. Explicitly not
  an LLM decision.
- **Phase 3 - lockout + HITL resume**: mesh lockout engages only AFTER the
  path is clear (lockout first would freeze the hazard in place). Resume goes
  through the existing HMAC override protocol with operator approval, and a
  declined resume leaves the lockout engaged.

Scored, not narrated: a ``DeclarativeBenchmark`` (predicate DSL) fails the run
if any robot re-enters the inflated corridor (clearance margin) at any tick
after the path is declared clear, and succeeds only when the personnel proxy
reaches the exit unimpeded AND the fleet-wide abort finished inside its
deadline.

Dependencies: pip install "strands-robots[sim-mujoco,mesh]"
              (--dry-run needs only the base package: no simulator, no Zenoh)
Expected output: alarm -> abort (timed) -> ordered retreat with live corridor
                 clearances -> lockout engaged at muster -> a wrong-code
                 resume refused with the lockout intact -> benchmark verdict
                 as the proxy traverses -> incident report from the signed
                 audit log -> HITL-approved resume.
Runtime: ~5 seconds with --dry-run; under ~90 seconds live.

Note: The live lockout drill needs `STRANDS_MESH_OVERRIDE_CODE`; when unset,
      the example generates a single-run code and says so loudly. Interactive
      operator approval is the default; set `STRANDS_MESH_HITL_ACTIONS=none`
      for unattended runs (CI posture). Set STRANDS_MESH_AUDIT_PSK to sign the
      audit trail the incident report is built from.

Part of the fleet suite (epic #2179). The retreat sits behind the small
``EvacuationWorld`` seam so the Isaac adapter (#2123) drops in later; the
read-only Rerun fleet dashboard (``examples/fleet/dashboard.py``) attaches to
the same mesh and shows the safety events live.
"""

from __future__ import annotations

import os

os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import secrets
import time
from collections.abc import Callable
from typing import Any

from strands_robots.mesh.audit import log_safety_event, read_audit_log, verify_audit_integrity
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.predicates import PREDICATE_REGISTRY, BoolPredicate, register_predicate

COORDINATOR_ID = "evac-coordinator"
FLEET_PEER_ID = "evac-fleet-sim"
ALARM_TOPIC = "strands/safety/alarm"
PROXY_BODY = "evac-proxy"

# The corridor is the evacuation path: an axis-aligned rectangle in the world
# XY plane. CLEARANCE_M is the margin every robot must keep from it after the
# path is declared clear; the benchmark checks it as an INFLATED rectangle
# (per-axis inflation is stricter than Euclidean distance near the corners,
# i.e. conservative in the safe direction).
CORRIDOR = {"x_min": -3.0, "x_max": 3.0, "y_min": -0.7, "y_max": 0.7}
CLEARANCE_M = 0.8
ABORT_DEADLINE_S = 10.0

# The personnel proxy traverses the corridor centerline toward the exit. It
# advances only while the next stretch of path is clear of robots - a blocked
# proxy waits, never teleports through, so "reaches the exit" is honest.
PROXY_START_XY = (-2.8, 0.0)
PROXY_STEP_M = 0.15
PROXY_STANDOFF_M = 0.6
EXIT_REGION = {"min": [2.4, -0.7, -2.0], "max": [3.6, 0.7, 3.0]}

RETREAT_STEP_M = 0.12  # scripted base setpoint advance per retreat tick
MUSTER_TOL_M = 0.05

# The fleet (epic D3 cast): two mobile robots mid-task inside the corridor and
# one arm whose reach overhangs it. ``muster`` is a pre-validated pose OUTSIDE
# the inflated corridor; the arm musters by folding (its base cannot move), so
# its tracked point is the wrist, not the base.
FLEET: list[dict[str, Any]] = [
    {"robot": "lekiwi-1", "data_config": "lekiwi", "kind": "mobile", "spawn": [0.4, 0.35, 0.0], "muster": [0.4, 2.2]},
    {
        "robot": "go2-1",
        "data_config": "unitree_go2",
        "kind": "mobile",
        "spawn": [-1.2, -0.4, 0.0],
        "muster": [-1.2, -2.2],
    },
    {"robot": "arm-1", "data_config": "so101", "kind": "arm", "spawn": [1.6, 1.7, 0.0], "muster": [1.6, 1.75]},
]

# Folded muster joint setpoints for the arm (radians), applied to the arm's
# joints in kinematic order (base rotation first) and clamped to each joint's
# range at discovery time. Pre-validated: rotates the reach away from the
# corridor and tucks it over the base.
ARM_MUSTER_SETPOINTS = [1.5, -1.5, 1.4, 1.0, 0.0, 0.0]
ARM_RETREAT_TICKS = 20


def corridor_distance(x: float, y: float, corridor: dict[str, float] | None = None) -> float:
    """Euclidean distance from a world XY point to the corridor rectangle.

    Zero inside the corridor. This is the live clearance telemetry; the
    benchmark's pass/fail uses the (stricter) inflated-rectangle form via the
    built-in ``inside_region`` predicate.
    """
    c = CORRIDOR if corridor is None else corridor
    dx = max(c["x_min"] - x, 0.0, x - c["x_max"])
    dy = max(c["y_min"] - y, 0.0, y - c["y_max"])
    return float((dx * dx + dy * dy) ** 0.5)


def _centerline_distance(x: float, y: float) -> float:
    """Distance to the corridor centerline (y = mid): the retreat priority key."""
    return abs(y - (CORRIDOR["y_min"] + CORRIDOR["y_max"]) / 2.0)


def retreat_order(positions: dict[str, tuple[float, float]]) -> list[str]:
    """Deterministic priority ordering: closest to the path moves first.

    Sorted by distance to the corridor centerline ascending, name-tiebroken.
    Explicitly not an LLM decision - the safety layer is deterministic code.
    """
    return sorted(positions, key=lambda name: (_centerline_distance(*positions[name]), name))


class AlarmGate:
    """Rate limiter for alarm handling: an alarm flood must not re-trigger the
    protocol back to back. Mirrors the robot_mesh tool's emergency_stop cap
    (3 per rolling 60 s); a suppressed alarm is audited, never silently dropped.
    """

    def __init__(self, max_alarms: int = 3, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self._max = max_alarms
        self._window = window_s
        self._clock = clock
        self._admitted: list[float] = []

    def admit(self, alarm_id: str) -> bool:
        now = self._clock()
        self._admitted = [t for t in self._admitted if now - t < self._window]
        if len(self._admitted) >= self._max:
            _audit("evacuation_alarm_suppressed", {"alarm_id": alarm_id, "window_s": self._window})
            return False
        self._admitted.append(now)
        return True


def _audit(event: str, payload: dict[str, Any]) -> None:
    log_safety_event(event, COORDINATOR_ID, payload)


def _evacuation_abort_within(deadline_s: float) -> BoolPredicate:
    """True when the recorded fleet-wide abort finished within ``deadline_s``.

    Reads ``sim.evacuation_abort_elapsed_s``, stamped by phase 1. Stateless by
    design (no closure over run state), so the guarded re-registration below
    is safe across reloads.
    """

    def check(sim: Any) -> bool:
        elapsed = getattr(sim, "evacuation_abort_elapsed_s", None)
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
            return False
        return 0.0 <= float(elapsed) <= float(deadline_s)

    return check


def register_evacuation_predicates() -> None:
    """Register the custom abort-deadline predicate (idempotent)."""
    if "evacuation_abort_within" not in PREDICATE_REGISTRY:
        register_predicate("evacuation_abort_within", _evacuation_abort_within)


def build_benchmark_spec(tracked_bodies: dict[str, str], abort_deadline_s: float = ABORT_DEADLINE_S) -> dict[str, Any]:
    """The evacuation scorecard as a declarative predicate-DSL spec.

    Failure at ANY scored tick: a tracked robot body inside the corridor
    inflated by the clearance margin - that is "min distance to the corridor
    stayed above X for all t after T_clear", expressed without a NOT.
    Success: the proxy inside the exit region AND the abort inside deadline.
    A proxy that a lingering robot blocks simply never reaches the exit, so
    the run fails on the step budget - unimpeded is asserted, not narrated.
    """
    inflated = {
        "min": [CORRIDOR["x_min"] - CLEARANCE_M, CORRIDOR["y_min"] - CLEARANCE_M, -2.0],
        "max": [CORRIDOR["x_max"] + CLEARANCE_M, CORRIDOR["y_max"] + CLEARANCE_M, 3.0],
    }
    return {
        "name": "emergency-evacuation-corridor-clear",
        "max_steps": 120,
        "default_robot": "lekiwi",
        "success": {
            "all": [
                {"predicate": "inside_region", "body": PROXY_BODY, **EXIT_REGION},
                {"predicate": "evacuation_abort_within", "deadline_s": abort_deadline_s},
            ]
        },
        "failure": {
            "any": [{"predicate": "inside_region", "body": body, **inflated} for body in tracked_bodies.values()]
        },
    }


# The evacuation-world seam. The protocol core talks to this surface only, so
# the Isaac adapter (#2123) is one more implementation, not a protocol change.
# Implementations: MujocoEvacuationWorld (live) and ScriptedEvacuationWorld
# (--dry-run and the smoke test). The contract:
#
#   robot_names            ordered fleet names
#   tracked_body(name)     body name the benchmark watches for that robot
#   tracked_xy(name)       that body's current world XY
#   tasks_running()        names still running a policy rollout
#   stop_all_tasks()       fleet-wide abort (mesh broadcast on the live path)
#   retreat_tick(name)     ONE deterministic setpoint tick toward muster;
#                          returns True once mustered
#   proxy_xy()/set_proxy_xy(x, y)   the personnel proxy
#   settle(n)              advance physics/time n ticks
#   sim                    the object benchmark predicates read


def run_evacuation(
    world: Any,
    *,
    abort_deadline_s: float = ABORT_DEADLINE_S,
    abort_poll_s: float = 0.2,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_tick: Callable[[str, float, float], None] | None = None,
) -> dict[str, Any]:
    """Phases 1 and 2: fleet-wide abort, then the ordered deterministic retreat.

    Returns ``{"abort_elapsed_s", "order", "clearances"}`` and stamps
    ``world.sim.evacuation_abort_elapsed_s`` for the benchmark's deadline
    predicate. Raises if the abort misses its deadline or a robot fails to
    muster clear of the corridor - the path being clear is asserted, never
    assumed, before phase 3 may engage the lockout.

    The stop is delivered at-least-once: it is re-issued on every poll until
    the fleet reports quiet. A single stop can race a rollout that is still
    warming up (its running flag is raised only once its control loop starts),
    and a stop that lands inside that window would be silently overwritten -
    re-issuing the idempotent stop closes the race without trusting any
    single acknowledgement.
    """
    print("\nphase 1 - abort: stopping every rollout fleet-wide")
    t0 = clock()
    while True:
        world.stop_all_tasks()
        if not world.tasks_running():
            break
        if clock() - t0 > abort_deadline_s:
            still = world.tasks_running()
            _audit("evacuation_abort_timeout", {"deadline_s": abort_deadline_s, "still_running": still})
            raise RuntimeError(f"abort missed its {abort_deadline_s:.0f}s deadline; still running: {still}")
        sleep(abort_poll_s)
    abort_elapsed = clock() - t0
    world.sim.evacuation_abort_elapsed_s = abort_elapsed
    _audit("evacuation_abort_complete", {"elapsed_s": round(abort_elapsed, 3), "robots": list(world.robot_names)})
    print(f"  all rollouts stopped in {abort_elapsed:.2f}s (deadline {abort_deadline_s:.0f}s)")

    print("\nphase 2 - clear path: deterministic retreat, closest to the path first")
    positions = {name: world.tracked_xy(name) for name in world.robot_names}
    order = retreat_order(positions)
    _audit("evacuation_retreat_order", {"order": order})
    print(f"  order: {' -> '.join(order)} (corridor-distance priority; the others hold)")
    clearances: dict[str, float] = {}
    tick = 0
    for name in order:
        budget = 400  # deterministic setpoints converge; a diverging retreat is a fault, not a wait
        while not world.retreat_tick(name):
            tick += 1
            budget -= 1
            if budget <= 0:
                _audit("evacuation_retreat_stuck", {"robot": name})
                raise RuntimeError(f"{name}: retreat did not converge; refusing to engage lockout")
            if on_tick is not None:
                for robot in world.robot_names:
                    on_tick(robot, float(tick), corridor_distance(*world.tracked_xy(robot)))
        clearance = corridor_distance(*world.tracked_xy(name))
        clearances[name] = round(clearance, 3)
        _audit("evacuation_robot_mustered", {"robot": name, "clearance_m": clearances[name]})
        print(f"  {name}: mustered, corridor clearance {clearance:.2f} m")

    breaching = {n: c for n, c in clearances.items() if c <= CLEARANCE_M}
    if breaching:
        _audit("evacuation_clear_failed", {"breaching": breaching, "required_m": CLEARANCE_M})
        raise RuntimeError(f"path not clear: {breaching} (required > {CLEARANCE_M} m); refusing to engage lockout")
    _audit("evacuation_path_clear", {"clearances": clearances, "required_m": CLEARANCE_M})
    print(f"  path clear: every clearance > {CLEARANCE_M} m")
    return {"abort_elapsed_s": abort_elapsed, "order": order, "clearances": clearances}


def score_evacuation(
    world: Any,
    benchmark: DeclarativeBenchmark,
    *,
    on_tick: Callable[[str, float, float], None] | None = None,
) -> dict[str, Any]:
    """Score the cleared path with the declarative benchmark, tick by tick.

    The proxy advances along the corridor only while the stretch ahead is
    clear of robots; every tick the benchmark's failure clause (a robot back
    inside the inflated corridor) and success clause (proxy at the exit,
    abort inside deadline) are evaluated. Deterministic: scripted setpoints,
    fixed step sizes, no randomness.
    """
    for step in range(benchmark.max_steps):
        px, py = world.proxy_xy()
        ahead = (px + PROXY_STEP_M, py)
        blocked = any(
            ((world.tracked_xy(n)[0] - ahead[0]) ** 2 + (world.tracked_xy(n)[1] - ahead[1]) ** 2) ** 0.5
            < PROXY_STANDOFF_M
            for n in world.robot_names
        )
        if not blocked:
            world.set_proxy_xy(*ahead)
        world.settle(1)
        if on_tick is not None:
            for robot in world.robot_names:
                on_tick(robot, float(step), corridor_distance(*world.tracked_xy(robot)))
        if benchmark.is_failure(world.sim):
            verdict = {"passed": False, "reason": "corridor clearance breached", "steps": step + 1}
            _audit("evacuation_scored", verdict)
            return verdict
        if benchmark.is_success(world.sim):
            verdict = {"passed": True, "reason": "proxy reached the exit unimpeded", "steps": step + 1}
            _audit("evacuation_scored", verdict)
            return verdict
    verdict = {"passed": False, "reason": "proxy never reached the exit", "steps": benchmark.max_steps}
    _audit("evacuation_scored", verdict)
    return verdict


def build_incident_report(records: list[dict[str, Any]], integrity: dict[str, Any] | None = None) -> str:
    """Deterministic incident report from the signed audit trail.

    Structured post-event reconstruction - the LLM stays outside the safety
    path; pass the result to an agent for a narrative if you want one
    (``--agent-report``), but the record of what happened is this.

    The integrity verdict attests exactly ``records``, because the header and
    the timeline below it have to describe the same records to be read
    together. Leave ``integrity`` unset and this pairs them for you; pass a
    verdict computed over some other record set and the report says two
    different things. ``verify_audit_integrity()`` with no argument is that
    other set - it re-reads the whole log, which on any machine that has run
    the mesh before is mostly other runs.

    Args:
        records: The audit records to report on, already scoped to this run.
        integrity: Optional pre-computed verdict for ``records``. Defaults to
            attesting ``records`` themselves.
    """
    if integrity is None:
        integrity = verify_audit_integrity(records)
    lines = [
        "# Evacuation incident report",
        "",
        f"Audit integrity: ok={integrity['ok']} (signed={integrity['signed']}/{integrity['total']})",
        "",
        "| t | peer | event | detail |",
        "|---|---|---|---|",
    ]
    t0: float | None = None
    for record in records:
        event = record.get("event")
        ts = record.get("ts")
        if not isinstance(event, str) or isinstance(ts, bool) or not isinstance(ts, (int, float)):
            continue
        if t0 is None:
            t0 = float(ts)
        payload = record.get("payload")
        detail = ""
        if isinstance(payload, dict):
            detail = ", ".join(
                f"{k}={payload[k]}" for k in sorted(payload) if isinstance(payload[k], (str, int, float))
            )
        lines.append(f"| +{float(ts) - t0:.2f}s | {record.get('peer_id', '?')} | {event} | {detail[:120]} |")
    return "\n".join(lines)


class EvacuationTrace:
    """Optional visualization: live corridor-clearance plot, camera tiles and
    a saved ``.rrd`` replay artifact via Rerun; a GIF of the retreat via
    imageio. Both degrade loudly to no-ops when unavailable - visualization
    must never gate the safety path.
    """

    def __init__(self, rrd_path: str | None = None, gif_path: str | None = None, camera: str = "default"):
        self._rr: Any | None = None
        self._gif_path = gif_path
        self._camera = camera
        self._frames: list[Any] = []
        self._seq: dict[str, int] = {}
        if rrd_path:
            try:
                from strands_robots.utils import require_optional

                rr = require_optional("rerun", pip_install="rerun-sdk", purpose="the evacuation replay artifact")
                rr.init("strands-fleet-evacuation")
                rr.save(rrd_path)
                self._rr = rr
                print(f"  rerun replay -> {rrd_path}")
            except ImportError as exc:
                print(f"  rerun unavailable ({exc}); continuing without the .rrd artifact")

    def clearance(self, robot: str, tick: float, clearance_m: float) -> None:
        del tick  # phases restart their tick counters; the plot wants one monotonic axis
        if self._rr is None:
            return
        seq = self._seq[robot] = self._seq.get(robot, 0) + 1
        set_time = getattr(self._rr, "set_time_sequence", None)
        if set_time is not None:
            set_time("tick", seq)
        scalar_type = getattr(self._rr, "Scalars", None) or getattr(self._rr, "Scalar", None)
        if scalar_type is not None:
            self._rr.log(f"corridor/clearance/{robot}", scalar_type(clearance_m))

    def frame(self, world: Any) -> None:
        """Capture one camera tile (live worlds only; scripted worlds have none)."""
        get_frame = getattr(world.sim, "get_frame", None)
        if get_frame is None:
            return
        try:
            rgb, _depth = get_frame(camera_name=self._camera, width=480, height=360)
        except Exception as exc:  # noqa: BLE001 - visualization must never break the drill
            print(f"  frame capture failed ({exc}); continuing")
            return
        if self._gif_path is not None:
            self._frames.append(rgb)
        if self._rr is not None:
            image_type = getattr(self._rr, "Image", None)
            if image_type is not None:
                self._rr.log(f"corridor/camera/{self._camera}", image_type(rgb))

    def close(self) -> None:
        if self._gif_path and self._frames:
            try:
                import imageio.v2 as imageio

                imageio.mimsave(self._gif_path, self._frames, duration=0.08)
                print(f"  retreat GIF -> {self._gif_path} ({len(self._frames)} frames)")
            except ImportError as exc:
                print(f"  imageio unavailable ({exc}); skipping the GIF artifact")


class ScriptedEvacuationWorld:
    """Kinematic evacuation world: no simulator, no Zenoh, fully deterministic.

    Doubles as the benchmark's engine: ``get_body_state`` serves scripted
    positions in the plain (non-envelope) shape the predicate helpers accept,
    so the REAL ``DeclarativeBenchmark`` scores the dry run too.
    """

    def __init__(self, fleet: list[dict[str, Any]] | None = None):
        self._fleet = {entry["robot"]: dict(entry) for entry in (FLEET if fleet is None else fleet)}
        self.robot_names = list(self._fleet)
        self._xy: dict[str, list[float]] = {}
        self._arm_ticks: dict[str, int] = {}
        for name, entry in self._fleet.items():
            if entry["kind"] == "arm":
                # The arm's tracked point is its reach, which overhangs the
                # corridor edge until the fold pulls it back over the base.
                self._xy[name] = [entry["spawn"][0], CORRIDOR["y_max"] + 0.35]
                self._arm_ticks[name] = 0
            else:
                self._xy[name] = [entry["spawn"][0], entry["spawn"][1]]
        self._proxy = list(PROXY_START_XY)
        self._running = set(self.robot_names)
        self.sim = self  # benchmark predicates read this object directly

    def tracked_body(self, name: str) -> str:
        return f"{name}/tracked"

    def tracked_xy(self, name: str) -> tuple[float, float]:
        return (self._xy[name][0], self._xy[name][1])

    def tasks_running(self) -> list[str]:
        return sorted(self._running)

    def stop_all_tasks(self) -> None:
        for name in self.tasks_running():
            print(f"  [scripted] {name}: rollout stopped")
        self._running.clear()

    def retreat_tick(self, name: str) -> bool:
        entry = self._fleet[name]
        if entry["kind"] == "arm":
            ticks = self._arm_ticks[name] = self._arm_ticks[name] + 1
            frac = min(1.0, ticks / ARM_RETREAT_TICKS)
            start_y = CORRIDOR["y_max"] + 0.35
            self._xy[name][1] = start_y + frac * (entry["muster"][1] - start_y)
            self._xy[name][0] = entry["muster"][0]
            return frac >= 1.0
        x, y = self._xy[name]
        tx, ty = entry["muster"]
        dx, dy = tx - x, ty - y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= MUSTER_TOL_M:
            return True
        step = min(RETREAT_STEP_M, dist)
        self._xy[name] = [x + dx / dist * step, y + dy / dist * step]
        return dist - step <= MUSTER_TOL_M

    def proxy_xy(self) -> tuple[float, float]:
        return (self._proxy[0], self._proxy[1])

    def set_proxy_xy(self, x: float, y: float) -> None:
        self._proxy = [x, y]

    def settle(self, n: int) -> None:
        """No-op: a kinematic world integrates nothing, so a tick advances nothing."""

    # Benchmark engine surface (plain non-envelope payloads).
    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name == PROXY_BODY:
            return {"status": "success", "position": [self._proxy[0], self._proxy[1], 0.1]}
        for name in self.robot_names:
            if body_name == self.tracked_body(name):
                x, y = self.tracked_xy(name)
                return {"status": "success", "position": [x, y, 0.2]}
        return {"status": "error", "content": [{"text": f"unknown body '{body_name}'"}]}


class MujocoEvacuationWorld:
    """Live evacuation world: one MuJoCo scene holding the whole fleet.

    The retreat writes base qpos directly (the documented teleport pattern:
    write qpos, run forward kinematics) under the sim lock - deterministic
    scripted setpoints, exactly what phase 2 promises. Joint-space folds go
    through the public ``set_joint_positions``.
    """

    def __init__(self, sim: Any, mesh: Any | None = None):
        self.sim = sim
        self.mesh = mesh  # coordinator peer; None = stop locally
        self.robot_names = [entry["robot"] for entry in FLEET]
        self._fleet = {entry["robot"]: dict(entry) for entry in FLEET}
        self._arm_ticks: dict[str, int] = {}
        # name -> (qpos adr, dof adr, tracked body); adr < 0 marks a fixed base.
        self._base: dict[str, tuple[int, int, str]] = {}
        self._arm_joints: dict[str, dict[str, tuple[float, float]]] = {}  # name -> joint -> (start, target)
        self._discover()

    def _discover(self) -> None:
        import mujoco as mj

        # Direct model access is required here: MuJoCo has no public
        # base-teleport API (Isaac's set_robot_pose is the #2123 seam), and
        # the free joint / wrist body of each robot is model metadata.
        model, data = self.sim._world._model, self.sim._world._data
        for name, entry in self._fleet.items():
            robot = self.sim._world.robots[name]
            if entry["kind"] == "arm":
                joints: dict[str, tuple[float, float]] = {}
                for rel, target in zip(robot.joint_names, ARM_MUSTER_SETPOINTS, strict=False):
                    jname = f"{robot.namespace}{rel}"
                    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
                    if jid < 0:
                        continue
                    clamped = float(target)
                    if model.jnt_limited[jid]:
                        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
                        clamped = min(max(clamped, lo), hi)
                    joints[jname] = (float(data.qpos[model.jnt_qposadr[jid]]), clamped)
                if not joints:
                    raise RuntimeError(f"{name}: no muster joints resolved; cannot script the arm retreat")
                self._arm_joints[name] = joints
                last_jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, next(reversed(joints)))
                wrist = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.jnt_bodyid[last_jid])
                self._base[name] = (-1, -1, wrist)
                self._arm_ticks[name] = 0
                continue
            # The library's shared free-joint detection: a NAMED free joint
            # first, then the kinematic-tree walk that finds an unnamed
            # <freejoint> (LeKiwi's base). A fixed-base arm returns -1.
            jid = self.sim._robot_free_base_joint_id(model, robot)
            if jid < 0:
                raise RuntimeError(f"{name}: no free base joint found; cannot script the base retreat")
            adr = int(model.jnt_qposadr[jid])
            dadr = int(model.jnt_dofadr[jid])
            body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.jnt_bodyid[jid])
            self._base[name] = (adr, dadr, body)

    def tracked_body(self, name: str) -> str:
        return self._base[name][2]

    def tracked_xy(self, name: str) -> tuple[float, float]:
        adr, _dadr, body = self._base[name]
        if adr >= 0:
            data = self.sim._world._data
            return (float(data.qpos[adr]), float(data.qpos[adr + 1]))
        result = self.sim.get_body_state(body_name=body)
        payload = next((c["json"] for c in result.get("content", []) if isinstance(c, dict) and "json" in c), {})
        pos = payload.get("position", [0.0, 0.0, 0.0])
        return (float(pos[0]), float(pos[1]))

    def tasks_running(self) -> list[str]:
        result = self.sim.list_policies_running()
        text = result["content"][0]["text"]
        if text.startswith("No policies"):
            return []
        return [line.strip("- ").strip() for line in text.splitlines()[1:]]

    def stop_all_tasks(self) -> None:
        if self.mesh is not None:
            # The fleet-wide abort is a mesh broadcast, answered by the sim
            # peer's stop_task (see _FleetSimPeer); a peer that cannot stop
            # answers ok=False and is surfaced, never assumed halted. The ack
            # window is short on purpose: run_evacuation re-issues the stop
            # until the fleet is quiet (at-least-once delivery), so waiting
            # out a long collection window per attempt would spend the abort
            # deadline on acks instead of on stopping.
            responses = self.mesh.broadcast({"action": "stop"}, timeout=1.5)
            not_stopped = [
                r
                for r in responses
                if isinstance(r, dict)
                and isinstance(r.get("result", r), dict)
                and (r.get("result", r).get("ok") is False or r.get("result", r).get("status") == "error")
            ]
            if not_stopped:
                raise RuntimeError(f"fleet abort: peers refused to stop: {not_stopped}")
            return
        for name in self.robot_names:
            self.sim.stop_policy(robot_name=name)

    def retreat_tick(self, name: str) -> bool:
        import mujoco as mj

        entry = self._fleet[name]
        if entry["kind"] == "arm":
            ticks = self._arm_ticks[name] = self._arm_ticks[name] + 1
            frac = min(1.0, ticks / ARM_RETREAT_TICKS)
            setpoints = {j: s + frac * (t - s) for j, (s, t) in self._arm_joints[name].items()}
            result = self.sim.set_joint_positions(positions=setpoints)
            if result.get("status") != "success":
                raise RuntimeError(f"{name}: scripted joint setpoint refused: {result}")
            self.sim.step(2)
            return frac >= 1.0
        adr, dadr, _body = self._base[name]
        model, data = self.sim._world._model, self.sim._world._data
        x, y = float(data.qpos[adr]), float(data.qpos[adr + 1])
        tx, ty = entry["muster"]
        dx, dy = tx - x, ty - y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= MUSTER_TOL_M:
            return True
        step = min(RETREAT_STEP_M, dist)
        with self.sim._lock:
            data.qpos[adr] = x + dx / dist * step
            data.qpos[adr + 1] = y + dy / dist * step
            # Zero the base twist so the teleported setpoint holds.
            data.qvel[dadr : dadr + 6] = 0.0
            mj.mj_forward(model, data)
        self.sim.step(2)
        return dist - step <= MUSTER_TOL_M

    def proxy_xy(self) -> tuple[float, float]:
        result = self.sim.get_body_state(body_name=PROXY_BODY)
        payload = next((c["json"] for c in result.get("content", []) if isinstance(c, dict) and "json" in c), {})
        pos = payload.get("position", list(PROXY_START_XY) + [0.0])
        return (float(pos[0]), float(pos[1]))

    def set_proxy_xy(self, x: float, y: float) -> None:
        result = self.sim.move_object(PROXY_BODY, position=[x, y, 0.12])
        if result.get("status") != "success":
            raise RuntimeError(f"proxy move refused: {result}")

    def settle(self, n: int) -> None:
        self.sim.step(max(1, n))


class _FleetSimPeer:
    """Mesh owner for the sim peer: makes the whole sim fleet stoppable.

    ``Simulation`` exposes no ``stop_task``, so a bare sim peer honestly
    answers a fleet stop with ok=False; this adapter is the stop surface -
    one call stops every robot's rollout and reports it.
    """

    def __init__(self, sim: Any, robot_names: list[str]):
        self._sim = sim
        self._robot_names = robot_names

    def stop_task(self) -> dict[str, Any]:
        stopped = []
        for name in self._robot_names:
            result = self._sim.stop_policy(robot_name=name)
            if result.get("status") != "success":
                return {"ok": False, "error": f"stop_policy({name}) failed: {result}"}
            stopped.append(name)
        return {"ok": True, "status": "stopped", "robots": stopped}


def _operator_approves(prompt: str) -> bool:
    """HITL gate for the resume. Interactive by default; the CI/smoke posture
    is STRANDS_MESH_HITL_ACTIONS=none (epic D4), which auto-approves here the
    way the robot_mesh tool skips its interrupt. A decline is a first-class
    outcome: nothing is sent, the lockout stays.
    """
    if os.getenv("STRANDS_MESH_HITL_ACTIONS", "").strip().lower() == "none":
        print(f"  [HITL] {prompt} -> auto-approved (STRANDS_MESH_HITL_ACTIONS=none)")
        return True
    reply = input(f"  [HITL] {prompt} [y/N]: ").strip().lower()
    return reply in ("y", "yes", "approve")


def _run_lockout_drill(coordinator: Any, target_peer: str) -> None:
    """Phase 3 live: engage the lockout at muster, prove a bad resume holds it,
    then resume through the HMAC override protocol with operator approval."""
    print("\nphase 3 - lockout at muster + HITL resume")
    coordinator.emergency_stop()
    probe = coordinator.send(
        target_peer,
        {"action": "execute", "instruction": "probe: must be refused", "policy_provider": "mock", "n_steps": 1},
        timeout=10.0,
    )
    if not (isinstance(probe, dict) and probe.get("type") == "error"):
        raise RuntimeError(f"lockout did not engage: execute was not refused: {probe!r}")
    print(f"  {target_peer}: lockout engaged (execute refused, status/resume only)")

    denied = coordinator.send(target_peer, {"action": "resume", "override_code": "not-the-code"}, timeout=10.0)
    denied_result = denied.get("result") if isinstance(denied, dict) else None
    if not (isinstance(denied_result, dict) and denied_result.get("status") == "error"):
        raise RuntimeError(f"the wrong-code resume was not refused: {denied!r}")
    probe = coordinator.send(target_peer, {"action": "status"}, timeout=10.0)
    if not isinstance(probe, dict) or probe.get("type") != "response":
        raise RuntimeError(f"{target_peer} stopped answering status during lockout: {probe!r}")
    still_refused = coordinator.send(
        target_peer,
        {"action": "execute", "instruction": "probe: must be refused", "policy_provider": "mock", "n_steps": 1},
        timeout=10.0,
    )
    if not (isinstance(still_refused, dict) and still_refused.get("type") == "error"):
        raise RuntimeError(f"a rejected resume cleared the lockout: {still_refused!r}")
    print(f"  wrong-code resume refused ({denied_result.get('error', 'resume rejected')!s}); lockout intact")

    if not _operator_approves(f"resume {target_peer} out of lockout with the override code?"):
        print("  resume declined by the operator; the lockout stays engaged. Rerun to resume.")
        return
    resumed = coordinator.send(
        target_peer,
        {"action": "resume", "override_code": os.environ["STRANDS_MESH_OVERRIDE_CODE"]},
        timeout=10.0,
    )
    resumed_result = resumed.get("result") if isinstance(resumed, dict) else None
    if not (isinstance(resumed_result, dict) and resumed_result.get("status") == "ok"):
        raise RuntimeError(f"approved resume was rejected: {resumed!r}")
    print(f"  {target_peer}: resumed via HMAC override (operator approved)")


def _build_live_world() -> tuple[MujocoEvacuationWorld, Any, Callable[[], None]]:
    """Corridor scene + fleet + proxy + mesh peers. Returns (world, coordinator, cleanup)."""
    from strands_robots.mesh import init_mesh
    from strands_robots.simulation import Simulation

    sim = Simulation()
    coordinator: Any | None = None
    sim_mesh: Any | None = None

    def cleanup() -> None:
        if coordinator is not None and coordinator.alive:
            coordinator.stop()
        if sim_mesh is not None and sim_mesh.alive:
            sim_mesh.stop()
        sim.destroy()

    try:
        steps: list[Callable[[], dict[str, Any]]] = [
            lambda: sim.create_world(),
            # Visual corridor walls (static) and the exit marker.
            lambda: sim.add_object(
                "corridor-wall-north",
                shape="box",
                position=[0.0, CORRIDOR["y_max"] + 0.05, 0.05],
                size=[CORRIDOR["x_max"] - CORRIDOR["x_min"], 0.05, 0.1],
                color=[0.9, 0.6, 0.1, 1.0],
                is_static=True,
            ),
            lambda: sim.add_object(
                "corridor-wall-south",
                shape="box",
                position=[0.0, CORRIDOR["y_min"] - 0.05, 0.05],
                size=[CORRIDOR["x_max"] - CORRIDOR["x_min"], 0.05, 0.1],
                color=[0.9, 0.6, 0.1, 1.0],
                is_static=True,
            ),
            lambda: sim.add_object(
                "corridor-exit",
                shape="box",
                position=[3.0, 0.0, 0.01],
                size=[0.6, 1.2, 0.02],
                color=[0.1, 0.8, 0.2, 1.0],
                is_static=True,
            ),
            lambda: sim.add_object(
                PROXY_BODY,
                shape="cylinder",
                position=[PROXY_START_XY[0], PROXY_START_XY[1], 0.12],
                size=[0.18, 0.18, 0.24],
                color=[0.2, 0.4, 0.9, 1.0],
                mass=1.0,
            ),
            lambda: sim.add_camera(
                "corridor-cam", position=[0.0, -4.5, 4.0], target=[0.0, 0.0, 0.0], width=480, height=360
            ),
        ]
        steps.extend(
            lambda e=entry: sim.add_robot(name=e["robot"], data_config=e["data_config"], position=e["spawn"])
            for entry in FLEET
        )
        for build in steps:
            result = build()
            if result.get("status") != "success":
                raise RuntimeError(f"world construction failed: {result}")

        # The fleet is mid-task when the alarm lands.
        for entry in FLEET:
            result = sim.start_policy(
                robot_name=entry["robot"], policy_provider="mock", instruction="routine task", duration=60.0
            )
            if result.get("status") != "success":
                raise RuntimeError(f"start_policy({entry['robot']}) failed: {result}")

        sim_mesh = init_mesh(_FleetSimPeer(sim, [e["robot"] for e in FLEET]), peer_id=FLEET_PEER_ID, peer_type="sim")
        coordinator = init_mesh(_Coordinator(), peer_id=COORDINATOR_ID)
        if sim_mesh is None or coordinator is None:
            raise RuntimeError("mesh is disabled (STRANDS_MESH=0); rerun with --dry-run")
        # A peer whose mesh did not start publishes no presence and discovers
        # none, so the presence wait below can only expire.  Refuse here, where
        # the cause is still known.
        for peer_id, peer in ((FLEET_PEER_ID, sim_mesh), (COORDINATOR_ID, coordinator)):
            if not peer.alive:
                raise RuntimeError(
                    f"mesh did not start for peer {peer_id!r} (mesh.alive is False): install the mesh "
                    'extra with pip install "strands-robots[mesh]", or rerun with --dry-run'
                )
    except BaseException:
        cleanup()
        raise
    return MujocoEvacuationWorld(sim, mesh=coordinator), coordinator, cleanup


class _Coordinator:
    """Minimal mesh owner for the coordinator peer (it only sends)."""

    tool_name_str = COORDINATOR_ID


def _wait_for(predicate: Callable[[], bool], *, timeout_s: float, what: str, poll_s: float = 0.5) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(poll_s)
    raise RuntimeError(f"timed out after {timeout_s:.0f}s waiting for {what}")


def _run_live(trace: EvacuationTrace) -> dict[str, Any]:
    if not os.getenv("STRANDS_MESH_OVERRIDE_CODE", "").strip():
        code = secrets.token_urlsafe(9)
        os.environ["STRANDS_MESH_OVERRIDE_CODE"] = code
        print(f"STRANDS_MESH_OVERRIDE_CODE was unset; using a single-run code for this demo: {code}")

    world, coordinator, cleanup = _build_live_world()
    try:
        _wait_for(
            lambda: FLEET_PEER_ID in coordinator.peers_by_id,
            timeout_s=15.0,
            what="presence discovery of the fleet sim peer",
        )

        # The alarm is an injected mesh broadcast event (epic D5), observed by
        # the coordinator's own subscriber - coordination, not perception.
        alarms: list[dict[str, Any]] = []
        coordinator.subscribe(ALARM_TOPIC, callback=lambda _t, payload: alarms.append(payload), name="alarm")
        gate = AlarmGate()
        print("\ninjecting the alarm broadcast")
        coordinator.publish(ALARM_TOPIC, {"alarm_id": "drill-1", "peer_id": COORDINATOR_ID, "t": time.time()})
        _wait_for(lambda: bool(alarms), timeout_s=10.0, what="the alarm to arrive on the safety topic")
        alarm = alarms[0]
        if not gate.admit(str(alarm.get("alarm_id", "?"))):
            raise RuntimeError("the first alarm of the run was rate-limited; gate state is wrong")
        _audit("evacuation_alarm", {"alarm_id": alarm.get("alarm_id"), "source": alarm.get("peer_id")})

        def on_tick(robot: str, tick: float, clearance: float) -> None:
            trace.clearance(robot, tick, clearance)
            if int(tick) % 5 == 0 and robot == world.robot_names[0]:
                trace.frame(world)

        summary = run_evacuation(world, on_tick=on_tick)

        _run_lockout_drill(coordinator, FLEET_PEER_ID)

        print("\nscoring: proxy traversal against the declarative benchmark")
        register_evacuation_predicates()
        tracked = {name: world.tracked_body(name) for name in world.robot_names}
        benchmark = DeclarativeBenchmark.from_dict(build_benchmark_spec(tracked))
        verdict = score_evacuation(world, benchmark, on_tick=on_tick)
        summary["verdict"] = verdict
        return summary
    finally:
        cleanup()


def _run_dry(trace: EvacuationTrace) -> dict[str, Any]:
    """Scripted kinematics; the protocol core, ordering, benchmark and report
    are all real. The lockout/resume drill needs a live mesh (the smoke test
    asserts it through the real safety handlers)."""
    world = ScriptedEvacuationWorld()
    gate = AlarmGate()
    print("\ninjecting the alarm (scripted; the live path publishes on the mesh)")
    if not gate.admit("drill-1"):
        raise RuntimeError("the first alarm of the run was rate-limited; gate state is wrong")
    _audit("evacuation_alarm", {"alarm_id": "drill-1", "source": COORDINATOR_ID})

    summary = run_evacuation(world, sleep=lambda _s: None, on_tick=trace.clearance)

    print("\nphase 3 - lockout + HITL resume: live-mode only (see the smoke test)")

    print("\nscoring: proxy traversal against the declarative benchmark")
    register_evacuation_predicates()
    tracked = {name: world.tracked_body(name) for name in world.robot_names}
    benchmark = DeclarativeBenchmark.from_dict(build_benchmark_spec(tracked))
    verdict = score_evacuation(world, benchmark, on_tick=trace.clearance)
    summary["verdict"] = verdict
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="no simulator, no mesh: scripted kinematics")
    parser.add_argument("--rrd", default="", help="save a Rerun .rrd replay artifact to this path")
    parser.add_argument("--gif", default="", help="save a retreat GIF to this path (live mode)")
    parser.add_argument("--agent-report", action="store_true", help="narrate the incident report with an agent")
    args = parser.parse_args(argv)

    run_start = time.time()
    trace = EvacuationTrace(rrd_path=args.rrd or None, gif_path=args.gif or None, camera="corridor-cam")
    try:
        summary = _run_dry(trace) if args.dry_run else _run_live(trace)
    finally:
        trace.close()

    verdict = summary["verdict"]
    print(
        f"\nbenchmark verdict: {'PASS' if verdict['passed'] else 'FAIL'} - {verdict['reason']} ({verdict['steps']} ticks)"
    )
    print(f"abort: {summary['abort_elapsed_s']:.2f}s; retreat order: {' -> '.join(summary['order'])}")
    print(f"clearances at muster: {summary['clearances']}")

    records = read_audit_log(since=run_start - 1.0)
    report = build_incident_report(records)
    print("\n" + report)
    if args.agent_report:
        # Post-event narration only - the LLM never touches the safety path.
        from strands import Agent

        agent = Agent()
        agent(f"Write a three-sentence operator summary of this evacuation incident report:\n\n{report}")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
