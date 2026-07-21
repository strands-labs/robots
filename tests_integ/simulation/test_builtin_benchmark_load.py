"""End-to-end: every shipped built-in locomotion benchmark runs on its real robot.

:func:`~strands_robots.simulation.builtin_benchmarks.register_builtin_benchmarks`
ships canonical velocity-tracking locomotion benchmarks (``go2_walk_forward`` and
the humanoid ``g1_walk_forward`` / ``t1_walk_forward``) whose success/failure/
dense-reward clauses read the embodiment-agnostic floating-base surface
(``base_pos`` / ``base_quat`` / ``base_lin_vel`` / ``base_ang_vel``) from
``get_observation``. The unit tests in
``tests/simulation/test_builtin_benchmarks.py`` drive the compiled predicates on
a SYNTHETIC inline freejoint MJCF at known poses - fast and GL-free, but they
never load the real ``default_robot`` each spec targets. So nothing guards the
integration seam between a shipped spec and the robot it names:

  * a ``robot_descriptions`` rename / removal or a broken asset would make a
    shipped benchmark dead-on-arrival (``add_robot`` raises), yet every synthetic
    unit test would still pass;
  * a regression in the floating-base observation surfacing (the base signals
    the DSL reads) would silently zero the reward / never fire the predicates;
  * a drift in a robot's spawn stance below its ``base_below_z`` collapse
    threshold (or a topple past ``base_tipped``) would make the benchmark FAIL
    every episode at ``t=0``, and above its ``base_beyond_x`` line would make it
    SUCCEED spuriously - a standing-spawn regression no synthetic-pose test sees.

This module closes that seam: for EVERY shipped built-in spec it loads the real
``default_robot`` in MuJoCo (auto-downloading the ``robot_descriptions`` asset),
asserts the floating-base observation surfaces, asserts the standing spawn
neither trips the benchmark's ``failure`` predicates nor satisfies its
``success`` predicate, and runs the whole benchmark harness end-to-end via
``evaluate_benchmark`` (a mock policy driving real physics, exactly as
``test_lekiwi_sim`` smoke-tests a real robot) so the reward composition + episode
scoring are exercised on the actual embodiment.

The cases are derived from :func:`builtin_benchmark_specs` so any benchmark added
to the shipped set is covered automatically - no hardcoded robot list to drift.
Network + MuJoCo + GPU-render integration (collected only via ``hatch run
test-integ``), so it is deliberately out of the GL-free unit suite.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402 - after the MUJOCO_GL default is set

from strands_robots.registry import has_sim  # noqa: E402 - after MUJOCO_GL default
from strands_robots.simulation import create_simulation  # noqa: E402
from strands_robots.simulation.benchmark import (  # noqa: E402
    get_benchmark,
    register_benchmark,
    unregister_benchmark,
)
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark  # noqa: E402
from strands_robots.simulation.builtin_benchmarks import builtin_benchmark_specs  # noqa: E402

# A short episode budget: enough to exercise the full control loop + scoring on
# the real robot without a 1000-step rollout. The spawn-contract assertions
# below cover the t=0 predicate behaviour; this only proves the harness runs.
_SMOKE_STEPS = 4

# Shipped benchmark names, derived from the module so a newly-added built-in is
# covered automatically (no hardcoded list to drift out of sync with the specs).
_SHIPPED_SPECS: dict[str, dict[str, Any]] = builtin_benchmark_specs()
_SHIPPED_NAMES: list[str] = sorted(_SHIPPED_SPECS)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the module-global benchmark registry clean around each test."""
    for _n in _SHIPPED_NAMES:
        unregister_benchmark(_n)
    yield
    for _n in _SHIPPED_NAMES:
        unregister_benchmark(_n)


def _base_signal_is_vec(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(isinstance(v, (int, float)) for v in value)


@pytest.mark.parametrize("bench_name", _SHIPPED_NAMES)
def test_shipped_benchmark_runs_on_its_real_robot(bench_name: str) -> None:
    """A shipped built-in benchmark loads its real robot, surfaces the base state
    its DSL reads, holds the standing-spawn contract, and runs end-to-end."""
    spec = _SHIPPED_SPECS[bench_name]
    robot = spec["default_robot"]

    # Structural: the spec's default robot is a supported, sim-resolvable robot.
    assert robot in spec["supported_robots"], f"{bench_name}: default_robot not in supported_robots"
    assert has_sim(robot), f"{bench_name}: default_robot '{robot}' is not simulatable"

    # Register a copy trimmed to a short episode so the end-to-end run is fast;
    # the trim does not affect the state-reading predicates asserted below.
    trimmed = copy.deepcopy(spec)
    trimmed["max_steps"] = _SMOKE_STEPS
    register_benchmark(bench_name, DeclarativeBenchmark.from_dict(trimmed))

    sim = create_simulation(backend="mujoco")
    sim.create_world(ground_plane=True)
    sim.add_robot(robot)  # auto-downloads the robot_descriptions asset
    try:
        # The floating-base observation surface the base_* DSL reads must exist.
        obs = sim.get_observation(skip_images=True)
        assert _base_signal_is_vec(obs.get("base_pos"), 3), f"{robot}: base_pos not surfaced"
        assert _base_signal_is_vec(obs.get("base_quat"), 4), f"{robot}: base_quat not surfaced"
        assert _base_signal_is_vec(obs.get("base_lin_vel"), 3), f"{robot}: base_lin_vel not surfaced"
        assert _base_signal_is_vec(obs.get("base_ang_vel"), 3), f"{robot}: base_ang_vel not surfaced"

        bench = get_benchmark(bench_name)
        assert bench is not None  # just registered above

        # Standing-spawn contract: a freshly-spawned, upright robot must NOT
        # already be "failed" (its spawn height/orientation is above the
        # base_below_z / base_tipped fall thresholds) nor already "succeeded"
        # (it has not yet walked past the base_beyond_x line). Either would make
        # the benchmark score at t=0 - the exact silent regression the synthetic
        # unit tests, which set poses by hand, cannot catch.
        assert bench.is_failure(sim) is False, f"{bench_name}: standing spawn spuriously trips a failure predicate"
        assert bench.is_success(sim) is False, f"{bench_name}: standing spawn spuriously satisfies success"

        # The dense-reward composition evaluates to a finite scalar on the real
        # robot (the reward terms resolve the base signals, not a degenerate 0).
        info = bench.on_step(sim, {}, {})
        assert math.isfinite(info.reward), f"{bench_name}: dense reward is not finite"
        assert info.done is False

        # Full harness end-to-end: build a policy, run the control loop, score
        # the episode against the compiled spec on the real embodiment.
        res = sim.evaluate_benchmark(bench_name, policy_provider="mock", n_episodes=1)
        assert res["status"] == "success", res
        payload = next(c["json"] for c in res["content"] if "json" in c)
        assert payload["episodes_completed"] == 1
        assert payload["success_measured"] is True
        assert math.isfinite(payload["avg_reward"]), f"{bench_name}: avg_reward is not finite"
    finally:
        sim.destroy()


# ---------------------------------------------------------------------------
# Terrain-curriculum composition: a shipped locomotion benchmark run on a
# create_world(terrain=...) heightfield world -- the #873 terrain-curriculum
# reset state a difficulty ramp would evaluate into every episode.
#
# The terrain heightfield kinds + difficulty knob (#1336/#1338/#1339/#1340/
# #1344), the seat-on-terrain spawn (#1386), the terrain-relative fall/height
# predicates that measure clearance above the LOCAL surface (#1364/#1368), and
# the runnable benchmark composition (#1259) each landed with isolated coverage.
# Nothing exercised them TOGETHER -- a real benchmark eval on raised ground:
# tests/simulation/test_add_robot_seats_on_terrain.py checks the seat on a
# SYNTHETIC base with no benchmark, and tests/simulation/test_base_below_z_terrain.py
# checks the predicate in isolation on a flat-registered synthetic base. So a
# regression in the seat, the terrain-relative predicate, or the eval-loop's
# per-episode reset-seat could silently break a terrain-curriculum benchmark
# while every existing test still passes. These two tests close that seam on the
# real Unitree Go2 (the canonical quadruped-locomotion embodiment).
_TERRAIN = "pyramid"
_DIFFICULTY = 2.0


def _lowest_collision_geom_z(sim: Any) -> float:
    """World z of the lowest collidable robot geom (skips the ground plane / hfield)."""
    model, data = sim._world._model, sim._world._data
    zs = [
        float(data.geom_xpos[g][2])
        for g in range(model.ngeom)
        if not (model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0)
        and model.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_HFIELD, mujoco.mjtGeom.mjGEOM_PLANE)
    ]
    assert zs, "no collidable robot geoms found"
    return min(zs)


def test_go2_walk_forward_seats_and_runs_on_terrain() -> None:
    """go2_walk_forward composes with a raised-terrain world (the curriculum reset state).

    On a ``create_world(terrain=..., difficulty=...)`` heightfield the shipped
    benchmark must: (1) spawn the Go2 SEATED on the local surface -- its lowest
    collidable geom on/above the plateau, not buried below it (the #1386 seat,
    which the benchmark eval's per-episode ``reset`` relies on to start each
    episode on the terrain); (2) hold the standing-spawn contract there (a
    seated, upright Go2 on the plateau is neither already-``failure`` nor
    already-``success``); and (3) run the whole ``evaluate_benchmark`` harness
    end-to-end on the terrain world (a mock policy driving real physics), scoring
    the episode without error. Reverting the seat leaves the feet ~one plateau
    height (``TERRAIN_ELEVATION * difficulty``) below the surface, failing (1).
    """
    spec = _SHIPPED_SPECS["go2_walk_forward"]
    robot = spec["default_robot"]
    assert has_sim(robot), f"default_robot {robot!r} is not simulatable"

    trimmed = copy.deepcopy(spec)
    trimmed["max_steps"] = _SMOKE_STEPS
    register_benchmark("go2_walk_forward", DeclarativeBenchmark.from_dict(trimmed))

    sim: Any = create_simulation(backend="mujoco")
    sim.create_world(ground_plane=True, terrain=_TERRAIN, difficulty=_DIFFICULTY)
    sim.add_robot(robot)  # seats the free base on the terrain at spawn (#1386)
    try:
        ground = sim._ground_height_at(0.0, 0.0)
        assert ground > 0.1, "expected a genuinely raised plateau (pyramid peak at the centre)"

        # (1) SEATED: the Go2's lowest collidable geom clears the LOCAL surface;
        # a reverted seat would leave it ~`ground` metres buried.
        assert _lowest_collision_geom_z(sim) >= ground - 0.02, "Go2 spawned buried in the terrain"

        bench = get_benchmark("go2_walk_forward")
        assert bench is not None  # just registered above

        # (2) valid terrain-curriculum start: a seated, upright Go2 is not scored.
        assert bench.is_failure(sim) is False, "seated Go2 on the plateau spuriously trips a failure predicate"
        assert bench.is_success(sim) is False, "seated Go2 on the plateau spuriously satisfies success"

        # (3) full harness end-to-end on the terrain world.
        res = sim.evaluate_benchmark("go2_walk_forward", policy_provider="mock", n_episodes=1)
        assert res["status"] == "success", res
        payload = next(c["json"] for c in res["content"] if "json" in c)
        assert payload["episodes_completed"] == 1
        assert payload["success_measured"] is True
        assert math.isfinite(payload["avg_reward"]), "avg_reward is not finite on the terrain world"
    finally:
        sim.destroy()


def test_go2_walk_forward_fall_predicate_is_terrain_relative() -> None:
    """The shipped ``base_below_z`` fall predicate fires TERRAIN-RELATIVE on the plateau.

    Places the Go2 base COLLAPSED but LEVEL on the raised plateau at a height
    whose clearance ABOVE THE LOCAL GROUND is below the spec's ``base_below_z``
    threshold, while its ABSOLUTE world z stays ABOVE that threshold. A
    terrain-relative predicate (#1364) detects the collapse; an absolute-z test
    would silently MISS it on raised ground -- so a robot could sink to the
    terrain and flail to ``max_steps`` without the episode ever terminating. The
    two branches are asserted on one scene: the real terrain-relative sim fires
    the failure, and the same pose under a flat-ground (``_ground_height_at`` ->
    0.0) reading does not.
    """
    spec = _SHIPPED_SPECS["go2_walk_forward"]
    robot = spec["default_robot"]
    assert has_sim(robot), f"default_robot {robot!r} is not simulatable"
    threshold = next(
        c["z"] for c in spec["failure"]["any"] if c["predicate"] == "base_below_z"
    )  # 0.18 m for go2_walk_forward
    register_benchmark("go2_walk_forward", DeclarativeBenchmark.from_dict(copy.deepcopy(spec)))

    sim: Any = create_simulation(backend="mujoco")
    sim.create_world(ground_plane=True, terrain=_TERRAIN, difficulty=_DIFFICULTY)
    sim.add_robot(robot)
    try:
        model, data = sim._world._model, sim._world._data
        ground = sim._ground_height_at(0.0, 0.0)
        # A collapse whose clearance above the local plateau is below the
        # threshold, but whose absolute z stays above it (isolates the terrain
        # relativity). Requires a plateau taller than the threshold gap.
        collapsed_z = ground + threshold / 2.0
        assert collapsed_z > threshold, (
            "test setup: absolute z must stay above the threshold so only a "
            "terrain-relative predicate can fire (needs a tall enough plateau)"
        )
        jid = sim._robot_free_base_joint_id(model, sim._world.robots[robot])
        adr = int(model.jnt_qposadr[jid])
        data.qpos[adr : adr + 3] = [0.0, 0.0, collapsed_z]
        data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]  # level: base_tipped stays False
        mujoco.mj_forward(model, data)

        bench = get_benchmark("go2_walk_forward")
        assert bench is not None
        # Terrain-relative: clearance above the plateau < threshold -> fall detected.
        assert bench.is_failure(sim) is True, "terrain-relative base_below_z did not fire for a collapse on the plateau"

        # The same pose read as an ABSOLUTE-z world (no local terrain offset)
        # would MISS the collapse -- the regression this composition guards.
        original = type(sim)._ground_height_at
        type(sim)._ground_height_at = lambda self, x, y: 0.0  # type: ignore[method-assign]
        try:
            assert bench.is_failure(sim) is False, (
                "an absolute-z fall predicate would spuriously report the plateau "
                "collapse as still-standing (the miss #1364 fixes)"
            )
        finally:
            type(sim)._ground_height_at = original  # type: ignore[method-assign]
    finally:
        sim.destroy()
