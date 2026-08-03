"""Regression pins for the Isaac Sim 6.0.x reset-kills-articulation-handles fix.

Found running ``examples/so101_curobo/app.py --smoke --backend isaac``
end-to-end on the pip ``isaacsim`` wheels (verified on 6.0.0.1): on those
builds ``world.reset()`` tears down and rebuilds the physics-tensor
simulation view, and the per-robot ``SingleArticulation`` handles are left
holding the torn-down view. After ANY reset, ``get_joint_positions()``
returns ``None``, ``get_observation()`` degrades to its documented
silent-empty mode, and every consumer of post-reset joint state breaks (the
example's collector dies with a ``KeyError`` reading ``home_q`` from an
empty observation). Same defect family as #1798 - reset/stop invalidates
physics views; wrapper handles need an explicit re-init on 6.0.x - one layer
up: #1798 fixed the scene-object path, this pins the articulation path
(#1895).

The fix: ``IsaacSimulation.reset()`` probes every registered robot's
articulation after ``world.reset()`` completes and re-initializes the dead
handles against the fresh view (``_revive_articulations_after_reset``) -
prevent-and-revive, mirroring #1798, rather than catching the downstream
``None``s.

The tests drive the REAL ``IsaacSimulation.reset()`` / ``get_observation()``
on a real (Isaac-free) ``IsaacSimulation`` instance with a stub world whose
``reset()`` kills the registered articulation handles - the observed 6.0.x
behaviour - and stub articulations that revive on ``initialize()``. No kit
install required (the pattern of ``test_isaac_backend.py``'s
``TestMainThreadAffinityGuard`` and ``test_deferred_physics_and_warmup.py``).
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState


class _StubArticulation:
    """Stub ``SingleArticulation`` whose handle dies with the physics view.

    Mirrors the live 6.0.x behaviour: a handle holding a torn-down view
    returns ``None`` from ``get_joint_positions()`` (the silent mode the
    issue's repro shows), and ``initialize()`` against the fresh view
    revives it.
    """

    def __init__(self, num_joints: int = 6) -> None:
        self._num_joints = num_joints
        self.alive = True
        self.initialize_calls = 0
        self.initialized_with: list[object] = []

    def get_joint_positions(self):
        if not self.alive:
            return None
        return [0.0] * self._num_joints

    def initialize(self, physics_sim_view=None) -> None:
        self.initialize_calls += 1
        self.initialized_with.append(physics_sim_view)
        self.alive = True


class _RaisingProbeArticulation(_StubArticulation):
    """A dead handle that RAISES on the probe instead of returning ``None``.

    Some SDK surfaces raise ``RuntimeError`` from a handle whose view is
    gone rather than returning ``None``; the probe must treat both as dead.
    """

    def get_joint_positions(self):
        if not self.alive:
            raise RuntimeError("Physics Simulation View is not created yet")
        return super().get_joint_positions()


class _ReinitFailsArticulation(_StubArticulation):
    """A dead handle whose ``initialize()`` fails (partial teardown)."""

    def initialize(self, physics_sim_view=None) -> None:
        self.initialize_calls += 1
        raise RuntimeError("articulation prim view could not be created")


class _StubWorld:
    """Stub Isaac ``World`` whose ``reset()`` invalidates articulation handles.

    Pins the observed pip-wheel 6.0.x behaviour: ``world.reset()`` rebuilds
    the physics-tensor view (modelled as a fresh ``physics_sim_view``
    sentinel), killing every wrapper handle built against the old one.
    """

    def __init__(self, articulations: list[_StubArticulation], kills_handles: bool = True) -> None:
        self._articulations = articulations
        self._kills_handles = kills_handles
        self.physics_sim_view: object = object()
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        if self._kills_handles:
            self.physics_sim_view = object()  # fresh view; old handles are stale
            for art in self._articulations:
                art.alive = False


@pytest.fixture()
def make_sim():
    """Factory: a live-world ``IsaacSimulation`` around stub world + robots.

    Real ``__init__`` (no SimulationApp boots for it - the pattern of
    ``test_isaac_backend.py``); the stub world and ``_RobotState`` entries
    stand in for the kit-affine handles. Teardown detaches the stubs before
    GC so ``SimEngine.__del__ -> cleanup() -> destroy()`` never runs real
    teardown against them.
    """
    sims: list[IsaacSimulation] = []

    def _make(world: _StubWorld, robots: dict[str, _RobotState]) -> IsaacSimulation:
        sim = IsaacSimulation(num_envs=1, headless=True)
        sim._world_created = True
        sim._world = world
        sim._robots = robots
        sims.append(sim)
        return sim

    yield _make

    for sim in sims:
        sim._world_created = False
        sim._world = None
        sim._robots = {}


def _robot_state(name: str, articulation, joint_names: list[str] | None = None) -> _RobotState:
    return _RobotState(
        name=name,
        prim_path=f"/World/Robots/{name}",
        joint_names=joint_names or [f"j{i}" for i in range(1, 7)],
        articulation=articulation,
    )


class TestResetRevivesArticulationHandles:
    """``reset()`` must revive the handles ``world.reset()`` killed (#1895)."""

    def test_reset_revives_a_handle_world_reset_killed(self, make_sim) -> None:
        """The issue's minimal repro, pinned: reset -> handle alive again.

        Pre-fix behaviour: ``reset()`` called ``world.reset()`` and returned;
        the handle stayed dead and ``get_joint_positions()`` returned ``None``
        until a caller manually invoked ``articulation.initialize()``.
        """
        art = _StubArticulation()
        world = _StubWorld([art])
        sim = make_sim(world, {"arm": _robot_state("arm", art)})

        result = sim.reset()

        assert result["status"] == "success"
        assert world.reset_calls == 1
        assert art.initialize_calls == 1
        assert art.get_joint_positions() is not None

    def test_reinit_targets_the_fresh_physics_view(self, make_sim) -> None:
        """The revive initializes against the view ``world.reset()`` rebuilt.

        Initializing against the stale view would reproduce the defect one
        reset later; the handle must be handed ``world.physics_sim_view`` as
        it stands AFTER the reset.
        """
        art = _StubArticulation()
        world = _StubWorld([art])
        sim = make_sim(world, {"arm": _robot_state("arm", art)})

        sim.reset()

        assert art.initialized_with == [world.physics_sim_view]

    def test_get_observation_is_non_empty_after_reset(self, make_sim) -> None:
        """The end-to-end symptom: post-reset observation carries joint keys.

        This is the read the so101_curobo collector makes (``home_q`` from
        ``get_observation``); pre-fix it got ``{}`` (the documented
        silent-empty mode) and died on a ``KeyError``.
        """
        joint_names = [f"j{i}" for i in range(1, 7)]
        art = _StubArticulation(num_joints=6)
        world = _StubWorld([art])
        sim = make_sim(world, {"arm": _robot_state("arm", art, joint_names)})

        sim.reset()
        obs = sim.get_observation("arm")

        assert obs, "post-reset observation must not be empty"
        assert set(joint_names) <= set(obs)
        assert all(isinstance(obs[j], float) for j in joint_names)

    def test_every_registered_robot_is_revived(self, make_sim) -> None:
        """Multi-robot worlds revive ALL handles, not just the first."""
        arts = [_StubArticulation(), _StubArticulation()]
        world = _StubWorld(arts)
        sim = make_sim(
            world,
            {
                "left": _robot_state("left", arts[0]),
                "right": _robot_state("right", arts[1]),
            },
        )

        sim.reset()

        assert all(a.initialize_calls == 1 for a in arts)
        assert all(a.get_joint_positions() is not None for a in arts)

    def test_a_probe_that_raises_counts_as_dead(self, make_sim) -> None:
        """A dead handle raising on the probe is revived, not skipped."""
        art = _RaisingProbeArticulation()
        world = _StubWorld([art])
        sim = make_sim(world, {"arm": _robot_state("arm", art)})

        result = sim.reset()

        assert result["status"] == "success"
        assert art.initialize_calls == 1
        assert art.get_joint_positions() is not None


class TestResetLeavesHealthyStateAlone:
    """The revive is probe-gated: no gratuitous re-inits, no stub crashes."""

    def test_an_alive_handle_is_not_reinitialized(self, make_sim) -> None:
        """Builds whose reset keeps handles live pay one probe and nothing else.

        A full ``initialize()`` on a healthy handle is not free (it rebuilds
        the DOF metadata) and load_scene's measured PhysX instability around
        redundant re-init (#1798 trail) says: touch only what is broken.
        """
        art = _StubArticulation()
        world = _StubWorld([art], kills_handles=False)
        sim = make_sim(world, {"arm": _robot_state("arm", art)})

        result = sim.reset()

        assert result["status"] == "success"
        assert art.initialize_calls == 0

    def test_a_phase1_stub_without_a_handle_is_skipped(self, make_sim) -> None:
        """A robot with ``articulation=None`` (procedural / load stub) is skipped."""
        world = _StubWorld([])
        sim = make_sim(world, {"stub": _robot_state("stub", None)})

        result = sim.reset()

        assert result["status"] == "success"

    def test_a_world_with_no_robots_resets_cleanly(self, make_sim) -> None:
        """Empty-registry reset (create_world's own path) stays a no-op."""
        world = _StubWorld([])
        sim = make_sim(world, {})

        assert sim.reset()["status"] == "success"


class TestFailedReviveIsLoudButNotFatal:
    """Re-init failure matches load_scene's tolerance: warn loudly, keep going."""

    def test_failed_reinit_warns_naming_the_robot_and_reset_succeeds(self, make_sim, caplog) -> None:
        """A handle whose re-init fails is logged loudly; reset still succeeds.

        The warning must say WHICH robot and that its joint observations will
        be EMPTY - the operator-facing symptom - matching the load_scene
        re-init path's contract.
        """
        art = _ReinitFailsArticulation()
        world = _StubWorld([art])
        sim = make_sim(world, {"arm": _robot_state("arm", art)})

        with caplog.at_level(logging.WARNING):
            result = sim.reset()

        assert result["status"] == "success"
        warnings = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
        assert "'arm'" in warnings
        assert "EMPTY" in warnings

    def test_one_failed_reinit_does_not_starve_the_next_robot(self, make_sim) -> None:
        """Best-effort per robot: a failure on one must not skip the rest."""
        bad = _ReinitFailsArticulation()
        good = _StubArticulation()
        world = _StubWorld([bad, good])
        sim = make_sim(
            world,
            {
                "bad": _robot_state("bad", bad),
                "good": _robot_state("good", good),
            },
        )

        result = sim.reset()

        assert result["status"] == "success"
        assert good.initialize_calls == 1
        assert good.get_joint_positions() is not None
