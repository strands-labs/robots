"""The accepted ``solver=`` domain is the solvers that can drive a robot.

Newton resolves eight solver names, and only three of them integrate a rigid
articulated body. The other five fail in two different ways when handed a robot,
and neither is something a caller can act on:

* ``vbd``, ``style3d`` and ``mpm`` raise from inside Newton, naming a
  ``ModelBuilder`` the caller never touched.
* ``xpbd`` and ``semi_implicit`` build and step without moving a joint, so
  ``add_robot`` / ``send_action`` / ``step`` all report success over a frozen
  world -- the worse of the two, because nothing reports it at all.

The rule is a pure function over the solver *name*, so everything except the
construction cases below runs with Newton uninstalled: that is the install on
which a caller most needs the refusal to be the one they read.
"""

from __future__ import annotations

import importlib.util

import pytest

from strands_robots.simulation.newton.backend import (
    articulated_solver_error,
    articulated_solvers,
    solver_registry,
)

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

# Measured on newton 1.5.0 / warp 1.16.0 against a two-hinge arm: a commanded
# 0.9 rad target moved each of these by 0.899 rad.
DRIVES_A_ROBOT = ("featherstone", "kamino", "mujoco")

# The same probe left these at 0.0 rad, whether commanded or stepped under
# gravity alone, or refused the model from inside Newton.
CANNOT_DRIVE_A_ROBOT = ("mpm", "semi_implicit", "style3d", "vbd", "xpbd")


class TestThePartitionCoversTheRegistry:
    """Every resolvable solver is classified, so a new one cannot slip through."""

    def test_the_two_groups_partition_the_registry(self):
        assert set(DRIVES_A_ROBOT) | set(CANNOT_DRIVE_A_ROBOT) == set(solver_registry())
        assert not set(DRIVES_A_ROBOT) & set(CANNOT_DRIVE_A_ROBOT)

    def test_articulated_solvers_is_the_driving_group_in_registry_order(self):
        assert articulated_solvers() == tuple(n for n in solver_registry() if n in DRIVES_A_ROBOT)

    def test_every_refused_name_is_a_solver_newton_resolves(self):
        # A typo here would silently disable the refusal for the real name.
        for name in CANNOT_DRIVE_A_ROBOT:
            assert name in solver_registry()


class TestTheDomainReportsEveryUndrivableSolver:
    @pytest.mark.parametrize("solver", CANNOT_DRIVE_A_ROBOT)
    def test_it_names_the_solver_the_reason_and_the_alternatives(self, solver):
        message = articulated_solver_error(solver)
        assert message is not None
        assert repr(solver) in message
        assert "cannot drive an articulated robot" in message
        # The caller is told what to use instead, not just what failed.
        for usable in articulated_solvers():
            assert usable in message

    @pytest.mark.parametrize("solver", DRIVES_A_ROBOT)
    def test_a_solver_that_drives_a_robot_is_accepted(self, solver):
        assert articulated_solver_error(solver) is None

    def test_the_name_is_matched_case_insensitively(self):
        # __init__ lowercases only after this guard, so it must fold here too.
        refusal = articulated_solver_error("VBD")
        assert refusal is not None
        assert articulated_solver_error("MuJoCo") is None
        # The refusal quotes the spelling the caller passed, not the folded one,
        # so they can find it in their own call.
        assert "'VBD'" in refusal

    @pytest.mark.parametrize("solver", CANNOT_DRIVE_A_ROBOT)
    def test_the_reason_distinguishes_a_frozen_world_from_a_newton_error(self, solver):
        message = articulated_solver_error(solver)
        assert message is not None
        frozen = solver in ("xpbd", "semi_implicit")
        assert ("frozen" in message) is frozen


@pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")
class TestConstructionRefusesAnUndrivableSolver:
    @pytest.mark.parametrize("solver", CANNOT_DRIVE_A_ROBOT)
    def test_the_engine_refuses_it_with_the_shared_verdict(self, solver):
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        with pytest.raises(ValueError) as excinfo:
            NewtonSimEngine(solver=solver)
        # The engine adds nothing of its own to the shared rule's wording.
        assert str(excinfo.value) == articulated_solver_error(solver)

    def test_an_unknown_solver_still_reports_that_it_is_unknown(self):
        # Ordering pin: the membership refusal runs before this domain, so a
        # misspelling is not reported as an undrivable solver.
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        with pytest.raises(ValueError, match="Unknown Newton solver"):
            NewtonSimEngine(solver="not_a_solver")

    def test_describe_advertises_only_the_solvers_it_accepts(self):
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        sim = NewtonSimEngine(solver="mujoco")
        try:
            advertised = sim.describe()["available_solvers"]
        finally:
            sim.destroy()
        assert advertised == sorted(articulated_solvers())
        for refused in CANNOT_DRIVE_A_ROBOT:
            assert refused not in advertised
