"""K2: bind validates before it acquires, and releases what it acquired on every exit path.

A ``Probe`` policy fails at one named hook; every run, single or lock-step, must detach exactly when it attached.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots import Robot, Simulation
from strands_robots.policy.requirements import Requirements

HOOKS = ("attach", "hz", "requirements", "reset", "act", "detach")
IN_BIND = ("hz", "requirements", "reset")  # after attach, before step one


class Boom(RuntimeError):
    pass


class Probe:
    """A policy that can fail at any one of its hooks; it logs every hook it was asked to run."""

    def __init__(self, fail_at: str | None = None, gains: bool = False) -> None:
        self.fail_at, self.log = fail_at, []
        if gains:
            self.gains = (np.full(6, 7.0), np.full(6, 0.7))

    def _maybe(self, hook: str) -> None:
        self.log.append(hook)
        if self.fail_at == hook:
            raise Boom(hook)

    def attach(self, robot):
        self._maybe("attach")

    @property
    def hz(self):
        self._maybe("hz")
        return 50.0

    def requirements(self):
        self._maybe("requirements")
        return Requirements()

    def reset(self):
        self._maybe("reset")

    def act(self, obs, instruction=None):
        self._maybe("act")
        return {}

    def detach(self):
        self._maybe("detach")

    @property
    def open(self) -> int:
        """Sessions this policy holds: attached and not yet detached."""
        return int(self.log.count("attach") > 0 and self.fail_at != "attach") - self.log.count("detach")


@pytest.mark.parametrize("gains", [False, True], ids=["no_gains", "gains"])
@pytest.mark.parametrize("fail_at", [None, *HOOKS])
def test_k2_every_exit_path_releases_what_bind_acquired(fail_at, gains):
    with Simulation() as sim:
        arm, pol = Robot("so101", sim=sim), Probe(fail_at, gains)
        if fail_at is None:
            arm.run(pol, steps=2)
        else:
            with pytest.raises(Boom):
                arm.run(pol, steps=2)
        assert pol.open == 0 and pol.log.count("detach") <= 1, pol.log  # released once, exactly when acquired
        if fail_at in ("attach", *IN_BIND):
            assert "act" not in pol.log  # a refused bind never reaches step one


def test_k2_a_named_scheme_is_checked_before_attach_opens_a_session():
    with Simulation() as sim:
        arm, pol = Robot("so101", sim=sim), Probe()
        with pytest.raises(ValueError, match="unknown units"):
            arm.run(pol, units="bogus", steps=1)
        assert pol.log == []


def test_k2_a_listener_that_raises_on_run_started_still_releases_the_session():
    class Loud:
        def emit(self, event, payload):
            raise Boom(event)

    with Simulation() as sim:
        arm, pol = Robot("so101", sim=sim), Probe()
        arm.events = Loud()
        with pytest.raises(Boom, match="run.started"):
            arm.run(pol, steps=1)
        assert pol.open == 0, pol.log


@pytest.mark.parametrize("fail_at", IN_BIND)
def test_k2_a_partner_refused_in_lockstep_leaves_nothing_open(fail_at):
    with Simulation() as sim:
        a, b = Robot("so101", sim=sim), Robot("so101", sim=sim, position=[0.5, 0, 0])
        ok, bad = Probe(), Probe(fail_at)
        with pytest.raises(Boom):
            sim.run({a: ok, b: bad}, steps=2)
        assert (ok.open, bad.open) == (0, 0), (ok.log, bad.log)
        assert "act" not in ok.log + bad.log


def test_k2_nothing_stays_open_across_many_refused_runs():
    with Simulation() as sim:
        arm, probes = Robot("so101", sim=sim), []
        for i in range(30):
            pol = Probe(HOOKS[i % len(HOOKS)])
            probes.append(pol)
            with pytest.raises(Boom):
                arm.run(pol, steps=1)
        arm.run(Probe(), steps=1)  # the body is still usable after every refusal
        assert sum(p.open for p in probes) == 0
