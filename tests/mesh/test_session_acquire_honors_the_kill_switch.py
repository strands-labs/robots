"""``STRANDS_MESH=false`` stops a session from being acquired, not just constructed.

README documents the variable as "a hard kill switch that also overrides an explicit
``mesh=True``", and :func:`~strands_robots.mesh.core.mesh_disabled_by_env` states the
contract this file grades: an operator who sets it "is asking for no Zenoh session and no
presence on the fleet, so every path that can open one answers this -- not only
``init_mesh``".

That claim was true of the two places a ``Mesh`` is CONSTRUCTED (``init_mesh`` and
``robot_mesh._gateway_mesh``, which each ask separately) and false of the one place a
session is OPENED. So any caller that acquires the transport directly -- ``ZenohTransport``,
the bridge factory, an integration harness, the ``get_session()`` entry point the module
docstring advertises -- reached ``zenoh.open`` with the switch engaged. What arrived was
not one quiet extra peer: with no explicit endpoints that path LISTENS on
``STRANDS_MESH_PORT``, so the process the operator disabled the mesh on became the
machine's hub and every later process on the box connected to it as a client.

Graded three ways, because each catches a different regression:

* behaviourally, over the kill-switch vocabulary its owner declares, asserting no
  ``zenoh.open`` was ATTEMPTED rather than only that ``None`` came back;
* in the affirmative and unset directions, so a guard that refuses everything is not
  mistaken for one that refuses a kill;
* structurally, over every function in the module that reaches ``zenoh.open``, so a
  third acquire door added later is caught the day it lands rather than the day an
  operator finds a hub they had switched off.
"""

from __future__ import annotations

import ast
import inspect
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import strands_robots.mesh.session as session_mod
from strands_robots._mesh_switch import AFFIRMATIVE, NEGATIVE

#: Both doors that reach ``zenoh.open``. Declared here for the behavioural cells and
#: checked against the module itself by :class:`TestEveryAcquireDoorAsksTheSwitch`.
_SESSION_OPENERS = ("get_session", "_get_zenoh_session_directly")


@pytest.fixture(autouse=True)
def _fresh_session_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """No cached session, and the raw Zenoh path rather than the transport factory."""
    monkeypatch.setattr(session_mod, "_SESSION", None)
    monkeypatch.setattr(session_mod, "_SESSION_REFS", 0)
    monkeypatch.delenv("STRANDS_MESH_BACKEND", raising=False)


def _acquire(opener: str, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, MagicMock]:
    """Call one acquire door against a stand-in ``zenoh``, reporting what it opened.

    Returns the session the door handed back and the stand-in module, so a cell can
    assert on the ATTEMPT rather than only on the return value: a door that opened a
    session and then discarded it would return ``None`` too, and only a door that never
    called ``zenoh.open`` leaves ``STRANDS_MESH_PORT`` unbound.

    ``_build_config`` is stubbed for the same reason the rest of this package's session
    tests stub it -- config building needs the real extra, and the topology is not what
    is under test here.
    """
    for var in ("ZENOH_LISTEN", "ZENOH_CONNECT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(session_mod, "_build_config", MagicMock(return_value=MagicMock()))
    zenoh = MagicMock()
    with patch.dict("sys.modules", {"zenoh": zenoh}):
        return getattr(session_mod, opener)(), zenoh


class TestTheSwitchIsAskedBeforeAnythingIsOpened:
    """The engaged switch is honoured at the resource, for every spelling of it."""

    @pytest.mark.parametrize("opener", _SESSION_OPENERS)
    @pytest.mark.parametrize("spelling", NEGATIVE)
    def test_no_session_is_opened(self, opener: str, spelling: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH", spelling)

        result, zenoh = _acquire(opener, monkeypatch)

        assert result is None, f"{opener}() handed back a session with STRANDS_MESH={spelling!r}"
        assert zenoh.open.call_count == 0, (
            f"{opener}() called zenoh.open {zenoh.open.call_count} time(s) with "
            f"STRANDS_MESH={spelling!r}. With no explicit endpoints that call LISTENS, so the "
            "process the operator switched the mesh off on becomes the machine's hub."
        )

    def test_the_iot_backend_is_refused_without_reaching_the_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The switch is about presence, not about Zenoh.

        An IoT/bridge transport advertises this robot to the fleet exactly as a Zenoh
        session does, so the question is asked before the backend branch and one gate
        covers every backend. Graded by proving the factory is never consulted.

        Only ``get_session`` has that branch: ``_get_zenoh_session_directly`` exists
        precisely to bypass the factory, so asserting it does not reach one would be
        vacuous there.
        """
        monkeypatch.setenv("STRANDS_MESH", "false")
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        from strands_robots.mesh.transport import factory

        consulted: list[str] = []
        monkeypatch.setattr(factory, "get_transport", lambda: consulted.append("get_transport"))

        session, _ = _acquire("get_session", monkeypatch)

        assert session is None
        assert consulted == [], "the kill switch was asked after the backend branch, not before it"

    def test_an_honoured_request_is_not_reported_as_a_degradation(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No WARNING: the operator asked for this outcome and got it.

        ``tests/mesh/test_zenoh_absent_is_reported.py`` requires WARNING for the other
        way these functions end with no session, and the difference is the point. An
        absent ``eclipse-zenoh`` is a degradation nobody asked for and undiagnosable
        without a name; a granted kill switch is the requested result, and warning about
        it on every acquire would train an operator to ignore the level.
        """
        monkeypatch.setenv("STRANDS_MESH", "false")
        with caplog.at_level(logging.DEBUG, logger="strands_robots.mesh.session"):
            session, _ = _acquire("get_session", monkeypatch)
        assert session is None

        assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
            "a granted kill switch was reported at WARNING or above: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )
        assert [r for r in caplog.records if "STRANDS_MESH" in r.getMessage()], (
            "nothing named STRANDS_MESH, so an operator debugging a mesh that will not "
            f"come up learns nothing from the log: {[r.getMessage() for r in caplog.records]}"
        )


class TestTheGuardRefusesAKillAndNothingElse:
    """Controls: a guard that refuses every acquire would pass the tests above."""

    @pytest.mark.parametrize("opener", _SESSION_OPENERS)
    def test_an_unset_variable_still_opens(self, opener: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("STRANDS_MESH", raising=False)

        session, zenoh = _acquire(opener, monkeypatch)

        assert session is not None
        assert zenoh.open.call_count == 1, f"{opener}() opened nothing with STRANDS_MESH unset"

    @pytest.mark.parametrize("opener", _SESSION_OPENERS)
    @pytest.mark.parametrize("spelling", AFFIRMATIVE)
    def test_an_affirmative_spelling_still_opens(
        self, opener: str, spelling: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``true`` is not a kill: the switch is one-directional."""
        monkeypatch.setenv("STRANDS_MESH", spelling)

        session, zenoh = _acquire(opener, monkeypatch)

        assert session is not None
        assert zenoh.open.call_count == 1, f"{opener}() opened nothing with STRANDS_MESH={spelling!r}"

    @pytest.mark.parametrize("opener", _SESSION_OPENERS)
    def test_an_unrecognized_spelling_still_opens(self, opener: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """``off`` is neither half of the vocabulary, and this guard does not decide that.

        Whether ``off`` should MEAN ``false`` is a behaviour change on a safety switch that
        :mod:`strands_robots._mesh_switch` deliberately does not make; it warns instead. A
        guard that quietly widened the vocabulary would make that warning a lie.
        """
        monkeypatch.setenv("STRANDS_MESH", "off")

        session, zenoh = _acquire(opener, monkeypatch)

        assert session is not None
        assert zenoh.open.call_count == 1, f"{opener}() treated an unrecognized spelling as a kill"


class TestEveryAcquireDoorAsksTheSwitch:
    """Structural: the population is derived from the module, not listed by hand.

    A hand-listed population is silent in the reassuring direction -- a third acquire
    door would not be reported as ungraded, it would leave this file green.
    """

    @staticmethod
    def _acquire_doors(source: str) -> dict[str, set[str]]:
        """Map each module-level function reaching ``zenoh.open`` to the names it calls."""
        doors: dict[str, set[str]] = {}
        for node in ast.parse(source).body:
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            called: set[str] = set()
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    func = inner.func
                    if isinstance(func, ast.Attribute):
                        called.add(ast.unparse(func))
                    elif isinstance(func, ast.Name):
                        called.add(func.id)
            if "zenoh.open" in called:
                doors[node.name] = called
        return doors

    def test_every_door_reaching_zenoh_open_consults_the_kill_switch(self) -> None:
        doors = self._acquire_doors(Path(inspect.getfile(session_mod)).read_text(encoding="utf-8"))

        assert len(doors) >= len(_SESSION_OPENERS), (
            f"found {sorted(doors)} but this file grades {len(_SESSION_OPENERS)} doors "
            "behaviourally -- the derivation stopped seeing a door it used to see"
        )
        ungated = sorted(name for name, called in doors.items() if "mesh_disabled_by_env" not in called)
        assert not ungated, (
            f"{ungated} reach zenoh.open without asking mesh_disabled_by_env(). "
            "STRANDS_MESH=false is documented as a hard kill switch, and a door that does "
            "not ask it opens a listener the operator switched off."
        )

    def test_the_derivation_selects_a_door_and_rejects_a_neighbour(self) -> None:
        """The predicate, on planted source: selecting nothing must not read as clean."""
        planted = (
            "def opens():\n"
            "    return zenoh.open(cfg)\n"
            "def releases():\n"
            "    _SESSION.close()\n"
            "def guarded():\n"
            "    if mesh_disabled_by_env():\n"
            "        return None\n"
            "    return zenoh.open(cfg)\n"
        )
        doors = self._acquire_doors(planted)

        assert sorted(doors) == ["guarded", "opens"], "a release path is not an acquire door"
        assert "mesh_disabled_by_env" not in doors["opens"]
        assert "mesh_disabled_by_env" in doors["guarded"]
