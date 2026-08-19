"""A missing ``eclipse-zenoh`` is reported at the level a consumer sees.

The mesh is optional, and a robot that does not need it must keep running when
the extra is absent -- so the session layer returns ``None`` and ``Mesh.start``
returns early rather than raising.  What made that degradation undiagnosable was
the *level*: both session-open paths reported the absent dependency at DEBUG, so
under default logging nothing named it and the first observable symptom was
whichever downstream wait expired first (in the fleet examples, a 15 s
presence-discovery timeout whose message mentions no dependency at all).

Every other way those functions end with no session already reports at WARNING,
and even a bad ``STRANDS_MESH_PORT`` -- which still yields a working mesh --
warns.  The most total outcome was the quietest one.  These tests pin the level,
the actionability of the remedy, that it is said once per process, and that
warning did not turn into raising.
"""

from __future__ import annotations

import ast
import inspect
import logging
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import strands_robots.mesh.session as session_mod
from strands_robots.mesh.core import Mesh

_REPO_ROOT = Path(session_mod.__file__).resolve().parents[2]

# Both session-open entry points reach the same lazy ``import zenoh``.  The
# public one routes ordinary callers; the private one is what a
# ``BridgeTransport``'s zenoh leg uses, and it carried the same DEBUG report.
_SESSION_OPENERS = ("get_session", "_get_zenoh_session_directly")


@pytest.fixture(autouse=True)
def _fresh_session_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no cached session and nothing reported yet."""
    monkeypatch.setattr(session_mod, "_SESSION", None)
    monkeypatch.setattr(session_mod, "_SESSION_REFS", 0)
    monkeypatch.setattr(session_mod, "_zenoh_missing_warned", set(), raising=False)
    # Keep get_session on the raw zenoh path rather than the transport factory.
    monkeypatch.delenv("STRANDS_MESH_BACKEND", raising=False)


def _open_without_zenoh(opener: str) -> Any:
    """Call one session opener with ``import zenoh`` failing.

    A ``None`` entry in ``sys.modules`` is what the interpreter treats as a
    halted import, so this reproduces an absent extra without touching the
    installed one.
    """
    with patch.dict("sys.modules", {"zenoh": None}):
        return getattr(session_mod, opener)()


class TestTheAbsentDependencyIsNamedAtWarning:
    """The cause reaches a default-configured consumer."""

    @pytest.mark.parametrize("opener", _SESSION_OPENERS)
    def test_every_session_opener_reports_it(self, opener: str, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.session"):
            result = _open_without_zenoh(opener)

        assert result is None, "premise: the opener must have taken the missing-dependency branch"
        named = [r for r in caplog.records if "eclipse-zenoh" in r.getMessage()]
        assert named, (
            f"{opener}() returned None because eclipse-zenoh is absent, but no record at WARNING or "
            f"above names it, so a default-configured consumer learns nothing. Records seen: "
            f"{[(r.levelname, r.getMessage()[:60]) for r in caplog.records]}"
        )

    def test_mesh_start_names_the_peer_whose_mesh_stayed_off(self, caplog: pytest.LogCaptureFixture) -> None:
        mesh = Mesh(SimpleNamespace(name="arm"), peer_id="bot-7")

        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
            with patch("strands_robots.mesh.core.get_session", return_value=None):
                mesh.start()

        assert mesh.alive is False, "premise: start() must have taken the no-transport branch"
        named = [r for r in caplog.records if "bot-7" in r.getMessage() and "mesh off" in r.getMessage()]
        assert named, (
            "start() left mesh.alive False and reported nothing at WARNING, so the caller finds out "
            f"from whichever downstream wait expires first. Records seen: "
            f"{[(r.levelname, r.getMessage()[:60]) for r in caplog.records]}"
        )


class TestTheRemedyIsRunnable:
    """The message quotes an extra that really supplies the dependency."""

    def test_the_named_extra_declares_eclipse_zenoh(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.session"):
            _open_without_zenoh("get_session")

        text = "\n".join(r.getMessage() for r in caplog.records)
        offered = re.findall(r"strands-robots\[([a-z0-9-]+)\]", text)
        assert offered, f"the report offers no installable extra: {text!r}"

        declared = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
        for extra in offered:
            assert extra in declared, f"the report names extra {extra!r}, which pyproject does not declare"
            assert any("eclipse-zenoh" in req for req in declared[extra]), (
                f"extra {extra!r} is declared but does not supply eclipse-zenoh, so following the "
                f"report leaves the mesh off: {declared[extra]}"
            )


class TestItIsSaidOncePerProcess:
    """A static fact is not repeated once per robot."""

    def test_repeated_opens_report_once(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.session"):
            for opener in (*_SESSION_OPENERS, *_SESSION_OPENERS):
                _open_without_zenoh(opener)

        named = [r for r in caplog.records if "eclipse-zenoh" in r.getMessage()]
        assert len(named) == 1, (
            f"four session opens reported the same absent dependency {len(named)} times; a fleet of "
            "robots in one process would bury the rest of the log"
        )


class TestTheReportIsAttachedToTheMissingDependencyOnly:
    """The root cause: the warning belongs to the ImportError branch."""

    def test_every_call_sits_in_an_importerror_handler(self) -> None:
        tree = ast.parse(inspect.getsource(session_mod))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_report_zenoh_missing"
        ]
        assert len(calls) == len(_SESSION_OPENERS), (
            f"expected one report per session opener ({len(_SESSION_OPENERS)}), found {len(calls)}"
        )

        guarded: set[int] = set()
        for handler in (n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)):
            caught = handler.type
            names = (
                {caught.id}
                if isinstance(caught, ast.Name)
                else {e.id for e in getattr(caught, "elts", []) if isinstance(e, ast.Name)}
            )
            if "ImportError" not in names:
                continue
            guarded |= {id(n) for n in ast.walk(handler) if isinstance(n, ast.Call)}

        for call in calls:
            assert id(call) in guarded, (
                f"_report_zenoh_missing() is called at line {call.lineno} outside an ImportError "
                "handler, so it would blame a missing dependency for some other failure"
            )


class TestReportingDidNotBecomeRaising:
    """Controls: the optional-dependency contract is unchanged."""

    @pytest.mark.parametrize("opener", _SESSION_OPENERS)
    def test_the_opener_still_returns_none(self, opener: str) -> None:
        result = _open_without_zenoh(opener)

        assert result is None
        assert session_mod._SESSION is None

    def test_mesh_start_is_still_a_clean_no_op(self) -> None:
        mesh = Mesh(SimpleNamespace(name="arm"), peer_id="bot-8")

        with patch("strands_robots.mesh.core.get_session", return_value=None):
            mesh.start()

        assert mesh.alive is False

    def test_init_mesh_returns_a_mesh_rather_than_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The suite sets STRANDS_MESH=false process-wide so tests never touch a
        # real mesh, and that kill switch makes init_mesh return None before it
        # reaches a session at all.  Clear it so this asserts the behaviour under
        # an absent dependency rather than under the switch.
        monkeypatch.delenv("STRANDS_MESH", raising=False)

        from strands_robots.mesh.core import init_mesh

        with patch("strands_robots.mesh.core.get_session", return_value=None):
            mesh = init_mesh(SimpleNamespace(name="arm", tool_name_str="arm"), peer_id="bot-9")

        assert mesh is not None
        assert mesh.alive is False
