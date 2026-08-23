"""A stop that fails must answer, so the e-stop accounting can see it.

``Mesh._dispatch``'s ``action="stop"`` branch has three sub-paths that can
raise: the hardware ``stop_task`` call, the sim ``stop_policy`` call and the
child-sim delegation. Two of them were wrapped, with the contract stated in the
handler itself -- ``except Exception ... # stop must answer, not raise`` -- and
the hardware one was not.

An exception escaping the hardware path reaches ``Mesh._exec_cmd``, which
publishes ``{"type": "error", "error": "dispatch error"}``. That envelope
carries no ``result``, so :func:`~strands_robots.mesh.core._peers_that_did_not_stop`
falls back to reading the envelope itself and finds neither ``ok`` nor
``status`` -- and ``emergency_stop`` counts the peer among its
acknowledgements. The arm whose stop failed is reported as halted, which the
e-stop honesty module calls the worst available failure mode on a safety path.

Raising is the only way a hardware stop can report failure: both return paths of
``Robot.stop_task`` answer ``status="success"``.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import textwrap
from typing import Any

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh.core import Mesh, _peers_that_did_not_stop

BUS_FAULT = "serial bus write failed mid-stop"


class _FaultingArm:
    """A hardware arm whose stop fails the way a real bus fault does."""

    def stop_task(self) -> dict[str, Any]:
        raise RuntimeError(BUS_FAULT)


class _StoppableArm:
    def stop_task(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": "stopped"}]}


class _NoStopArm:
    def get_task_status(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": "idle"}]}


class _FaultingSim:
    """A sim peer whose ``stop_policy`` raises -- the already-wrapped sibling."""

    _world = object()

    def stop_policy(self, robot_name: str) -> dict[str, Any]:
        raise RuntimeError("world already destroyed")


class _RefusingSim:
    """A sim peer whose ``stop_policy`` refuses by returning an error envelope."""

    _world = object()

    def stop_policy(self, robot_name: str) -> dict[str, Any]:
        return {"status": "error", "content": [{"text": "unknown robot 'arm'"}]}


def _wire_reply(robot: Any, peer_id: str, cmd: dict[str, Any] | None = None) -> dict[str, Any]:
    """Drive the real receive path and return the envelope actually published."""
    mesh = Mesh(robot, peer_id=peer_id)
    published: list[dict[str, Any]] = []
    mesh.publish = lambda key, payload: published.append(payload)  # type: ignore[method-assign]
    mesh._running = True
    mesh._exec_cmd(
        {
            "sender_id": "operator",
            "turn_id": f"t-{peer_id}",
            "command": dict(cmd) if cmd else {"action": "stop"},
        }
    )
    assert published, "premise: the receive path published no reply at all"
    return published[-1]


class TestAFailedHardwareStopIsCountedAsNotStopped:
    def test_the_wire_reply_is_one_the_estop_accounting_reads(self) -> None:
        reply = _wire_reply(_FaultingArm(), "arm-fault")

        flagged = _peers_that_did_not_stop([reply])

        if not flagged:
            pytest.fail(
                "a hardware stop_task that raised produced "
                f"{json.dumps(reply, default=str)}, which the e-stop accounting "
                "reads as an acknowledgement -- the arm is reported as halted"
            )
        assert flagged == {"arm-fault"}

    def test_emergency_stop_names_the_peer_in_peers_not_stopped(self) -> None:
        """End to end: real replies through the real accounting."""
        replies = [
            _wire_reply(_StoppableArm(), "arm-ok"),
            _wire_reply(_NoStopArm(), "arm-nostop"),
            _wire_reply(_FaultingArm(), "arm-fault"),
        ]
        operator = Mesh(_StoppableArm(), peer_id="operator")
        operator._running = True
        operator.broadcast = lambda cmd, timeout=3.0: replies  # type: ignore[method-assign]
        operator.publish = lambda key, payload: None  # type: ignore[method-assign]
        captured: dict[str, Any] = {}
        operator._publish_safety_envelope = lambda key, payload: captured.setdefault(  # type: ignore[method-assign]
            "envelope", payload
        )
        operator.publish_safety_event = lambda **kw: None  # type: ignore[method-assign]

        operator.emergency_stop()

        not_stopped = captured["envelope"]["peers_not_stopped"]
        assert "arm-fault" in not_stopped, f"two of three peers did not stop; the safety envelope named {not_stopped}"
        assert sorted(not_stopped) == ["arm-fault", "arm-nostop"]

    def test_the_answer_carries_the_reason(self) -> None:
        out = Mesh(_FaultingArm(), peer_id="p")._dispatch({"action": "stop"})

        assert out["ok"] is False
        assert "stop_task failed" in out["error"]
        assert BUS_FAULT in out["error"]

    def test_the_failure_is_loud_in_the_local_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """The operator on the failing peer must see it, not only the requester."""
        with caplog.at_level(logging.ERROR):
            Mesh(_FaultingArm(), peer_id="p")._dispatch({"action": "stop"})

        loud = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("stop_task" in m and "NOTHING was stopped" in m for m in loud), loud


class TestEveryRaisingStopPathAnswers:
    """Root cause: the stop branch may not let a stop call raise."""

    @staticmethod
    def _stop_branch() -> ast.If:
        source = textwrap.dedent(inspect.getsource(Mesh._dispatch))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.If) and "action == 'stop'" in ast.unparse(node.test):
                return node
        # raise rather than pytest.fail: fail() is only terminal at runtime, so a
        # helper that returns a value and calls it mixes an explicit return with
        # an implicit one. Raising makes every path return or raise.
        raise AssertionError("could not locate the action=='stop' branch of Mesh._dispatch")

    @classmethod
    def _stop_calls(cls) -> list[tuple[str, bool]]:
        """Every ``stop_*`` call in the branch, with whether it sits in a ``try``."""
        branch = cls._stop_branch()
        guarded: set[int] = set()
        for node in ast.walk(branch):
            if isinstance(node, ast.Try):
                for statement in node.body:
                    guarded.update(id(inner) for inner in ast.walk(statement))
        found: list[tuple[str, bool]] = []
        for node in ast.walk(branch):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("stop_task", "stop_policy")
            ):
                found.append((ast.unparse(node), id(node) in guarded))
        return found

    def test_the_scan_found_the_stop_calls(self) -> None:
        """Non-vacuity: a scan that reaches nothing would report a clean branch."""
        calls = self._stop_calls()
        assert len(calls) >= 3, f"expected at least three stop calls, found {calls}"
        assert any("stop_task" in expression for expression, _ in calls), calls
        assert any("stop_policy" in expression for expression, _ in calls), calls

    def test_no_stop_call_is_left_to_raise(self) -> None:
        unguarded = [expression for expression, is_guarded in self._stop_calls() if not is_guarded]
        assert not unguarded, (
            "these stop calls can raise out of _dispatch, so _exec_cmd answers with a "
            f"type='error' envelope the e-stop accounting cannot read: {unguarded}"
        )


class TestNothingElseChanges:
    def test_a_successful_hardware_stop_is_passed_through_verbatim(self) -> None:
        out = Mesh(_StoppableArm(), peer_id="p")._dispatch({"action": "stop"})

        assert out == {"status": "success", "content": [{"text": "stopped"}]}

    def test_a_peer_with_no_stop_task_keeps_its_message(self) -> None:
        out = Mesh(_NoStopArm(), peer_id="p")._dispatch({"action": "stop"})

        assert out["ok"] is False
        assert out["error"] == "peer exposes no stop_task; nothing was stopped"

    def test_the_sim_paths_keep_their_own_wording(self) -> None:
        """The already-wrapped sibling this fix mirrors must be untouched."""
        out = Mesh(_FaultingSim(), peer_id="p")._dispatch({"action": "stop", "robot_name": "arm"})

        assert out["ok"] is False
        assert out["error"].startswith("stop_policy failed:")

    def test_a_sim_refusal_is_still_flagged_through_its_status_key(self) -> None:
        """The third documented shape: an error envelope returned, not raised.

        Driven through ``_dispatch`` rather than the wire because
        ``validate_command`` does not carry ``robot_name``, so the per-robot sim
        branch is reachable only by a direct call.
        """
        out = Mesh(_RefusingSim(), peer_id="sim-refuse")._dispatch({"action": "stop", "robot_name": "arm"})

        assert out["status"] == "error", out
        assert _peers_that_did_not_stop([{"responder_id": "sim-refuse", "result": out}]) == {"sim-refuse"}

    def test_an_error_envelope_is_still_not_flagged_by_the_classifier(self) -> None:
        """Scope boundary: this fix does not widen the classifier.

        A ``type="error"`` reply also arrives for a validation rejection, a
        non-dict envelope and a lockout refusal. The lockout refusal is
        deliberately indistinguishable from the others -- ``_dispatch`` raises a
        generic ``LockoutError("command rejected")`` so a remote caller cannot
        map the lockout window -- and a peer already in lockout was stopped by
        the e-stop that engaged it. Reading every ``type="error"`` as "did not
        stop" would therefore report an already-halted peer as possibly moving,
        which is the false alarm ``_peers_that_did_not_stop`` is written to
        avoid. That trade is a contract decision, not this fix.
        """
        envelope = {"type": "error", "responder_id": "arm-x", "turn_id": "t", "error": "validation: nope"}

        assert _peers_that_did_not_stop([envelope]) == set()

    def test_a_non_stop_action_still_sanitises_its_dispatch_error(self) -> None:
        """The generic dispatch-error path must keep hiding internal detail."""
        mesh = Mesh(_StoppableArm(), peer_id="me")
        published: list[dict[str, Any]] = []
        mesh.publish = lambda key, payload: published.append(payload)  # type: ignore[method-assign]
        mesh._running = True
        mesh._dispatch = lambda cmd: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeError("boom-with-internal-detail")
        )

        mesh._exec_cmd({"sender_id": "operator", "turn_id": "t", "command": {"action": "status"}})

        assert published[-1]["type"] == "error"
        assert published[-1]["error"] == "dispatch error"
        assert "boom-with-internal-detail" not in published[-1]["error"]

    def test_the_docstring_names_the_shapes_the_dispatch_can_produce(self) -> None:
        """Each shape the census names must be reachable from ``_dispatch``."""
        census = inspect.getdoc(mesh_core._peers_that_did_not_stop) or ""
        normalised = " ".join(census.split())

        assert "stop_task failed" in normalised, normalised
        assert '{"ok": False, ...}' in normalised, normalised
        assert '{"status": "error", ...}' in normalised, normalised
