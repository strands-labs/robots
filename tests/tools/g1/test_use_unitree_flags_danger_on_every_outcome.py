"""A ``use_unitree`` call's danger classification survives the call failing.

:mod:`~strands_robots.tools.g1.use_unitree` is the raw escape hatch to the
Unitree SDK2 clients, and the thing that makes it safe to hand an agent is that
the response says what the call *was*: ``mutative`` for anything that writes,
``high_danger`` for the seven pairs in
:data:`~strands_robots.tools.g1.use_unitree.HIGH_DANGER_OPS` that can drop or
walk the robot. Its sibling
``test_use_unitree_meta_discovery_is_robot_free.py`` pins that table and the
predicates that read it; this file pins the only thing a caller can actually
see, which is the envelope.

The distinction matters because the classification is most load-bearing on the
outcome where the call did *not* cleanly succeed. An RPC that times out is not
evidence the command never landed - ``loco.SetVelocity`` answering
``RPC_CLIENT_API_TIMEOUT`` is consistent with a robot that is walking - so an
agent or an operator reading that envelope has to be able to tell a failed
``ZeroTorque`` from a failed ``GetFsmId``. Absent keys cannot express that:
``res.get("high_danger")`` is falsy both for a read and for a collapse command
whose fate is unknown, which reads the classification exactly backwards.

So the pin is that the two flags are present and correct on EVERY outcome, and
present with ``False`` rather than absent for a harmless call - a flag that only
appears when it is ``True`` cannot be distinguished from a surface that forgot to
set it.
"""

from __future__ import annotations

from typing import Any

import pytest

import strands_robots.tools.g1.use_unitree as uu

#: A read: not mutative, not high-danger. The control every parametrized failure
#: row is graded against, so a pin that flagged everything would fail here.
HARMLESS_READ = ("loco", "GetFsmId")


class _Loco:
    """Stand-in for ``LocoClient`` with the four call outcomes the SDK produces.

    The real client is reached only through ``getattr(client, operation_name)``,
    so a class carrying the operation names under test is a faithful stand-in for
    everything :func:`~strands_robots.tools.g1.use_unitree._execute` does with
    it: attribute lookup, signature bind, then the call.
    """

    def ZeroTorque(self) -> int:
        return 0

    def SetVelocity(self, vx: float, vy: float, vyaw: float, duration: float = 1.0) -> int:
        return 0

    def Move(self, vx: float, vy: float, vyaw: float) -> int:
        return 0

    def WaveHand(self, turn_flag: bool = False) -> int:
        return 0

    def ShakeHand(self, stage: int = -1) -> int:
        return 0

    def SetFsmId(self, fsm_id: int) -> int:
        return 0

    def GetFsmId(self) -> tuple[int, int]:
        return (0, 801)


class _MotionSwitcher:
    def ReleaseMode(self) -> int:
        return 0


#: The SDK class each service resolves to, standing in for the real import.
STAND_INS = {
    "unitree_sdk2py.g1.loco.g1_loco_client.LocoClient": _Loco,
    "unitree_sdk2py.comm.motion_switcher.motion_switcher_client.MotionSwitcherClient": _MotionSwitcher,
}


class _Raises:
    """Every operation raises, standing in for an RPC that never answers."""

    def __getattr__(self, name: str) -> Any:
        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("RPC_CLIENT_API_TIMEOUT (3104)")

        return _boom


@pytest.fixture
def bus(monkeypatch: pytest.MonkeyPatch) -> None:
    """A usable DDS bus and pre-Init()ed clients, with no global state leaked.

    ``_CLIENTS`` is a module-level singleton cache, so it is replaced rather than
    mutated: a client left behind here would be handed to the next test in the
    session as though it had been ``Init()``ed against a real bus.

    The class import is stood in for too, which makes the file hermetic: the
    error envelopes below carry diagnostics built by ``describe_operation`` and
    ``list_operations``, and those read the client class. Left real, this file
    would answer differently on a machine with ``unitree_sdk2py`` installed than
    on one without, and would leave SDK submodules in ``sys.modules`` that a
    sibling file pins the absence of.
    """
    monkeypatch.setattr(uu, "ensure_dds", lambda _iface: None)
    monkeypatch.setattr(uu, "_CLIENTS", {"loco": _Loco(), "motion_switcher": _MotionSwitcher()})
    monkeypatch.setattr(uu, "_import_client_class", lambda qualname: STAND_INS[qualname])


def _failure_rows() -> list[tuple[str, dict[str, Any]]]:
    """The distinct ways a non-meta call can come back ``status: error``.

    Each row is a name and the patches that provoke it. Together they cover every
    ``ok: False`` return in :func:`~strands_robots.tools.g1.use_unitree._execute`
    plus the raise its ``_CALL_LOCK`` block converts, which is what makes "every
    outcome" a measured claim rather than a sampled one.
    """
    return [
        ("dds_unavailable", {"ensure_dds": lambda _i: "cyclonedds is not installed"}),
        ("client_init_failed", {"_CLIENTS": {}, "_import_client_class": _raise_import}),
        ("rpc_raised", {"_CLIENTS": {"loco": _Raises(), "motion_switcher": _Raises()}}),
    ]


def _raise_import(_qualname: str) -> Any:
    raise ImportError("No module named 'unitree_sdk2py'")


@pytest.mark.parametrize("outcome", [name for name, _p in _failure_rows()])
@pytest.mark.parametrize("service_name,operation_name", sorted(uu.HIGH_DANGER_OPS))
def test_a_high_danger_call_that_fails_is_still_flagged_high_danger(
    monkeypatch: pytest.MonkeyPatch,
    bus: None,
    outcome: str,
    service_name: str,
    operation_name: str,
) -> None:
    """Every pair in ``HIGH_DANGER_OPS``, on every failure shape, stays flagged.

    The population is read off the table rather than listed, so an eighth
    dangerous operation inherits the pin by being added there.
    """
    for attr, value in dict(_failure_rows())[outcome].items():
        monkeypatch.setattr(uu, attr, value)

    res = uu.use_unitree(service_name, operation_name, {})

    assert res["status"] == "error", res
    assert res["high_danger"] is True, f"{outcome}: {res}"
    assert res["mutative"] is True, f"{outcome}: {res}"


@pytest.mark.parametrize("outcome", [name for name, _p in _failure_rows()])
def test_a_read_that_fails_is_flagged_harmless_rather_than_left_unflagged(
    monkeypatch: pytest.MonkeyPatch, bus: None, outcome: str
) -> None:
    """A failed read answers ``False``, not a missing key.

    This is the cell that keeps the fix honest: flagging every envelope ``True``
    would satisfy the test above and destroy the signal.
    """
    for attr, value in dict(_failure_rows())[outcome].items():
        monkeypatch.setattr(uu, attr, value)

    res = uu.use_unitree(*HARMLESS_READ, {})

    assert res["status"] == "error", res
    assert res["high_danger"] is False, f"{outcome}: {res}"
    assert res["mutative"] is False, f"{outcome}: {res}"


def test_a_parameter_mismatch_is_flagged_and_still_reports_the_signature(
    bus: None,
) -> None:
    """The bind failure keeps both the flags and ``_execute``'s own diagnostic.

    ``expected`` is the help a caller needs to retry, and the flags are what say
    whether retrying is dangerous; the envelope owes both at once.
    """
    res = uu.use_unitree("loco", "SetVelocity", {"speed": 0.5})

    assert res["status"] == "error"
    assert "parameter mismatch" in res["message"]
    assert [param["name"] for param in res["expected"]["parameters"]] == ["vx", "vy", "vyaw", "duration"]
    assert res["high_danger"] is True
    assert res["mutative"] is True


def test_an_unknown_operation_is_flagged_not_dangerous_and_lists_the_real_ones(
    bus: None,
) -> None:
    """A name that is not on the client cannot be in the table, so it is ``False``.

    ``available_operations`` survives alongside the flags for the same reason
    ``expected`` does above.
    """
    res = uu.use_unitree("loco", "GetNoSuchThing", {})

    assert res["status"] == "error"
    assert res["high_danger"] is False
    assert res["mutative"] is False
    assert "GetFsmId" in res["available_operations"]


def test_a_diagnostic_key_from_execute_cannot_shadow_the_safety_flags(
    monkeypatch: pytest.MonkeyPatch, bus: None
) -> None:
    """The classification wins over anything the inner result carries.

    ``_execute``'s error dicts are spread into the envelope, so the flags have to
    be applied after that spread. Pinned because the ordering is the whole
    guarantee and is invisible in a passing happy-path test.
    """
    monkeypatch.setattr(
        uu,
        "_execute",
        lambda *_a, **_kw: {"ok": False, "error": "boom", "high_danger": False, "mutative": False},
    )

    res = uu.use_unitree("loco", "ZeroTorque", {})

    assert res["high_danger"] is True, res
    assert res["mutative"] is True, res


def test_a_successful_call_still_carries_the_flags_and_the_parameters(bus: None) -> None:
    """Control: the outcome that already worked is unchanged."""
    res = uu.use_unitree("loco", "SetVelocity", {"vx": 0.1, "vy": 0.0, "vyaw": 0.0})

    assert res["status"] == "success"
    assert res["result"] == 0
    assert res["high_danger"] is True
    assert res["mutative"] is True
    assert res["parameters"] == {"vx": 0.1, "vy": 0.0, "vyaw": 0.0}


def test_the_danger_warning_is_logged_even_when_the_call_then_fails(
    monkeypatch: pytest.MonkeyPatch, bus: None, caplog: pytest.LogCaptureFixture
) -> None:
    """The loud half of the flagging does not depend on the RPC succeeding.

    The log record is written before the call is attempted, which is the only
    ordering that records an attempt whose outcome is unknown.
    """
    monkeypatch.setattr(uu, "ensure_dds", lambda _i: "cyclonedds is not installed")

    with caplog.at_level("WARNING", logger="strands_robots.tools.g1.use_unitree"):
        res = uu.use_unitree("loco", "ZeroTorque", {})

    assert res["status"] == "error"
    assert [r.getMessage() for r in caplog.records] == ["use_unitree: loco.ZeroTorque is HIGH_DANGER"]
