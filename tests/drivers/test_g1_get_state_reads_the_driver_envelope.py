"""``g1_get_state`` returns exactly what ``G1Driver.get_status`` gives it, plus decided gates.

``g1_get_state`` is the first driver-instance-taking verb in
:mod:`strands_robots.tools.g1`. Every earlier verb there is a pure reader
over module-level constants; this one is a live read through
:meth:`~strands_robots.drivers.g1.G1Driver.get_status`, and every field it
returns has to come from that call. The tests here fix that contract by
handing a hand-rolled driver double to the verb and asserting the returned
dict names each field the driver reported, plus the two ``admits_arm`` /
``admits_loco`` booleans this verb decides against the driver's own gate
constants (:data:`HANDSHAKE_FSMS` / :data:`WALK_FSMS`).

The membership rules are read here off the driver's constants rather than
being restated in the tests, so a widen or narrow of an admission set in
the driver moves both the write path and this verb together. What the
tests do restate is the SDK-load-hygiene contract every file under
:mod:`strands_robots.tools.g1` carries: importing the module must not
pull any ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import HANDSHAKE_FSMS, WALK_FSMS
from strands_robots.tools.g1.g1_state import g1_get_state


class _StubG1Driver:
    """A driver double whose ``get_status`` returns a fixed envelope.

    ``g1_get_state`` calls ``await driver.get_status()`` and reads the
    inner ``content[0]['json']`` dict; every field on that dict shows up
    on the verb's return value verbatim. This double sits under the same
    interface without pulling the real driver's imports (the real class
    reaches CycloneDDS at construction time in some paths), so a test
    can hand a wired-shape envelope to the verb without a bus.
    """

    def __init__(self, inner: dict[str, Any], *, status: str = "success") -> None:
        self._inner = inner
        self._status = status

    async def get_status(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "content": [{"json": self._inner}],
        }


def _call(driver: _StubG1Driver) -> dict[str, Any]:
    """Run the async verb on a fresh event loop and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped coroutine
    directly when called in-process, but a caller cannot rely on that:
    the wrapper's contract is that it awaits the wrapped function. This
    helper is where a shape drift would surface once, rather than at
    every call site.
    """
    return asyncio.run(g1_get_state(driver=driver))


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import time
    would break every headless CI runner and Thor before an office
    bring-up. The driver enforces the same rule against itself; this cell
    holds the state verb to it too - which is why the driver type hint is
    a forward reference under ``TYPE_CHECKING``, not a runtime import.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_state")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_state imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_a_disconnected_driver_reports_every_none_field() -> None:
    """A driver that never connected returns ``None`` on every wire field.

    The mesh publishes ``get_status`` on presence, so a driver that just
    got built has no ``fsm_id``, no ``mode_machine``, no ``battery_pct``.
    The verb must not raise on that shape and must not fabricate a value
    where the driver reported ``None``.
    """
    inner = {
        "tool_name": "g1",
        "connected": False,
        "connect_error": None,
        "port": "1.2.3.4",
        "network_interface": "eth0",
        "fsm_id": None,
        "mode_machine": None,
        "battery_pct": None,
        "fsm_mode_name": None,
        "fsm_refusal": None,
        "motion_switcher_open_error": None,
    }
    result = _call(_StubG1Driver(inner))

    assert result["status"] == "success"
    assert result["tool_name"] == "g1"
    assert result["connected"] is False
    assert result["port"] == "1.2.3.4"
    assert result["network_interface"] == "eth0"
    assert result["fsm_id"] is None
    assert result["mode_machine"] is None
    assert result["battery_pct"] is None
    # A None fsm_id cannot open either gate: the read never arrived.
    assert result["admits_arm"] is False
    assert result["admits_loco"] is False


def test_the_admission_sets_are_read_off_the_driver_constants() -> None:
    """The returned ``handshake_fsms`` / ``walk_fsms`` name the driver's own sets.

    A caller quoting the gate's admitted ids in its own voice reads them
    off this verb's return value rather than the ``_g1_common`` module. A
    drift between the two would let the caller quote one set while the
    driver's write path enforces another; this test refuses that drift.
    """
    inner: dict[str, Any] = {
        "tool_name": "g1",
        "connected": True,
        "connect_error": None,
        "port": "1.2.3.4",
        "network_interface": "eth0",
        "fsm_id": 501,
        "mode_machine": 4,
        "battery_pct": 85.0,
        "fsm_mode_name": "ai",
        "fsm_refusal": None,
        "motion_switcher_open_error": None,
    }
    result = _call(_StubG1Driver(inner))
    assert result["handshake_fsms"] == sorted(HANDSHAKE_FSMS)
    assert result["walk_fsms"] == sorted(WALK_FSMS)


def test_fsm_500_admits_arm_but_not_loco() -> None:
    """FSM 500 (sitting) admits arm gestures but refuses walking.

    :data:`HANDSHAKE_FSMS` carries 500 because the arm-SDK gate opens on
    sitting; :data:`WALK_FSMS` does not, because the robot cannot walk
    from a sit. The verb's two admits booleans must reflect that split.
    """
    inner: dict[str, Any] = {
        "tool_name": "g1",
        "connected": True,
        "connect_error": None,
        "port": "1.2.3.4",
        "network_interface": "eth0",
        "fsm_id": 500,
        "mode_machine": 5,
        "battery_pct": 92.0,
        "fsm_mode_name": "ai",
        "fsm_refusal": None,
        "motion_switcher_open_error": None,
    }
    result = _call(_StubG1Driver(inner))
    assert result["fsm_id"] == 500
    assert result["admits_arm"] is True
    assert result["admits_loco"] is False


def test_fsm_501_and_801_admit_both_scopes() -> None:
    """The two shared ids in :data:`HANDSHAKE_FSMS` and :data:`WALK_FSMS` admit both.

    :data:`WALK_FSMS` is the strict subset ``{501, 801}``; every id in it
    is also in :data:`HANDSHAKE_FSMS`, so a live-walking robot admits arm
    gestures too. The verb's booleans must both be ``True`` for either id.
    """
    for fsm_id in (501, 801):
        inner: dict[str, Any] = {
            "tool_name": "g1",
            "connected": True,
            "connect_error": None,
            "port": "1.2.3.4",
            "network_interface": "eth0",
            "fsm_id": fsm_id,
            "mode_machine": 4,
            "battery_pct": 85.0,
            "fsm_mode_name": "ai",
            "fsm_refusal": None,
            "motion_switcher_open_error": None,
        }
        result = _call(_StubG1Driver(inner))
        assert result["admits_arm"] is True, f"fsm_id={fsm_id} should admit arm"
        assert result["admits_loco"] is True, f"fsm_id={fsm_id} should admit loco"


def test_an_unrelated_fsm_id_admits_neither_scope() -> None:
    """An FSM outside both sets closes both gates.

    An FSM id like ``4`` (main-op-control) is neither in
    :data:`HANDSHAKE_FSMS` nor :data:`WALK_FSMS`; a caller reading this
    verb before a write must see both booleans False so it can phrase its
    own refusal instead of triggering the driver's at wire time.
    """
    inner: dict[str, Any] = {
        "tool_name": "g1",
        "connected": True,
        "connect_error": None,
        "port": "1.2.3.4",
        "network_interface": "eth0",
        "fsm_id": 4,
        "mode_machine": 1,
        "battery_pct": 72.0,
        "fsm_mode_name": "ai",
        "fsm_refusal": None,
        "motion_switcher_open_error": None,
    }
    result = _call(_StubG1Driver(inner))
    assert result["fsm_id"] == 4
    assert result["admits_arm"] is False
    assert result["admits_loco"] is False


def test_a_bool_fsm_id_is_not_treated_as_admitting() -> None:
    """``True`` is ``int(1)`` but cannot open a motion gate.

    A driver whose ``fsm_id`` field was mis-populated with a ``bool`` (a
    caller typo, or a bad decode path) would silently pass an ``int``
    membership check because ``True == 1``. The verb refuses that: a
    ``bool`` cannot be an FSM id, so neither gate opens even if the
    numeric value would.
    """
    for value in (True, False):
        inner: dict[str, Any] = {
            "tool_name": "g1",
            "connected": True,
            "connect_error": None,
            "port": "1.2.3.4",
            "network_interface": "eth0",
            "fsm_id": value,
            "mode_machine": 4,
            "battery_pct": 85.0,
            "fsm_mode_name": "ai",
            "fsm_refusal": None,
            "motion_switcher_open_error": None,
        }
        result = _call(_StubG1Driver(inner))
        assert result["admits_arm"] is False, f"fsm_id={value!r} must not admit arm"
        assert result["admits_loco"] is False, f"fsm_id={value!r} must not admit loco"


def test_motion_switcher_diagnostics_round_trip_verbatim() -> None:
    """The three motion-switcher diagnostic fields ride through unchanged.

    ``fsm_mode_name`` / ``fsm_refusal`` / ``motion_switcher_open_error``
    are how the driver names why its last read declined; a caller reading
    this verb sees the same strings the driver would log so its own
    refusal can cite them. The verb is not the place to reword those.
    """
    inner: dict[str, Any] = {
        "tool_name": "g1",
        "connected": True,
        "connect_error": None,
        "port": "1.2.3.4",
        "network_interface": "eth0",
        "fsm_id": None,
        "mode_machine": None,
        "battery_pct": None,
        "fsm_mode_name": "",
        "fsm_refusal": "CheckMode timed out (rc=3104)",
        "motion_switcher_open_error": None,
    }
    result = _call(_StubG1Driver(inner))
    assert result["fsm_mode_name"] == ""
    assert result["fsm_refusal"] == "CheckMode timed out (rc=3104)"
    assert result["motion_switcher_open_error"] is None


def test_status_field_propagates_from_the_envelope() -> None:
    """The driver's envelope ``status`` reaches the verb's return value.

    ``get_status`` returns ``{"status": ..., "content": [...]}``; the
    verb's dict adopts that status verbatim. A driver returning
    ``"error"`` here (as it may in a future path that closes the port
    without also clearing ``_last_status``) must surface as an error on
    the verb too, not a silent success.
    """
    inner: dict[str, Any] = {
        "tool_name": "g1",
        "connected": False,
        "connect_error": "ENOENT",
        "port": None,
        "network_interface": "eth0",
        "fsm_id": None,
        "mode_machine": None,
        "battery_pct": None,
        "fsm_mode_name": None,
        "fsm_refusal": None,
        "motion_switcher_open_error": None,
    }
    result = _call(_StubG1Driver(inner, status="error"))
    assert result["status"] == "error"
    assert result["connect_error"] == "ENOENT"


def test_the_verb_awaits_the_driver_call_exactly_once() -> None:
    """The verb calls ``get_status`` exactly once per invocation.

    Reading state twice on a single verb call would double the mesh
    presence chip's cost and (on a driver with side effects on read)
    double the wire touch. This test counts the calls on a hand-rolled
    double so the single-await contract is graded rather than assumed.
    """

    class _CountingDriver:
        def __init__(self) -> None:
            self.calls = 0

        async def get_status(self) -> dict[str, Any]:
            self.calls += 1
            return {
                "status": "success",
                "content": [
                    {
                        "json": {
                            "tool_name": "g1",
                            "connected": True,
                            "connect_error": None,
                            "port": "1.2.3.4",
                            "network_interface": "eth0",
                            "fsm_id": 501,
                            "mode_machine": 4,
                            "battery_pct": 90.0,
                            "fsm_mode_name": "ai",
                            "fsm_refusal": None,
                            "motion_switcher_open_error": None,
                        }
                    }
                ],
            }

    driver = _CountingDriver()
    _call(driver)  # type: ignore[arg-type]
    assert driver.calls == 1
