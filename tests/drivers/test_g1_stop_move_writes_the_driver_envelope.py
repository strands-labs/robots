"""``g1_stop_move`` returns exactly what ``G1Driver.stop_move`` gives it.

``g1_stop_move`` is the write-side companion to the neon bundle's
``g1_stop_move`` verb: where the neon verb wraps
``LocoClient.StopMove()`` under a single-writer lock, this one
hands the halt request to the driver's own write path and reads
back the envelope the driver produced. The driver's ``stop_move``
method is not yet plumbed today (refs strands-labs/robots#358 for
the SDK-facing gate work the write belongs on), so the
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal` grader
refuses a handle without a ``stop_move`` accessor with a message
naming the verb, the ``driver`` parameter and the accessor; these
tests fix the shape the verb passes through for each of the driver's
current and future outcomes (a driver-side refusal, a future success
envelope, and the verb's own ``driver`` refusals).

The refusal-string tests do not restate the driver's exact prose -
that would trap the verb to the driver's refusal wording of one
release, which is exactly what the driver's own release notes say
verbatim quotes should not do (refs strands-labs/robots#2874). They
grade the shape (``status="error"``, an envelope-shaped
``content[0]["text"]``, the verb's own four refusal invariants when
the verb produced them) and pass the driver's own text through
unchanged when the driver produced it. The SDK-load-hygiene
contract every file under :mod:`strands_robots.tools.g1` carries is
fixed first: importing the module must not pull any
``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_stop_move import g1_stop_move


class _StubG1Driver:
    """A driver double whose ``stop_move`` returns a fixed envelope.

    ``g1_stop_move`` calls ``driver.stop_move()`` and returns the
    envelope verbatim. This double sits under the same interface
    without pulling the real driver's imports (the real class
    reaches CycloneDDS at construction time in some paths), so a
    test can hand a wired-shape envelope to the verb without a bus.
    ``call_count`` records the number of invocations so a test can
    pin "the verb writes the driver exactly once" without asking
    the driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.call_count: int = 0

    def stop_move(self) -> dict[str, Any]:
        self.call_count += 1
        return self._envelope


def _call(driver: Any) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim. This helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return g1_stop_move(driver=driver)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a submodule
    at import time would break every headless CI runner and Thor
    before an office bring-up. The driver enforces the same rule
    against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the
    only path that loads the SDK); this cell holds the stop-move
    verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_stop_move")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_stop_move imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        "loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_side_refusal_surfaces_verbatim() -> None:
    """The driver's refusal envelope round-trips through the verb unchanged.

    ``G1Driver.stop_move`` will (once landed) refuse an SDK-side
    raise with a named error envelope, and today's driver refuses
    every call because the method is not yet plumbed. Both shapes
    are ``{"status": "error", "content": [{"text": ...}]}`` and the
    verb passes either through; a wording drift on the driver side
    moves this verb with it (refs strands-labs/robots#2874).
    """
    refusal_text = "stop_move: StopMove raised: RPC_CLIENT_API_TIMEOUT"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_future_success_envelope_round_trips_verbatim() -> None:
    """Once ``stop_move`` lands, the success envelope surfaces verbatim.

    When the driver's method is plumbed, ``G1Driver.stop_move``
    will surface the SDK's ``rc`` inside a ``{"status": "success",
    "content": [{"json": {"rc": 0, "message": ...}}]}`` envelope.
    The verb does not reshape it - a future field the driver adds
    reaches a caller the moment the driver writes it, and this cell
    holds that pass-through explicit so the verb is ready the moment
    the write path lands.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "rc": 0,
                    "message": "StopMove rc=0 (OK)",
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver)

    assert result is envelope or result == envelope
    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["rc"] == 0


def test_a_missing_driver_is_refused_with_a_message_naming_the_parameter() -> None:
    """A ``None`` driver is refused before the accessor is called.

    ``driver`` is a live Python object typed :class:`~typing.Any`,
    so the tool schema carries no signal that a caller cannot
    synthesize it. A model that leaves the parameter out reaches
    the verb with ``None``, and the verb owes an envelope-shaped
    refusal instead of an exception the ``@tool`` wrapper cannot
    format. The shared
    :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard produces it; this cell fixes that the guard is called
    before the accessor path.
    """
    result = g1_stop_move(driver=None)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_stop_move" in text
    assert "driver" in text


def test_a_wrong_shape_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot *name* (string) is refused before the accessor call.

    A model that synthesizes the argument as a robot name reaches
    the verb with a ``str``. The verb owes an envelope-shaped
    refusal that names the type it received and the remedy - the
    four invariants every ``@tool`` handler in this package holds -
    and this cell fixes the shape.
    """
    result = g1_stop_move(driver="unitree_g1")
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_stop_move" in text
    assert "'str'" in text


def test_a_numeric_shape_driver_is_refused_before_the_accessor_path() -> None:
    """A numeric handle is refused with an envelope, not an exception.

    The shared
    :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard is called first; a numeric handle is refused with an
    envelope and the driver's method is never reached. This cell
    holds the ordering by grading a handle that would raise if
    called (``int`` has no ``stop_move`` attribute) and observing
    that the refusal envelope has the four invariants without an
    exception in flight.
    """
    result = g1_stop_move(driver=42)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_stop_move" in text
    assert "'int'" in text


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.stop_move``.

    A double-call from a wrapper that retried inside the verb would
    issue two ``StopMove`` writes on the same ``rt/lococmd``
    admission window; the SDK's handler is not re-entrant and neon's
    own bundle held a single-writer lock (refs
    strands-labs/robots#2916 for the driver's arm-SDK gate). This
    cell pins the verb to a single driver call per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    _call(driver)
    assert driver.call_count == 1


def test_a_rpc_timeout_refusal_from_the_driver_surfaces_verbatim() -> None:
    """The SDK's ``rc=3104`` RPC-timeout refusal round-trips through the verb.

    The SDK's ``StopMove`` handler surfaces ``rc=3104``
    ("RPC_CLIENT_API_TIMEOUT"; refs
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`) when the
    ``rt/lococmd`` request-response wedges on the loco side (the
    same failure mode the neon bundle documented for ``read_fsm_id``
    calls on the same client). The driver's own stop path surfaces
    the SDK's answer through its refusal envelope; this cell pins
    that the verb passes the envelope through unchanged - a caller
    reading the driver's ``rc`` sees the same number the SDK wrote.
    """
    envelope = {
        "status": "error",
        "content": [
            {
                "json": {
                    "rc": 3104,
                    "message": "StopMove rc=3104 (RPC_CLIENT_API_TIMEOUT)",
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver)

    assert result["status"] == "error"
    assert result["content"][0]["json"]["rc"] == 3104
