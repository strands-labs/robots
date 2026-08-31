"""``g1_move_velocity`` returns exactly what ``G1Driver.move_velocity`` gives it.

``g1_move_velocity`` is the write-side companion to the neon bundle's
``g1_move_velocity`` verb: where the neon verb wraps
``LocoClient.SetVelocity(vx, vy, vyaw, duration)`` under a
single-writer lock, this one hands the velocity quadruple to the
driver's own write path and reads back the envelope the driver
produced. The driver's ``move_velocity`` method is not yet plumbed
today (refs strands-labs/robots#358 for the SDK-facing gate work
the write belongs on), so the
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal` grader
refuses a handle without a ``move_velocity`` accessor with a
message naming the verb, the ``driver`` parameter and the accessor;
these tests fix the shape the verb passes through for each of the
driver's current and future outcomes (a driver-side refusal, a
future success envelope, and the verb's own ``driver``/data
refusals).

The refusal-string tests do not restate the driver's exact prose -
that would trap the verb to the driver's refusal wording of one
release, which is exactly what the driver's own release notes say
verbatim quotes should not do (refs strands-labs/robots#2874).
They grade the shape (``status="error"``, an envelope-shaped
``content[0]["text"]``, the verb's own four refusal invariants when
the verb produced them) and pass the driver's own text through
unchanged when the driver produced it. The SDK-load-hygiene
contract every file under :mod:`strands_robots.tools.g1` carries is
fixed first: importing the module must not pull any
``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

from strands_robots.tools.g1.g1_move_velocity import g1_move_velocity


class _StubG1Driver:
    """A driver double whose ``move_velocity`` returns a fixed envelope.

    ``g1_move_velocity`` calls ``driver.move_velocity(vx, vy, vyaw,
    duration)`` and returns the envelope verbatim. This double sits
    under the same interface without pulling the real driver's
    imports (the real class reaches CycloneDDS at construction time
    in some paths), so a test can hand a wired-shape envelope to
    the verb without a bus. ``call_count`` and ``last_args`` record
    the number of invocations and the last argument tuple so a
    test can pin the verb's dispatch contract without asking the
    driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.call_count: int = 0
        self.last_args: tuple[float, float, float, float] | None = None

    def move_velocity(self, vx: float, vy: float, vyaw: float, duration: float) -> dict[str, Any]:
        self.call_count += 1
        self.last_args = (vx, vy, vyaw, duration)
        return self._envelope


def _call(
    driver: Any,
    vx: float = 0.1,
    vy: float = 0.0,
    vyaw: float = 0.0,
    duration: float = 1.0,
) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim. This helper is where a shape
    drift would surface once, rather than at every call site.  The
    defaults (0.1 m/s forward, 1 s) are the smallest safe-ish
    velocity a caller could send; they never actually reach a bus in
    these tests because the double records them without dispatch.
    """
    return g1_move_velocity(driver=driver, vx=vx, vy=vy, vyaw=vyaw, duration=duration)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a submodule
    at import time would break every headless CI runner and Thor
    before an office bring-up. The driver enforces the same rule
    against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the
    only path that loads the SDK); this cell holds the move-velocity
    verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_move_velocity")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_move_velocity imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the "
        "SDK loads only inside function bodies "
        "(refs strands-labs/robots#358)."
    )


def test_a_driver_side_refusal_surfaces_verbatim() -> None:
    """The driver's refusal envelope round-trips through the verb unchanged.

    ``G1Driver.move_velocity`` will (once landed) refuse an SDK-side
    raise with a named error envelope, and today's driver refuses
    every call because the method is not yet plumbed. Both shapes
    are ``{"status": "error", "content": [{"text": ...}]}`` and the
    verb passes either through; a wording drift on the driver side
    moves this verb with it (refs strands-labs/robots#2874).
    """
    refusal_text = "move_velocity: SetVelocity raised: FSM=1004 not in {501, 801}"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_future_success_envelope_round_trips_verbatim() -> None:
    """Once ``move_velocity`` lands, the success envelope surfaces verbatim.

    When the driver's method is plumbed, ``G1Driver.move_velocity``
    will surface the SDK's ``rc`` inside a ``{"status": "success",
    "content": [{"json": {"rc": 0, "message": ...}}]}`` envelope.
    The verb does not reshape it - a future field the driver adds
    reaches a caller the moment the driver writes it, and this cell
    holds that pass-through explicit so the verb is ready the
    moment the write path lands.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "rc": 0,
                    "message": "SetVelocity(vx=0.1, vy=0.0, vyaw=0.0, dur=1.0) rc=0 (OK)",
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
    result = g1_move_velocity(driver=None, vx=0.1, vy=0.0, vyaw=0.0, duration=1.0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_move_velocity" in text
    assert "driver" in text


def test_a_wrong_shape_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot *name* (string) is refused before the accessor call.

    A model that synthesizes the argument as a robot name reaches
    the verb with a ``str``. The verb owes an envelope-shaped
    refusal that names the type it received and the remedy - the
    four invariants every ``@tool`` handler in this package holds -
    and this cell fixes the shape.
    """
    result = g1_move_velocity(driver="unitree_g1", vx=0.1, vy=0.0, vyaw=0.0, duration=1.0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_move_velocity" in text
    assert "'str'" in text


def test_a_numeric_shape_driver_is_refused_before_the_accessor_path() -> None:
    """A numeric handle is refused with an envelope, not an exception.

    The shared
    :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard is called first; a numeric handle is refused with an
    envelope and the driver's method is never reached. This cell
    holds the ordering by grading a handle that would raise if
    called (``int`` has no ``move_velocity`` attribute) and
    observing that the refusal envelope has the four invariants
    without an exception in flight.
    """
    result = g1_move_velocity(driver=42, vx=0.1, vy=0.0, vyaw=0.0, duration=1.0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_move_velocity" in text
    assert "'int'" in text


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.move_velocity``.

    A double-call from a wrapper that retried inside the verb would
    issue two ``SetVelocity`` writes on the same ``rt/lococmd``
    admission window; the SDK's handler is not re-entrant and
    neon's own bundle held a single-writer lock (refs
    strands-labs/robots#2916 for the driver's arm-SDK gate).  This
    cell pins the verb to a single driver call per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    _call(driver)
    assert driver.call_count == 1


def test_the_verb_passes_the_argument_quadruple_through_verbatim() -> None:
    """The verb hands the driver the exact ``(vx, vy, vyaw, duration)`` it received.

    A reshape here (a rounding, a clamp, a sign flip) would fork
    the SDK's own contract into two sources of truth; a caller
    reading the walkable magnitude bound off the driver's own
    write path would decide the refusal against a different
    envelope than the write actually reached.
    This cell pins the argument round-trip explicit: whatever
    finite-numeric quadruple the caller passed reaches the driver
    unchanged.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    _call(driver, vx=-0.3, vy=0.15, vyaw=-0.25, duration=2.5)
    assert driver.last_args == (-0.3, 0.15, -0.25, 2.5)


def test_a_none_vx_is_refused_with_a_message_naming_the_parameter() -> None:
    """A ``None`` velocity component is refused by the shared validator.

    ``vx``/``vy``/``vyaw`` are data parameters the tool schema
    describes as floats, but a model that leaves one out reaches
    the verb with ``None``. The shared
    :func:`~strands_robots.utils.finite_number_error` validator
    refuses it with a message naming the parameter and the verb;
    this cell fixes the shape and pins that the driver's method
    is not reached (a ``None`` velocity would raise a ``TypeError``
    at the SDK boundary).
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_move_velocity(
        driver=driver,
        vx=None,
        vy=0.0,
        vyaw=0.0,
        duration=1.0,  # type: ignore[arg-type]
    )
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_move_velocity" in text
    assert "vx" in text
    assert driver.call_count == 0


def test_a_nan_velocity_component_is_refused_before_dispatch() -> None:
    """``nan`` poisons every comparison it reaches; the validator refuses it.

    A caller that computes a velocity from a divide-by-zero can
    reach the verb with ``nan``; the SDK's own handler would treat
    the value as a garbage float and the controller's behaviour
    is undefined. The shared
    :func:`~strands_robots.utils.finite_number_error` validator
    refuses it before dispatch so the driver's rc-decoded refusal
    is reserved for the SDK's own return codes.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = _call(driver, vy=math.nan)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_move_velocity" in text
    assert "vy" in text
    assert driver.call_count == 0


def test_an_infinite_yaw_rate_is_refused_before_dispatch() -> None:
    """``inf`` collapses the arithmetic downstream; the validator refuses it.

    A caller that reaches the verb with ``inf`` on ``vyaw`` would
    ask the SDK's ``SetVelocity`` for an unbounded rotation rate;
    the SDK does not clamp the input and the controller's behaviour
    above the envelope is undefined. The shared
    :func:`~strands_robots.utils.finite_number_error` validator
    refuses it before dispatch.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = _call(driver, vyaw=math.inf)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_move_velocity" in text
    assert "vyaw" in text
    assert driver.call_count == 0


def test_a_bool_velocity_is_refused_before_silent_coercion() -> None:
    """A ``bool`` velocity is refused before it coerces to ``0.0``/``1.0``.

    Python's ``bool`` is a subclass of ``int`` and coerces to
    ``0.0`` / ``1.0`` in every arithmetic context. A caller that
    reached the verb with ``True`` on ``vx`` would silently
    command a 1 m/s forward walk (well outside the envelope the
    neon bundle observed as walkable); the shared
    :func:`~strands_robots.utils.finite_number_error` validator
    refuses the ``bool`` subclass before that silent coercion
    reaches the driver.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = _call(driver, vx=True)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_move_velocity" in text
    assert "vx" in text
    assert driver.call_count == 0


def test_a_zero_duration_is_refused_before_dispatch() -> None:
    """A zero-second duration is refused by the positive-finite validator.

    A caller who reached the verb with ``duration=0.0`` would ask
    the SDK for a zero-length walk - the SDK's own handler admits
    it as a no-op, but a caller who reached the verb with that
    value asked for something the verb cannot honour (the
    intended semantics of ``g1_move_velocity`` is "walk for a
    positive length of time").  The shared
    :func:`~strands_robots.utils.positive_finite_number_error`
    validator refuses zero and every negative value; this cell
    pins the shape.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = _call(driver, duration=0.0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_move_velocity" in text
    assert "duration" in text
    assert driver.call_count == 0


def test_a_negative_duration_is_refused_before_dispatch() -> None:
    """A negative-second duration is refused by the positive-finite validator.

    A negative duration is the intent-inversion caller-side; the
    SDK's ``SetVelocity`` handler treats it as ``UINT32_MAX``-shaped
    garbage the controller's behaviour above the envelope is
    undefined on. The shared
    :func:`~strands_robots.utils.positive_finite_number_error`
    validator refuses it; this cell pins the shape.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = _call(driver, duration=-1.5)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_move_velocity" in text
    assert "duration" in text
    assert driver.call_count == 0


def test_a_rpc_timeout_refusal_from_the_driver_surfaces_verbatim() -> None:
    """The SDK's ``rc=3104`` RPC-timeout refusal round-trips through the verb.

    The SDK's ``SetVelocity`` handler surfaces ``rc=3104``
    ("RPC_CLIENT_API_TIMEOUT"; refs
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`) when
    the ``rt/lococmd`` request-response wedges on the loco side.
    The driver's own path surfaces the SDK's answer through its
    refusal envelope; this cell pins that the verb passes the
    envelope through unchanged - a caller reading the driver's
    ``rc`` sees the same number the SDK wrote.
    """
    envelope = {
        "status": "error",
        "content": [
            {
                "json": {
                    "rc": 3104,
                    "message": "SetVelocity rc=3104 (RPC_CLIENT_API_TIMEOUT)",
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver)

    assert result["status"] == "error"
    assert result["content"][0]["json"]["rc"] == 3104
