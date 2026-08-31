"""``g1_set_swing_height`` returns exactly what ``G1Driver.set_swing_height`` gives it.

``g1_set_swing_height`` is the write-side companion to the neon
bundle's ``g1_set_swing_height`` verb: where the neon verb wraps a
raw ``_Call`` on API id ``7103`` (the SDK's Python ``LocoClient``
does not expose ``SetSwingHeight``) under a single-writer lock,
this one hands a target height to the driver's own write path and
reads back the envelope the driver produced. The driver's
``G1Driver.set_swing_height``
method is not yet plumbed today (refs strands-labs/robots#358 for
the SDK-facing gate work the write belongs on), so the
:func:`live_handle_refusal` grader refuses a handle without a
``set_swing_height`` accessor with a message naming the verb, the
``driver`` parameter and the accessor; these tests fix the shape
the verb passes through for each of the driver's current and
future outcomes (a driver-side refusal, a future success envelope,
and the verb's own ``driver`` / ``height`` refusals).

The refusal-string tests do not restate the driver's exact prose -
that would trap the verb to the driver's refusal wording of one
release, which is exactly what the driver's own release notes say
verbatim quotes should not do (refs strands-labs/robots#2874).
They grade the shape (``status="error"``, an envelope-shaped
``content[0]["text"]``, the verb's own four refusal invariants
when the verb produced them) and pass the driver's own text
through unchanged when the driver produced it. The SDK-load-
hygiene contract every file under :mod:`strands_robots.tools.g1`
carries is fixed first: importing the module must not pull any
``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_set_swing_height import g1_set_swing_height


class _StubG1Driver:
    """A driver double whose ``set_swing_height`` returns a fixed envelope.

    ``g1_set_swing_height`` calls ``driver.set_swing_height(height)``
    and returns the envelope verbatim. This double sits under the
    same interface without pulling the real driver's imports (the
    real class reaches CycloneDDS at construction time in some
    paths), so a test can hand a wired-shape envelope to the verb
    without a bus. ``calls`` records the ``height`` per invocation
    so a test can pin "the verb writes the driver exactly once" and
    "the verb passes the argument through unchanged" without
    asking the driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls: list[float] = []

    def set_swing_height(self, height: float) -> dict[str, Any]:
        self.calls.append(height)
        return self._envelope


def _call(
    driver: Any,
    *,
    height: float | None = 0.1,
) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim. This helper is where a shape
    drift would surface once, rather than at every call site. The
    default ``height=0.1`` is the middle of the neon-bundle-
    observed typical safe range (``0.05..0.15 m``), so a call that
    omits the value here still reaches the driver with a target
    the firmware admits.
    """
    return g1_set_swing_height(driver=driver, height=height)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a
    submodule at import time would break every headless CI runner
    and Thor before an office bring-up. The driver enforces the
    same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the
    only path that loads the SDK); this cell holds the set-swing-
    height verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_set_swing_height")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_set_swing_height imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        "loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_side_refusal_surfaces_verbatim() -> None:
    """The driver's refusal envelope round-trips through the verb unchanged.

    ``G1Driver.set_swing_height`` will (once landed) refuse an
    SDK-side raise with a named error envelope, and today's driver
    refuses every call because the method is not yet plumbed. Both
    shapes are ``{"status": "error", "content": [{"text": ...}]}``
    and the verb passes either through; a wording drift on the
    driver side moves this verb with it (refs
    strands-labs/robots#2874).
    """
    refusal_text = "set_swing_height: _Call(7103, ...) raised: RPC_CLIENT_API_TIMEOUT"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, height=0.12)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_future_success_envelope_round_trips_verbatim() -> None:
    """Once ``set_swing_height`` lands, the success envelope surfaces verbatim.

    When the driver's method is plumbed,
    ``G1Driver.set_swing_height`` will surface the SDK's ``rc``
    inside a ``{"status": "success", "content": [{"json": {"rc":
    0, "message": ...}}]}`` envelope. The verb does not reshape it
    - a future field the driver adds reaches a caller the moment
    the driver writes it, and this cell holds that pass-through
    explicit so the verb is ready the moment the write path lands.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "rc": 0,
                    "message": "SetSwingHeight(0.100m) rc=0 (OK)",
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, height=0.1)

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
    result = g1_set_swing_height(driver=None, height=0.1)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_swing_height" in text
    assert "driver" in text


def test_a_wrong_shape_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot *name* (string) is refused before the accessor call.

    A model that synthesizes the argument as a robot name reaches
    the verb with a ``str``. The verb owes an envelope-shaped
    refusal that names the type it received and the remedy - the
    four invariants every ``@tool`` handler in this package holds
    - and this cell fixes the shape.
    """
    result = g1_set_swing_height(driver="unitree_g1", height=0.1)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_swing_height" in text
    assert "'str'" in text


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.set_swing_height``.

    A double-call from a wrapper that retried inside the verb
    would issue two API-7103 writes on the same admission window;
    the SDK's ``_Call`` handler is not re-entrant and neon's own
    bundle held a single-writer lock. This cell pins the verb to
    a single driver call per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    _call(driver)
    assert len(driver.calls) == 1


def test_the_verb_passes_the_argument_through_unchanged() -> None:
    """The height the driver receives is the one the caller passed.

    The verb's contract is a pass-through: it does not translate
    argument names, does not synthesize defaults the driver did
    not ask for, and does not clamp the value. This cell fixes
    that ``height`` reaches the driver method verbatim - a rename
    or coercion on either side is a driver-level contract change,
    not a silent verb-side translation.
    """
    envelope = {"status": "success", "content": [{"json": {}}]}
    driver = _StubG1Driver(envelope=envelope)
    _call(driver, height=0.135)
    assert driver.calls == [0.135]


def test_the_wrong_shape_driver_is_not_called() -> None:
    """A wrong-shape driver is refused before the accessor path.

    The shared
    :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard is called first; a numeric handle is refused with an
    envelope and the driver's method is never reached. This cell
    holds the ordering by grading a handle that would raise if
    called (``int`` has no ``set_swing_height`` attribute) and
    observing that the refusal envelope has the four invariants
    without an exception in flight.
    """
    result = g1_set_swing_height(driver=42, height=0.1)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_swing_height" in text
    assert "'int'" in text


def test_a_missing_height_is_refused_with_a_message_naming_the_parameter() -> None:
    """A ``None`` ``height`` is refused before the driver is called.

    ``height`` is a data parameter the tool schema *does*
    describe, so a model can synthesize the wrong shape here as
    easily as it can reach the verb with the right one. ``None``
    is the "the model left it out" shape: the verb owes an
    envelope-shaped refusal naming the parameter and the remedy,
    and this cell fixes that the refusal fires before the driver
    is reached.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_swing_height(driver=driver, height=None)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_swing_height" in text
    assert "height" in text
    assert driver.calls == []


def test_a_wrong_shape_height_is_refused_with_a_message_naming_the_parameter() -> None:
    """A non-numeric ``height`` is refused before the driver is called.

    A model may synthesize the height as a string ``"0.1"``; the
    SDK's raw ``_Call`` handler on API 7103 is float-only and the
    driver's own path would raise on a non-numeric shape. The
    shared :func:`~strands_robots.utils.finite_number_error`
    validator refuses the shape here so the envelope names
    ``height`` and the remedy.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_swing_height(driver=driver, height="0.1")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_swing_height" in text
    assert "height" in text
    assert driver.calls == []


def test_a_boolean_height_is_refused_because_it_would_act_as_a_silent_number() -> None:
    """A ``True`` payload is refused rather than coerced to ``1.0`` meters.

    ``bool`` is an ``int`` subclass, so a caller passing ``True``
    would reach the SDK's raw ``_Call(7103, 1.0)`` - a leg-lift 5x
    the neon-bundle-observed upper bound - through a signature
    that names none of that; ``False`` would collapse to the LOW
    / shuffle gait silently. The shared
    :func:`~strands_robots.utils.finite_number_error` validator
    refuses ``bool`` explicitly rather than silently transitioning
    to a gait clearance the caller did not name; a caller who
    wants a specific height reaches the verb with a float
    explicitly. This cell holds the guard.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_swing_height(driver=driver, height=True)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_swing_height" in text
    assert "height" in text
    assert driver.calls == []


def test_a_nan_height_is_refused_before_the_driver_is_called() -> None:
    """A ``nan`` ``height`` is refused rather than poisoning the SDK call.

    ``nan`` compares false against every finite bound; the SDK's
    raw ``_Call`` handler would silently accept the frame and the
    driver's own arithmetic on the target would produce nonsense
    gait targets. The shared validator refuses the shape here so
    the envelope names the parameter and the remedy is decidable
    rather than firmware-dependent.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_swing_height(driver=driver, height=float("nan"))
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_swing_height" in text
    assert "height" in text
    assert driver.calls == []


def test_an_inf_height_is_refused_before_the_driver_is_called() -> None:
    """An ``inf`` ``height`` is refused rather than reaching the SDK.

    Positive infinity collapses the swing-height arithmetic to an
    unbounded target; the firmware would clamp but the driver's
    own :func:`float` conversion path may not. The shared
    validator refuses the shape here so the envelope names the
    parameter and the remedy is decidable rather than platform-
    dependent.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_swing_height(driver=driver, height=float("inf"))
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_swing_height" in text
    assert "height" in text
    assert driver.calls == []


def test_a_zero_height_is_admitted_because_the_domain_includes_the_shuffle_gait() -> None:
    """A ``height=0.0`` reaches the driver: 0 is the minimum-clearance gait.

    Unlike a positive-only knob, ``height`` at ``0.0`` is a
    caller-facing value the neon bundle's own wrapper did not
    reject: it is the minimum-clearance shuffle gait, and the
    read-only envelope
    ``g1_swing_height_envelope`` (removed; envelope constants live inline here)
    names it as the inclusive lower bound. The finite-number
    validator - not the positive-finite one - admits ``0.0``, so
    this cell pins that a caller who wants the shuffle gait
    reaches the driver with the target verbatim.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, height=0.0)

    assert result["status"] == "success"
    assert driver.calls == [0.0]


def test_a_negative_height_reaches_the_driver_unchanged() -> None:
    """A negative ``height`` is passed to the driver, not clamped in the verb.

    The neon bundle's own wrapper rounded any strictly-negative
    input up to ``0.0`` before dispatch (its
    ``max(0.0, min(0.2, float(height)))`` clamp), but the clamp
    is a driver-side / firmware-side concern, not a verb-side
    one; the module docstring names "does not clamp" as one of
    the things this verb does not do. This cell pins that the
    verb does not refuse or transform a negative value on its
    own - a rewording of the driver's clamp would then reach a
    caller through the envelope, not through a verb-side
    interception.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, height=-0.05)

    assert result["status"] == "success"
    assert driver.calls == [-0.05]


def test_a_height_above_the_neon_envelope_reaches_the_driver_unchanged() -> None:
    """A ``height`` above the neon bundle's ``0.2`` upper bound is not refused.

    The read-only envelope
    ``g1_swing_height_envelope`` (removed; envelope constants live inline here)
    names ``0.2 m`` as the inclusive upper bound the neon
    bundle's own wrapper enforced; above this the controller's
    response is undefined and the SDK places no clamp of its
    own. The module docstring names "does not clamp" as one of
    the things this verb does not do - refusing above the
    envelope here would fork the neon bundle's admission set
    into a second source of truth. This cell pins that a caller
    who passes ``0.5`` reaches the driver's own refusal (or the
    firmware's clamp-and-warn) through the verb's pass-through,
    rather than being intercepted here.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, height=0.5)

    assert result["status"] == "success"
    assert driver.calls == [0.5]
