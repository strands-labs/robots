"""``g1_set_stand_height`` returns exactly what ``G1Driver.set_stand_height`` gives it.

``g1_set_stand_height`` is the write-side companion to the neon
bundle's ``g1_set_stand_height`` verb: where the neon verb wraps
``LocoClient.SetStandHeight`` and adds a negative-value fallback to
``LocoClient.HighStand`` (which uses the SDK's ``UINT32_MAX`` height
sentinel), this one hands a target height to the driver's own
write path and reads back the envelope the driver produced.  The
driver's ``set_stand_height``
method is not yet plumbed today (refs strands-labs/robots#358 for
the SDK-facing gate work the write belongs on), so the
:func:`live_handle_refusal` grader refuses a handle without a
``set_stand_height`` accessor with a message naming the verb, the
``driver`` parameter and the accessor; these tests fix the shape
the verb passes through for each of the driver's current and future
outcomes (a driver-side refusal, a future success envelope, and the
verb's own ``driver`` / ``height`` refusals).

The refusal-string tests do not restate the driver's exact prose -
that would trap the verb to the driver's refusal wording of one
release, which is exactly what the driver's own release notes say
verbatim quotes should not do (refs strands-labs/robots#2874).  They
grade the shape (``status="error"``, an envelope-shaped
``content[0]["text"]``, the verb's own four refusal invariants when
the verb produced them) and pass the driver's own text through
unchanged when the driver produced it.  The SDK-load-hygiene
contract every file under :mod:`strands_robots.tools.g1` carries is
fixed first: importing the module must not pull any
``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_set_stand_height import g1_set_stand_height


class _StubG1Driver:
    """A driver double whose ``set_stand_height`` returns a fixed envelope.

    ``g1_set_stand_height`` calls ``driver.set_stand_height(height)``
    and returns the envelope verbatim.  This double sits under the
    same interface without pulling the real driver's imports (the
    real class reaches CycloneDDS at construction time in some
    paths), so a test can hand a wired-shape envelope to the verb
    without a bus.  ``calls`` records the ``height`` per invocation
    so a test can pin "the verb writes the driver exactly once" and
    "the verb passes the argument through unchanged" without asking
    the driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls: list[float] = []

    def set_stand_height(self, height: float) -> dict[str, Any]:
        self.calls.append(height)
        return self._envelope


def _call(
    driver: Any,
    *,
    height: float | None = 0.5,
) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim.  This helper is where a shape
    drift would surface once, rather than at every call site.  The
    default ``height=0.5`` is a mid-range stance (the neon bundle
    documents ``0.0..~0.8`` as the typical range), so a call that
    omits the value here still reaches the driver with a target the
    firmware admits.
    """
    return g1_set_stand_height(driver=driver, height=height)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a submodule
    at import time would break every headless CI runner and Thor
    before an office bring-up.  The driver enforces the same rule
    against itself (:func:`~strands_robots.tools.g1._g1_common.ensure_dds`
    is the only path that loads the SDK); this cell holds the
    set-stand-height verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_set_stand_height")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_set_stand_height imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        "loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_side_refusal_surfaces_verbatim() -> None:
    """The driver's refusal envelope round-trips through the verb unchanged.

    ``G1Driver.set_stand_height`` will (once landed) refuse an
    SDK-side raise with a named error envelope, and today's driver
    refuses every call because the method is not yet plumbed.  Both
    shapes are ``{"status": "error", "content": [{"text": ...}]}``
    and the verb passes either through; a wording drift on the
    driver side moves this verb with it (refs
    strands-labs/robots#2874).
    """
    refusal_text = "set_stand_height: SetStandHeight raised: RPC_CLIENT_API_TIMEOUT"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, height=0.6)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_future_success_envelope_round_trips_verbatim() -> None:
    """Once ``set_stand_height`` lands, the success envelope surfaces verbatim.

    When the driver's method is plumbed,
    ``G1Driver.set_stand_height`` will surface the SDK's ``rc``
    inside a ``{"status": "success", "content": [{"json": {"rc": 0,
    "message": ...}}]}`` envelope.  The verb does not reshape it - a
    future field the driver adds reaches a caller the moment the
    driver writes it, and this cell holds that pass-through explicit
    so the verb is ready the moment the write path lands.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "rc": 0,
                    "message": "SetStandHeight(0.5m) rc=0 (OK)",
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, height=0.5)

    assert result is envelope or result == envelope
    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["rc"] == 0


def test_a_highstand_fallback_success_envelope_round_trips_verbatim() -> None:
    """A negative-height ``HighStand`` envelope round-trips verbatim.

    The negative-value fallback routes the driver's method to
    ``LocoClient.HighStand()``; the driver's envelope for that
    branch names the label rather than the height (the sentinel
    encoding is a wire-side concern).  The verb passes the negative
    ``height`` through to the driver and returns the envelope
    unchanged.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "rc": 0,
                    "message": "SetStandHeight(HighStand) rc=0 (OK)",
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, height=-1.0)

    assert result["status"] == "success"
    assert driver.calls == [-1.0]


def test_a_missing_driver_is_refused_with_a_message_naming_the_parameter() -> None:
    """A ``None`` driver is refused before the accessor is called.

    ``driver`` is a live Python object typed :class:`~typing.Any`, so
    the tool schema carries no signal that a caller cannot synthesize
    it.  A model that leaves the parameter out reaches the verb with
    ``None``, and the verb owes an envelope-shaped refusal instead of
    an exception the ``@tool`` wrapper cannot format.  The shared
    :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard produces it; this cell fixes that the guard is called
    before the accessor path.
    """
    result = g1_set_stand_height(driver=None, height=0.5)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_stand_height" in text
    assert "driver" in text


def test_a_wrong_shape_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot *name* (string) is refused before the accessor call.

    A model that synthesizes the argument as a robot name reaches
    the verb with a ``str``.  The verb owes an envelope-shaped
    refusal that names the type it received and the remedy - the
    four invariants every ``@tool`` handler in this package holds -
    and this cell fixes the shape.
    """
    result = g1_set_stand_height(driver="unitree_g1", height=0.5)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_stand_height" in text
    assert "'str'" in text


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.set_stand_height``.

    A double-call from a wrapper that retried inside the verb would
    issue two ``SetStandHeight`` writes on the same admission window;
    the SDK's handler is not re-entrant and neon's own bundle held a
    single-writer lock.  This cell pins the verb to a single driver
    call per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    _call(driver)
    assert len(driver.calls) == 1


def test_the_verb_passes_the_argument_through_unchanged() -> None:
    """The height the driver receives is the one the caller passed.

    The verb's contract is a pass-through: it does not translate
    argument names, does not synthesize defaults the driver did not
    ask for, and does not clamp the value.  This cell fixes that
    ``height`` reaches the driver method verbatim - a rename or
    coercion on either side is a driver-level contract change, not
    a silent verb-side translation.
    """
    envelope = {"status": "success", "content": [{"json": {}}]}
    driver = _StubG1Driver(envelope=envelope)
    _call(driver, height=0.75)
    assert driver.calls == [0.75]


def test_the_wrong_shape_driver_is_not_called() -> None:
    """A wrong-shape driver is refused before the accessor path.

    The shared :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard is called first; a numeric handle is refused with an
    envelope and the driver's method is never reached.  This cell
    holds the ordering by grading a handle that would raise if
    called (``int`` has no ``set_stand_height`` attribute) and
    observing that the refusal envelope has the four invariants
    without an exception in flight.
    """
    result = g1_set_stand_height(driver=42, height=0.5)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_stand_height" in text
    assert "'int'" in text


def test_a_missing_height_is_refused_with_a_message_naming_the_parameter() -> None:
    """A ``None`` ``height`` is refused before the driver is called.

    ``height`` is a data parameter the tool schema *does* describe,
    so a model can synthesize the wrong shape here as easily as it
    can reach the verb with the right one.  ``None`` is the "the
    model left it out" shape: the verb owes an envelope-shaped
    refusal naming the parameter and the remedy, and this cell
    fixes that the refusal fires before the driver is reached.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_stand_height(driver=driver, height=None)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_stand_height" in text
    assert "height" in text
    assert driver.calls == []


def test_a_wrong_shape_height_is_refused_with_a_message_naming_the_parameter() -> None:
    """A non-numeric ``height`` is refused before the driver is called.

    A model may synthesize the height as a string ``"0.5"``; the
    SDK's ``SetStandHeight`` handler is float-only and the driver's
    own path would raise on a non-numeric shape.  The shared
    :func:`~strands_robots.utils.finite_number_error` validator
    refuses the shape here so the envelope names ``height`` and the
    remedy.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_stand_height(driver=driver, height="0.5")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_stand_height" in text
    assert "height" in text
    assert driver.calls == []


def test_a_boolean_height_is_refused_because_it_would_act_as_a_silent_number() -> None:
    """A ``True`` payload is refused rather than coerced to ``1.0`` meters.

    ``bool`` is an ``int`` subclass, so a caller passing ``True``
    would reach the SDK's ``SetStandHeight(1.0)`` call - a nearly-max
    stance - through a signature that names none of that; ``False``
    would collapse to the LOW / crouched stance.  The shared
    :func:`~strands_robots.utils.finite_number_error` validator
    refuses ``bool`` explicitly rather than silently transitioning to
    a stance the caller did not name; a caller who wants a specific
    height reaches the verb with a float explicitly.  This cell holds
    the guard.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_stand_height(driver=driver, height=True)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_stand_height" in text
    assert "height" in text
    assert driver.calls == []


def test_a_nan_height_is_refused_before_the_driver_is_called() -> None:
    """A ``nan`` ``height`` is refused rather than poisoning the SDK call.

    ``nan`` compares false against every finite bound; the SDK's
    ``SetStandHeight`` handler would silently accept the frame and
    the driver's own arithmetic on the target would produce nonsense
    stance targets.  The shared validator refuses the shape here so
    the envelope names the parameter and the remedy is decidable
    rather than firmware-dependent.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_stand_height(driver=driver, height=float("nan"))
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_stand_height" in text
    assert "height" in text
    assert driver.calls == []


def test_an_inf_height_is_refused_before_the_driver_is_called() -> None:
    """An ``inf`` ``height`` is refused rather than reaching the SDK.

    Positive infinity collapses the stand-height arithmetic to an
    unbounded target; the firmware would clamp but the driver's own
    :func:`float` conversion path may not.  The shared validator
    refuses the shape here so the envelope names the parameter and
    the remedy is decidable rather than platform-dependent.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_stand_height(driver=driver, height=float("inf"))
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_stand_height" in text
    assert "height" in text
    assert driver.calls == []


def test_a_zero_height_is_admitted_because_the_domain_includes_the_low_stance() -> None:
    """A ``height=0.0`` reaches the driver: 0 is the LOW / crouched stance.

    Unlike ``g1_set_fsm``'s ``wait``, ``height`` is a caller-facing
    value the SDK's ``SetStandHeight`` admits at ``0`` (the neon
    bundle documents ``0.0`` as the crouched stance).  The finite-
    number validator - not the positive-finite one - admits ``0.0``,
    so this cell pins that a caller who wants the LOW stance reaches
    the driver with the target verbatim.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, height=0.0)

    assert result["status"] == "success"
    assert driver.calls == [0.0]


def test_a_negative_height_reaches_the_driver_for_the_highstand_fallback() -> None:
    """A negative ``height`` is passed to the driver: it selects HighStand.

    The neon bundle's ``g1_set_stand_height`` verb documented the
    negative-value fallback as its one addition over the raw SDK
    call: a caller passing any negative number reaches the SDK's
    ``HighStand`` fallback (which uses ``UINT32_MAX`` as the height
    sentinel).  The verb passes the value through unchanged and the
    driver's method decides the routing; this cell pins that the
    verb does not refuse a negative value on the ergonomic contract.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, height=-0.5)

    assert result["status"] == "success"
    assert driver.calls == [-0.5]
