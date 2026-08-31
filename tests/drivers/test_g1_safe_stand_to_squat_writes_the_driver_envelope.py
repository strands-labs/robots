"""``g1_safe_stand_to_squat`` returns exactly what ``G1Driver.safe_stand_to_squat`` gives it.

``g1_safe_stand_to_squat`` is the write-side companion to the
neon bundle's ``g1_safe_stand_to_squat`` verb: where the neon verb
wraps ``LocoClient.Damp`` + ``LocoClient.SetFsmId(2)`` under an
``_assert_safe_for_damp`` FSM+pose guard, this one hands a target
Damp-preamble duration to the driver's own write path and reads
back the envelope the driver produced.  The driver's
``G1Driver.safe_stand_to_squat`` method is not yet plumbed today
(refs strands-labs/robots#358 for the SDK-facing gate work the
write belongs on), so the :func:`live_handle_refusal` grader
refuses a handle without a ``safe_stand_to_squat`` accessor with a
message naming the verb, the ``driver`` parameter and the
accessor; these tests fix the shape the verb passes through for
each of the driver's current and future outcomes (a driver-side
refusal, a future success envelope, and the verb's own ``driver``
/ ``preamble_s`` refusals).

The refusal-string tests do not restate the driver's exact prose -
that would trap the verb to the driver's refusal wording of one
release, which is exactly what the driver's own release notes say
verbatim quotes should not do (refs strands-labs/robots#2874).
They grade the shape (``status="error"``, an envelope-shaped
``content[0]["text"]``, the verb's own four refusal invariants
when the verb produced them) and pass the driver's own text
through unchanged when the driver produced it.  The SDK-load-
hygiene contract every file under :mod:`strands_robots.tools.g1`
carries is fixed first: importing the module must not pull any
``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

from strands_robots.tools.g1.g1_safe_stand_to_squat import g1_safe_stand_to_squat


class _StubG1Driver:
    """A driver double whose ``safe_stand_to_squat`` returns a fixed envelope.

    ``g1_safe_stand_to_squat`` calls
    ``driver.safe_stand_to_squat(preamble_s)`` and returns the
    envelope verbatim.  This double sits under the same interface
    without pulling the real driver's imports (the real class
    reaches CycloneDDS at construction time in some paths), so a
    test can hand a wired-shape envelope to the verb without a
    bus.  ``calls`` records the ``preamble_s`` per invocation so a
    test can pin "the verb writes the driver exactly once" and
    "the verb passes the argument through unchanged" without
    asking the driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls: list[float] = []

    def safe_stand_to_squat(self, preamble_s: float) -> dict[str, Any]:
        self.calls.append(preamble_s)
        return self._envelope


def _call(
    driver: Any,
    *,
    preamble_s: float = 0.5,
) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim.  This helper is where a
    shape drift would surface once, rather than at every call
    site.  The default ``preamble_s=0.5`` is the neon-bundle-
    observed default the verb inherits, so a call that omits the
    value here still reaches the driver with the same target the
    field notes against the real robot preferred.
    """
    return g1_safe_stand_to_squat(driver=driver, preamble_s=preamble_s)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a
    submodule at import time would break every headless CI runner
    and Thor before an office bring-up.  The driver enforces the
    same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the
    only path that loads the SDK); this cell holds the
    safe-squat-to-stand verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_safe_stand_to_squat")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_safe_stand_to_squat imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        "loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_side_refusal_surfaces_verbatim() -> None:
    """The driver's refusal envelope round-trips through the verb unchanged.

    ``G1Driver.safe_stand_to_squat`` will (once landed) refuse an
    SDK-side raise, an FSM outside ``{500, 501, 801}`` with a named error envelope,
    and today's driver refuses every call because the method is
    not yet plumbed.  Both shapes are ``{"status": "error",
    "content": [{"text": ...}]}`` and the verb passes either
    through; a wording drift on the driver side moves this verb
    with it (refs strands-labs/robots#2874).
    """
    refusal_text = (
        "safe_stand_to_squat: FSM=None is not in {500, 501, 801}. "
        "Damping outside a controller-managed state risks collapse."
    )
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, preamble_s=0.5)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_future_success_envelope_round_trips_verbatim() -> None:
    """Once ``safe_stand_to_squat`` lands, the success envelope surfaces verbatim.

    When the driver's method is plumbed,
    ``G1Driver.safe_stand_to_squat`` will surface the SDK's
    outcome inside a ``{"status": "success", "content": [{"json":
    {"rc": 0, "message": "Damp -> sleep(0.5s) -> SetFsmId(2)
    dispatched"}}]}`` envelope.  The verb does not reshape it - a
    future field the driver adds reaches a caller the moment the
    driver writes it, and this cell holds that pass-through
    explicit so the verb is ready the moment the write path
    lands.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "rc": 0,
                    "message": "Damp -> sleep(0.5s) -> SetFsmId(2=Squat) dispatched",
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, preamble_s=0.5)

    assert result is envelope or result == envelope
    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["rc"] == 0


def test_a_missing_driver_is_refused_with_a_message_naming_the_parameter() -> None:
    """A ``None`` driver is refused before the accessor is called.

    ``driver`` is a live Python object typed :class:`~typing.Any`,
    so the tool schema carries no signal that a caller cannot
    synthesize it.  A model that leaves the parameter out reaches
    the verb with ``None``, and the verb owes an envelope-shaped
    refusal instead of an exception the ``@tool`` wrapper cannot
    format.  The shared
    :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard produces it; this cell fixes that the guard is called
    before the accessor path.
    """
    result = g1_safe_stand_to_squat(driver=None, preamble_s=0.5)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_safe_stand_to_squat" in text
    assert "driver" in text


def test_a_wrong_shape_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot *name* (string) is refused before the accessor call.

    A model that synthesizes the argument as a robot name reaches
    the verb with a ``str``.  The verb owes an envelope-shaped
    refusal that names the type it received and the remedy - the
    four invariants every ``@tool`` handler in this package holds
    - and this cell fixes the shape.
    """
    result = g1_safe_stand_to_squat(driver="unitree_g1", preamble_s=0.5)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_safe_stand_to_squat" in text
    assert "'str'" in text


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.safe_stand_to_squat``.

    A double-call from a wrapper that retried inside the verb
    would issue two Damp+SetFsmId(2) writes on the same
    admission window; the SDK's handlers are not re-entrant and
    neon's own bundle held a single-writer lock.  This cell pins
    the verb to a single driver call per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    _call(driver)
    assert len(driver.calls) == 1


def test_the_verb_passes_the_argument_through_unchanged() -> None:
    """The preamble the driver receives is the one the caller passed.

    The verb's contract is a pass-through: it does not translate
    argument names, does not synthesize defaults the driver did
    not ask for, and does not clamp the value.  This cell fixes
    that ``preamble_s`` reaches the driver method verbatim - a
    rename or coercion on either side is a driver-level contract
    change, not a silent verb-side translation.
    """
    envelope = {"status": "success", "content": [{"json": {}}]}
    driver = _StubG1Driver(envelope=envelope)
    _call(driver, preamble_s=1.25)
    assert driver.calls == [1.25]


def test_the_default_preamble_matches_the_neon_bundle_field_notes() -> None:
    """A caller who omits ``preamble_s`` reaches the driver with ``0.5``.

    The neon bundle's ``g1_safe_stand_to_squat`` verb defaulted to
    ``preamble_s=0.5`` seconds
    (``cagataycali/neon-the-g1/tools/g1_safe_posture.py``); the
    bundle's field notes name the same value as its observed
    default.  This cell
    pins that the verb's own default matches, so a caller
    upgrading from the neon bundle reaches the driver with the
    same target the field notes against the real robot preferred.
    """
    envelope = {"status": "success", "content": [{"json": {}}]}
    driver = _StubG1Driver(envelope=envelope)
    g1_safe_stand_to_squat(driver=driver)
    assert driver.calls == [0.5]


def test_the_wrong_shape_driver_is_not_called() -> None:
    """A wrong-shape driver is refused before the accessor path.

    The shared
    :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard is called first; a numeric handle is refused with an
    envelope and the driver's method is never reached.  This cell
    holds the ordering by grading a handle that would raise if
    called (``int`` has no ``safe_stand_to_squat`` attribute) and
    observing that the refusal envelope has the four invariants
    without an exception in flight.
    """
    result = g1_safe_stand_to_squat(driver=42, preamble_s=0.5)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_safe_stand_to_squat" in text
    assert "'int'" in text


def test_a_missing_preamble_is_refused_with_a_message_naming_the_parameter() -> None:
    """A ``None`` ``preamble_s`` is refused before the driver is called.

    ``preamble_s`` is a data parameter the tool schema *does*
    describe (a positive finite float duration), so a model can
    synthesize the wrong shape here as easily as it can reach the
    verb with the right one.  ``None`` is the "the model
    explicitly nulled the default" shape: the verb owes an
    envelope-shaped refusal naming the parameter and the remedy,
    and this cell fixes that the refusal fires before the driver
    is reached.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_safe_stand_to_squat(driver=driver, preamble_s=None)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_safe_stand_to_squat" in text
    assert "preamble_s" in text
    assert driver.calls == []


def test_a_wrong_shape_preamble_is_refused_with_a_message_naming_the_parameter() -> None:
    """A non-numeric ``preamble_s`` is refused before the driver is called.

    A model may synthesize the duration as a string ``"0.5"``; the
    driver's own :func:`time.sleep` call is float-only (the neon
    bundle's own wrapper coerced through :class:`float` before
    dispatch, so a string reaching that path would either be
    silently ``float()``-coerced or raise on the wire).  The verb
    refuses the shape here so the envelope names ``preamble_s`` and
    the remedy is decidable rather than SDK-version dependent.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_safe_stand_to_squat(driver=driver, preamble_s="0.5")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_safe_stand_to_squat" in text
    assert "preamble_s" in text
    assert driver.calls == []


def test_a_zero_preamble_is_refused_because_it_collapses_to_a_bare_squat_to_stand() -> None:
    """A ``preamble_s=0.0`` is refused before the driver is called.

    A zero-length preamble collapses to a bare ``SetFsmId(2)`` write into Squat (FSM 2)
    write, which the neon bundle's own wrapper documented as a
    distinct ``use_unitree(service='loco',
    operation='SetFsmId(2)', ...)`` verb.  The shared
    :func:`~strands_robots.utils.positive_finite_number_error`
    validator refuses the ``0`` boundary because a caller who
    wants the bare transition names the ``use_unitree`` verb
    explicitly rather than silently reaching this compound
    handler with a preamble that skips the Damp fire.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_safe_stand_to_squat(driver=driver, preamble_s=0.0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_safe_stand_to_squat" in text
    assert "preamble_s" in text
    assert driver.calls == []


def test_a_negative_preamble_is_refused_because_time_sleep_cannot_use_it() -> None:
    """A negative ``preamble_s`` is refused before the driver is called.

    The driver's own :func:`time.sleep` call refuses a negative
    value (Python's stdlib raises :class:`ValueError` on
    ``time.sleep(-0.5)``); refusing the shape here rather than
    letting the raise reach the driver keeps the envelope naming
    ``preamble_s`` and the remedy decidable.  The shared
    :func:`~strands_robots.utils.positive_finite_number_error`
    validator refuses the sign for the same reason it refuses on
    every other positive-real duration knob in this package.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_safe_stand_to_squat(driver=driver, preamble_s=-0.5)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_safe_stand_to_squat" in text
    assert "preamble_s" in text
    assert driver.calls == []


def test_a_boolean_preamble_is_refused_because_it_would_act_as_a_silent_float() -> None:
    """A ``True`` payload is refused rather than coerced to ``1.0``.

    ``bool`` is an ``int`` (and float-castable) subclass, so a
    caller passing ``True`` would reach the driver's
    ``time.sleep(1.0)`` (a preamble twice as long as the neon
    default) through a signature that names none of that;
    ``False`` would collapse to ``time.sleep(0.0)`` (skips the
    Damp preamble and silently collapses to a bare
    ``SetFsmId(2)`` write).  Refusing ``bool`` explicitly rather
    than silently transitioning matches the shape refusal every
    other numeric verb in this package renders on the same
    subclass hazard; a caller who wants a specific duration
    reaches the verb with a float explicitly.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_safe_stand_to_squat(driver=driver, preamble_s=True)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_safe_stand_to_squat" in text
    assert "preamble_s" in text
    assert driver.calls == []


def test_a_non_finite_preamble_is_refused_before_the_driver_is_called() -> None:
    """A ``nan`` / ``inf`` ``preamble_s`` is refused before the driver is called.

    ``time.sleep(math.inf)`` blocks the driver's write path
    indefinitely; ``time.sleep(math.nan)`` silently no-ops (the
    stdlib treats ``nan`` as a zero-length pause).  Neither is a
    duration a caller writing that value would have named on
    purpose - the shared
    :func:`~strands_robots.utils.positive_finite_number_error`
    validator refuses both for the same reason it refuses them
    on every other positive-real duration knob in this package.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_safe_stand_to_squat(driver=driver, preamble_s=math.inf)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_safe_stand_to_squat" in text
    assert "preamble_s" in text
    assert driver.calls == []

    driver_nan = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result_nan = g1_safe_stand_to_squat(driver=driver_nan, preamble_s=math.nan)
    assert result_nan["status"] == "error"
    text_nan = result_nan["content"][0]["text"]
    assert "g1_safe_stand_to_squat" in text_nan
    assert "preamble_s" in text_nan
    assert driver_nan.calls == []


def test_a_large_preamble_reaches_the_driver_unchanged() -> None:
    """A ``preamble_s=5.0`` reaches the driver: the verb does not domain-refuse.

    The neon bundle's field notes name the observed usable
    range; the module
    docstring names "does not refuse a preamble_s outside the
    neon-bundle-observed usable range" as one of the things this
    verb does not do.  Refusing an unlisted duration here would
    fork the neon bundle's admission set into a second source of
    truth this module would then have to keep in sync with the
    envelope lookup.  This cell pins that a caller who passes
    ``5.0`` reaches the driver's own refusal (or the SDK's
    long-sleep handler) through the verb's pass-through, rather
    than being intercepted here.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, preamble_s=5.0)

    assert result["status"] == "success"
    assert driver.calls == [5.0]
