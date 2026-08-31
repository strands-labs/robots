"""``g1_shake_hand_loco`` returns exactly what ``G1Driver.shake_hand_loco`` gives it.

``g1_shake_hand_loco`` is the write-side companion to the neon
bundle's ``g1_shake_hand_loco`` verb: where the neon verb wraps
``LocoClient.ShakeHand`` (which internally composes a ``SetTaskId``
payload against a three-stage table) under a single-writer lock,
this one hands a target ``stage`` to the driver's own write path
and reads back the envelope the driver produced.  The driver's
``G1Driver.shake_hand_loco`` method is not yet plumbed today (refs
strands-labs/robots#358 for the SDK-facing gate work the write
belongs on), so the
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
grader refuses a handle without a ``shake_hand_loco`` accessor
with a message naming the verb, the ``driver`` parameter and the
accessor; these tests fix the shape the verb passes through for
each of the driver's current and future outcomes (a driver-side
refusal, a future success envelope, and the verb's own ``driver``
/ ``stage`` refusals).

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
import sys
from typing import Any

from strands_robots.tools.g1.g1_shake_hand_loco import g1_shake_hand_loco


class _StubG1Driver:
    """A driver double whose ``shake_hand_loco`` returns a fixed envelope.

    ``g1_shake_hand_loco`` calls ``driver.shake_hand_loco(stage)``
    and returns the envelope verbatim.  This double sits under the
    same interface without pulling the real driver's imports (the
    real class reaches CycloneDDS at construction time in some
    paths), so a test can hand a wired-shape envelope to the verb
    without a bus.  ``calls`` records the ``stage`` per invocation
    so a test can pin "the verb writes the driver exactly once"
    and "the verb passes the argument through unchanged" without
    asking the driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls: list[int] = []

    def shake_hand_loco(self, stage: int) -> dict[str, Any]:
        self.calls.append(stage)
        return self._envelope


def _call(
    driver: Any,
    *,
    stage: int | None = 0,
) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim.  This helper is where a
    shape drift would surface once, rather than at every call
    site.  The default ``stage=0`` is the reach-out stage the neon
    bundle documented, so a call that omits the value here still
    reaches the driver with a stage the SDK's dispatcher admits.
    """
    return g1_shake_hand_loco(driver=driver, stage=stage)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a
    submodule at import time would break every headless CI runner
    and Thor before an office bring-up.  The driver enforces the
    same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the
    only path that loads the SDK); this cell holds the
    shake-hand-loco verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_shake_hand_loco")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_shake_hand_loco imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        "loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_side_refusal_surfaces_verbatim() -> None:
    """The driver's refusal envelope round-trips through the verb unchanged.

    ``G1Driver.shake_hand_loco`` will (once landed) refuse an
    SDK-side raise with a named error envelope, and today's driver
    refuses every call because the method is not yet plumbed.
    Both shapes are ``{"status": "error", "content": [{"text":
    ...}]}`` and the verb passes either through; a wording drift
    on the driver side moves this verb with it (refs
    strands-labs/robots#2874).
    """
    refusal_text = "shake_hand_loco: ShakeHand(stage=0) raised: RPC_CLIENT_API_TIMEOUT"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, stage=0)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_future_success_envelope_round_trips_verbatim() -> None:
    """Once ``shake_hand_loco`` lands, the success envelope surfaces verbatim.

    When the driver's method is plumbed,
    ``G1Driver.shake_hand_loco`` will surface the SDK's outcome
    inside a ``{"status": "success", "content": [{"json": {"rc":
    0, "message": "ShakeHand(stage=0) dispatched"}}]}`` envelope.
    The verb does not reshape it - a future field the driver adds
    reaches a caller the moment the driver writes it, and this
    cell holds that pass-through explicit so the verb is ready
    the moment the write path lands.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "rc": 0,
                    "message": "ShakeHand(stage=0) dispatched",
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, stage=0)

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
    result = g1_shake_hand_loco(driver=None, stage=0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_shake_hand_loco" in text
    assert "driver" in text


def test_a_wrong_shape_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot *name* (string) is refused before the accessor call.

    A model that synthesizes the argument as a robot name reaches
    the verb with a ``str``.  The verb owes an envelope-shaped
    refusal that names the type it received and the remedy - the
    four invariants every ``@tool`` handler in this package holds
    - and this cell fixes the shape.
    """
    result = g1_shake_hand_loco(driver="unitree_g1", stage=0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_shake_hand_loco" in text
    assert "'str'" in text


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.shake_hand_loco``.

    A double-call from a wrapper that retried inside the verb
    would issue two ``ShakeHand`` writes on the same admission
    window; the SDK's handler is not re-entrant and neon's own
    bundle held a single-writer lock.  This cell pins the verb to
    a single driver call per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    _call(driver)
    assert len(driver.calls) == 1


def test_the_verb_passes_the_argument_through_unchanged_reach() -> None:
    """A ``stage=0`` reaches the driver: reach out.

    The verb's contract is a pass-through: it does not translate
    argument names, does not synthesize defaults the driver did
    not ask for, and does not compose the ``SetTaskId`` payload
    itself.  This cell fixes that ``stage=0`` reaches the driver
    method verbatim - the reach-out stage the neon bundle
    observed, and the ``composed_task_id`` the driver's own
    write path composes from it.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, stage=0)

    assert result["status"] == "success"
    assert driver.calls == [0]


def test_the_verb_passes_the_argument_through_unchanged_shake() -> None:
    """A ``stage=1`` reaches the driver: shake extended hand.

    The neon bundle observed the shake stage as the second
    admitted target of ``LocoClient.ShakeHand``.  This cell pins
    that the verb reaches the driver
    with ``1`` verbatim (no substitution to the reach-out
    default), so a caller upgrading from the neon bundle reaches
    the same behaviour.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, stage=1)

    assert result["status"] == "success"
    assert driver.calls == [1]


def test_the_verb_passes_the_argument_through_unchanged_toggle() -> None:
    """A ``stage=-1`` reaches the driver: toggle the SDK's internal counter.

    The SDK's ``ShakeHand`` admits ``-1`` as a sentinel that
    toggles its internal stage counter (the SDK's own default
    reads through it); the read-only envelope names ``-1`` as
    the third admitted stage.  This cell pins that the sentinel
    reaches the driver verbatim, so a caller who wants the SDK's
    internal-counter semantics reaches the driver with ``-1``
    (not the reach-out ``0`` default substituted at the verb).
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, stage=-1)

    assert result["status"] == "success"
    assert driver.calls == [-1]


def test_the_wrong_shape_driver_is_not_called() -> None:
    """A wrong-shape driver is refused before the accessor path.

    The shared
    :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard is called first; a numeric handle is refused with an
    envelope and the driver's method is never reached.  This cell
    holds the ordering by grading a handle that would raise if
    called (``int`` has no ``shake_hand_loco`` attribute) and
    observing that the refusal envelope has the four invariants
    without an exception in flight.
    """
    result = g1_shake_hand_loco(driver=42, stage=0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_shake_hand_loco" in text
    assert "'int'" in text


def test_a_missing_stage_is_refused_with_a_message_naming_the_parameter() -> None:
    """A ``None`` ``stage`` is refused before the driver is called.

    ``stage`` is a data parameter the tool schema *does* describe,
    so a model can synthesize the wrong shape here as easily as
    it can reach the verb with the right one.  ``None`` is the
    "the model left it out" shape: the three admitted stages are
    the data points the read-only envelope surfaces, and a caller
    who did not pass one has not decided the write.  The verb
    owes an envelope-shaped refusal naming the parameter and the
    remedy, and this cell fixes that the refusal fires before the
    driver is reached.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_shake_hand_loco(driver=driver, stage=None)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_shake_hand_loco" in text
    assert "stage" in text
    assert driver.calls == []


def test_a_bool_stage_is_refused_because_int_coercion_would_be_silent() -> None:
    """A ``bool`` ``stage`` is refused rather than routed through ``int()``.

    ``bool`` is an ``int`` subclass, so a bare
    ``isinstance(..., int)`` test would let ``True`` through as a
    silent ``1`` (shake) and ``False`` as a silent ``0`` (reach
    out) - both inside the admitted set, so the SDK's dispatcher
    would silently transition to a stage the caller writing the
    boolean did not name on purpose.  Refusing ``bool`` explicitly
    matches the shape refusal ``g1_balance_stand`` renders for the
    same reason on ``balance_mode``, and keeps the verb's message
    naming the parameter and the shape received.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_shake_hand_loco(driver=driver, stage=True)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_shake_hand_loco" in text
    assert "stage" in text
    assert "'bool'" in text
    assert driver.calls == []


def test_a_string_stage_is_refused_with_a_message_naming_the_type() -> None:
    """A string ``stage`` is refused rather than routed through ``int()``.

    A model may synthesize the stage as a numeric string ``"0"``
    or a label ``"reach"``; the neon bundle's own ``int(stage)``
    coercion would raise on the label but silently parse the
    numeric string as an admitted stage.  Refusing the shape
    here keeps the refusal decidable at the tool surface rather
    than SDK-version dependent, and matches the ``g1_balance_stand``
    verb's own ``str`` refusal shape a caller can grep for.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_shake_hand_loco(driver=driver, stage="0")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_shake_hand_loco" in text
    assert "stage" in text
    assert "'str'" in text
    assert driver.calls == []


def test_a_float_stage_is_refused_with_a_message_naming_the_type() -> None:
    """A ``float`` ``stage`` is refused rather than routed through ``int()``.

    The SDK's three-stage table is a discrete set - there is no
    continuous middle ground the SDK admits - so a caller passing
    ``0.5`` is not naming a stage on purpose.  The neon bundle's
    ``int(stage)`` coercion would silently truncate ``0.5`` to
    ``0`` (reach out) and ``-0.7`` to ``0`` too; refusing the
    shape here matches the ``g1_balance_stand`` verb's own
    ``float`` refusal so both paths render the same shape a
    caller can grep for.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_shake_hand_loco(driver=driver, stage=0.5)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_shake_hand_loco" in text
    assert "stage" in text
    assert "'float'" in text
    assert driver.calls == []
