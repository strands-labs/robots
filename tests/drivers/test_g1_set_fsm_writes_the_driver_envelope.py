"""``g1_set_fsm`` returns exactly what ``G1Driver.set_fsm`` gives it.

``g1_set_fsm`` is the write side of the FSM-id conversation: its own
``fsm_id`` docstring lists the ids the SDK's ``LocoClient.SetFsmId``
handler admits, and
this one hands one of those ids to the driver's own SetFsmId write
path and reads back the fsm-before / fsm-after / rc round-trip the
neon bundle's ``g1_set_fsm`` verb documented.  The driver's
``G1Driver.set_fsm`` method is not yet
plumbed today (refs strands-labs/robots#358 for the SDK-facing gate
work the write belongs on), so the :func:`live_handle_refusal`
grader refuses a handle without a ``set_fsm`` accessor with a
message naming the verb, the ``driver`` parameter and the accessor;
these tests fix the shape the verb passes through for each of the
driver's current and future outcomes (a driver-side refusal, a gate
refusal, a future success envelope carrying the fsm-before /
fsm-after round-trip, and the verb's own ``driver`` / ``fsm_id`` /
``wait`` refusals).

The refusal-string tests do not restate the driver's exact prose -
that would trap the verb to the driver's refusal wording of one
release, which is exactly what the driver's own release notes say
verbatim quotes should not do (refs strands-labs/robots#2874).  They
grade the shape (``status="error"``, an envelope-shaped
``content[0]["text"]``, the verb's own four refusal invariants when
the verb produced them) and pass the driver's own text through
unchanged when the driver produced it.  The SDK-load-hygiene contract
every file under :mod:`strands_robots.tools.g1` carries is fixed
first: importing the module must not pull any ``unitree_sdk2py``
submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_set_fsm import g1_set_fsm


class _StubG1Driver:
    """A driver double whose ``set_fsm`` returns a fixed envelope.

    ``g1_set_fsm`` calls ``driver.set_fsm(fsm_id, wait=...)`` and
    returns the envelope verbatim.  This double sits under the same
    interface without pulling the real driver's imports (the real
    class reaches CycloneDDS at construction time in some paths), so
    a test can hand a wired-shape envelope to the verb without a
    bus.  ``calls`` records ``(fsm_id, wait)`` per invocation so a
    test can pin "the verb writes the driver exactly once" and "the
    verb passes the two arguments through unchanged" without asking
    the driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls: list[tuple[int, float]] = []

    def set_fsm(self, fsm_id: int, wait: float = 3.0) -> dict[str, Any]:
        self.calls.append((fsm_id, wait))
        return self._envelope


def _call(
    driver: Any,
    *,
    fsm_id: int | None = 500,
    wait: float = 3.0,
) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim.  This helper is where a shape
    drift would surface once, rather than at every call site.  The
    default ``fsm_id=500`` is the "Start" state - one of the arm-SDK
    admission set - so a call that omits the id here still reaches
    the driver with an admitted target.
    """
    return g1_set_fsm(driver=driver, fsm_id=fsm_id, wait=wait)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a submodule
    at import time would break every headless CI runner and Thor
    before an office bring-up.  The driver enforces the same rule
    against itself (:func:`~strands_robots.tools.g1._g1_common.ensure_dds`
    is the only path that loads the SDK); this cell holds the
    set-fsm verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_set_fsm")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_set_fsm imports pulled SDK submodules: "
        f"{leaked}. The rule for this package is that the SDK loads only "
        "inside function bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_side_refusal_surfaces_verbatim() -> None:
    """The driver's refusal envelope round-trips through the verb unchanged.

    ``G1Driver.set_fsm`` will (once landed) refuse an id the
    SDK's ``SetFsmId`` handler rejects with a ``rc=7302`` envelope,
    and today's driver refuses every call because the method is not
    yet plumbed.  Both shapes are ``{"status": "error", "content":
    [{"text": ...}]}`` and the verb passes either through; a wording
    drift on the driver side moves this verb with it (refs
    strands-labs/robots#2874).
    """
    refusal_text = "set_fsm: SetFsmId(4=Locomotion) rc=7302 (Invalid FSM id (loco))"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, fsm_id=4)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_gate_refusal_from_the_driver_surfaces_verbatim() -> None:
    """The driver's FSM/battery gate refusal reaches a caller unchanged.

    :meth:`G1Driver._check_motion_gates` is the arm-write gate;
    ``SetFsmId`` sits on the loco side of the same DDS singleton, so
    the driver's own ``set_fsm`` may or may not consult that gate.
    When it does refuse (battery under the floor, a driver-side
    kill-switch), the verb passes the envelope through - a second
    gate call here would double the read against the same cache and
    fork the source of truth for a rule the driver's own path
    already enforces (refs strands-labs/robots#2916).
    """
    refusal_text = "set_fsm refused: battery under the driver's floor at 8%"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_future_success_envelope_round_trips_verbatim() -> None:
    """Once ``set_fsm`` lands, the fsm-round-trip envelope surfaces verbatim.

    When the driver's method is plumbed, ``G1Driver.set_fsm``
    will surface the fsm-before / fsm-after / rc / message
    round-trip inside a ``{"status": "success", "content": [{"json":
    {"fsm_before": ..., "fsm_after": ..., "rc": ..., "message":
    ...}}]}`` envelope.  The verb does not reshape it - a future
    field the driver adds reaches a caller the moment the driver
    writes it, and this cell holds that pass-through explicit so
    the verb is ready the moment the write path lands.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "fsm_before": 1,
                    "fsm_after": 500,
                    "rc": 0,
                    "message": "SetFsmId(500=Start) rc=0 (OK) | FSM 1 -> 500",
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver)

    assert result is envelope or result == envelope
    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["fsm_before"] == 1
    assert payload["fsm_after"] == 500
    assert payload["rc"] == 0


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
    result = g1_set_fsm(driver=None, fsm_id=500)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_fsm" in text
    assert "driver" in text


def test_a_wrong_shape_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot *name* (string) is refused before the accessor call.

    A model that synthesizes the argument as a robot name reaches the
    verb with a ``str``.  The verb owes an envelope-shaped refusal
    that names the type it received and the remedy - the four
    invariants every ``@tool`` handler in this package holds - and
    this cell fixes the shape.
    """
    result = g1_set_fsm(driver="unitree_g1", fsm_id=500)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_fsm" in text
    assert "'str'" in text


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.set_fsm``.

    A double-call from a wrapper that retried inside the verb would
    issue two ``SetFsmId`` writes on the same admission window; the
    SDK's handler is not re-entrant and neon's own bundle held a
    single-writer lock.  This cell pins the verb to a single driver
    call per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    _call(driver)
    assert len(driver.calls) == 1


def test_the_verb_passes_the_arguments_through_unchanged() -> None:
    """The two arguments the driver receives are the ones the caller passed.

    The verb's contract is a pass-through: it does not translate
    argument names, does not synthesize defaults the driver did not
    ask for, and does not rearrange keyword order.  This cell fixes
    that ``fsm_id`` and ``wait`` reach the driver method verbatim -
    a rename or reorder on either side is a driver-level contract
    change, not a silent verb-side translation.
    """
    envelope = {"status": "success", "content": [{"json": {}}]}
    driver = _StubG1Driver(envelope=envelope)
    _call(driver, fsm_id=801, wait=1.5)
    assert driver.calls == [(801, 1.5)]


def test_a_default_call_passes_the_signature_default_wait_to_the_driver() -> None:
    """A caller omitting ``wait`` still reaches the driver with the default.

    The verb's ``wait=3.0`` default matches the neon bundle's own
    default and the driver's method signature.  A caller that names
    only ``driver`` and ``fsm_id`` reaches the driver with the same
    default the driver would have filled in on its own; this cell
    holds the parity so a driver-side default change surfaces on the
    verb without a silent divergence.
    """
    envelope = {"status": "success", "content": [{"json": {}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = g1_set_fsm(driver=driver, fsm_id=500)

    assert result["status"] == "success"
    assert driver.calls == [(500, 3.0)]


def test_the_wrong_shape_driver_is_not_called() -> None:
    """A wrong-shape driver is refused before the accessor path.

    The shared :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard is called first; a numeric handle is refused with an
    envelope and the driver's method is never reached.  This cell
    holds the ordering by grading a handle that would raise if
    called (``int`` has no ``set_fsm`` attribute) and observing
    that the refusal envelope has the four invariants without an
    exception in flight.
    """
    result = g1_set_fsm(driver=42, fsm_id=500)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_fsm" in text
    assert "'int'" in text


def test_a_missing_fsm_id_is_refused_with_a_message_naming_the_parameter() -> None:
    """A ``None`` ``fsm_id`` is refused before the driver is called.

    ``fsm_id`` is a data parameter the tool schema *does* describe,
    so a model can synthesize the wrong shape here as easily as it
    can reach the verb with the right one.  ``None`` is the "the
    model left it out" shape: the verb owes an envelope-shaped
    refusal naming the parameter and the remedy, and this cell
    fixes that the refusal fires before the driver is reached.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_fsm(driver=driver, fsm_id=None)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_fsm" in text
    assert "fsm_id" in text
    assert driver.calls == []


def test_a_wrong_shape_fsm_id_is_refused_with_a_message_naming_the_type() -> None:
    """A non-int ``fsm_id`` is refused before the driver is called.

    A model may synthesize the id as a string ``"500"`` or a float
    ``500.0``; the SDK's ``SetFsmId`` handler is int-only and the
    driver's own path would raise on a non-int shape.  This cell
    fixes the four invariants (envelope, verb name, parameter name,
    type name) on the verb-side refusal so a caller reading the
    envelope can fix the call without inspecting the driver's raise.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_fsm(driver=driver, fsm_id="500")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_fsm" in text
    assert "fsm_id" in text
    assert "'str'" in text
    assert driver.calls == []


def test_a_boolean_fsm_id_is_refused_because_it_would_act_as_a_silent_one() -> None:
    """A ``True`` payload is refused rather than coerced to ``1`` Damp.

    ``bool`` is an ``int`` subclass, so a caller passing ``True``
    would reach the SDK's ``SetFsmId(1)`` call - a transition to
    Damp - through a signature that names none of that.  The verb
    refuses the shape rather than silently transitioning to a state
    the caller did not name; a caller who wants Damp reaches the
    verb with ``fsm_id=1`` explicitly.  This cell holds the guard.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_fsm(driver=driver, fsm_id=True)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_fsm" in text
    assert "fsm_id" in text
    assert "'bool'" in text
    assert driver.calls == []


def test_a_negative_wait_is_refused_before_the_driver_is_called() -> None:
    """A negative ``wait`` is refused with a message naming the parameter.

    ``wait`` is the seconds-to-sleep between ``SetFsmId`` and the
    ``fsm_after`` read; a negative value reverses the sleep's
    contract and the driver would raise on the :func:`time.sleep`
    call.  The shared
    :func:`~strands_robots.utils.positive_finite_number_error`
    validator refuses the shape here so the envelope names ``wait``
    and the remedy.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_fsm(driver=driver, fsm_id=500, wait=-1.0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_fsm" in text
    assert "wait" in text
    assert driver.calls == []


def test_a_nan_wait_is_refused_before_the_driver_is_called() -> None:
    """A ``nan`` ``wait`` is refused rather than poisoning the sleep call.

    ``nan`` compares false against every finite bound; a sleep call
    on ``nan`` raises on some platforms and returns immediately on
    others.  The shared validator refuses the shape here so the
    envelope names the parameter and the remedy is decidable rather
    than platform-dependent.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_fsm(driver=driver, fsm_id=500, wait=float("nan"))
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_fsm" in text
    assert "wait" in text
    assert driver.calls == []


def test_a_zero_wait_is_refused_because_the_domain_is_positive_finite() -> None:
    """A ``wait=0.0`` is refused with a message naming the parameter.

    A zero sleep collapses the fsm-after read to the same instant as
    the SetFsmId return; the SDK's transition takes real time and
    ``fsm_after`` would report the pre-transition state.  The shared
    validator refuses the shape here so the envelope names ``wait``
    and points the caller at a positive finite value.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_set_fsm(driver=driver, fsm_id=500, wait=0.0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_set_fsm" in text
    assert "wait" in text
    assert driver.calls == []
