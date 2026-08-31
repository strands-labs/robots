"""``g1_wave_hand_loco`` returns exactly what ``G1Driver.wave_hand_loco`` gives it.

``g1_wave_hand_loco`` is the write-side companion to the neon
bundle's ``g1_wave_hand_loco`` verb: where the neon verb wraps
``LocoClient.WaveHand`` (which internally composes one of two
``SetTaskId`` payloads) under a single-writer lock, this one hands
a target ``turn_flag`` to the driver's own write path and reads
back the envelope the driver produced.  The driver's
``G1Driver.wave_hand_loco`` method is not yet plumbed today (refs
strands-labs/robots#358 for the SDK-facing gate work the write
belongs on), so the
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
grader refuses a handle without a ``wave_hand_loco`` accessor with
a message naming the verb, the ``driver`` parameter and the
accessor; these tests fix the shape the verb passes through for
each of the driver's current and future outcomes (a driver-side
refusal, a future success envelope, and the verb's own ``driver``
/ ``turn_flag`` refusals).

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

from strands_robots.tools.g1.g1_wave_hand_loco import g1_wave_hand_loco


class _StubG1Driver:
    """A driver double whose ``wave_hand_loco`` returns a fixed envelope.

    ``g1_wave_hand_loco`` calls ``driver.wave_hand_loco(turn_flag)``
    and returns the envelope verbatim.  This double sits under the
    same interface without pulling the real driver's imports (the
    real class reaches CycloneDDS at construction time in some
    paths), so a test can hand a wired-shape envelope to the verb
    without a bus.  ``calls`` records the ``turn_flag`` per
    invocation so a test can pin "the verb writes the driver
    exactly once" and "the verb passes the argument through
    unchanged" without asking the driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls: list[bool] = []

    def wave_hand_loco(self, turn_flag: bool) -> dict[str, Any]:
        self.calls.append(turn_flag)
        return self._envelope


def _call(
    driver: Any,
    *,
    turn_flag: bool | None = False,
) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim.  This helper is where a
    shape drift would surface once, rather than at every call
    site.  The default ``turn_flag=False`` is the wave-in-place
    variant the neon bundle documented, so a call that omits the
    value here still reaches the driver with a target the
    controller admits.
    """
    return g1_wave_hand_loco(driver=driver, turn_flag=turn_flag)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a
    submodule at import time would break every headless CI runner
    and Thor before an office bring-up.  The driver enforces the
    same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the
    only path that loads the SDK); this cell holds the
    wave-hand-loco verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_wave_hand_loco")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_wave_hand_loco imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        "loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_side_refusal_surfaces_verbatim() -> None:
    """The driver's refusal envelope round-trips through the verb unchanged.

    ``G1Driver.wave_hand_loco`` will (once landed) refuse an
    SDK-side raise with a named error envelope, and today's driver
    refuses every call because the method is not yet plumbed.
    Both shapes are ``{"status": "error", "content": [{"text":
    ...}]}`` and the verb passes either through; a wording drift
    on the driver side moves this verb with it (refs
    strands-labs/robots#2874).
    """
    refusal_text = "wave_hand_loco: WaveHand(turn=False) raised: RPC_CLIENT_API_TIMEOUT"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, turn_flag=False)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_future_success_envelope_round_trips_verbatim() -> None:
    """Once ``wave_hand_loco`` lands, the success envelope surfaces verbatim.

    When the driver's method is plumbed,
    ``G1Driver.wave_hand_loco`` will surface the SDK's outcome
    inside a ``{"status": "success", "content": [{"json": {"rc":
    0, "message": "WaveHand(turn=False) dispatched"}}]}`` envelope.
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
                    "message": "WaveHand(turn=False) dispatched",
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, turn_flag=False)

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
    result = g1_wave_hand_loco(driver=None, turn_flag=False)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_wave_hand_loco" in text
    assert "driver" in text


def test_a_wrong_shape_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot *name* (string) is refused before the accessor call.

    A model that synthesizes the argument as a robot name reaches
    the verb with a ``str``.  The verb owes an envelope-shaped
    refusal that names the type it received and the remedy - the
    four invariants every ``@tool`` handler in this package holds
    - and this cell fixes the shape.
    """
    result = g1_wave_hand_loco(driver="unitree_g1", turn_flag=False)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_wave_hand_loco" in text
    assert "'str'" in text


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.wave_hand_loco``.

    A double-call from a wrapper that retried inside the verb
    would issue two ``WaveHand`` writes on the same admission
    window; the SDK's handler is not re-entrant and neon's own
    bundle held a single-writer lock.  This cell pins the verb to
    a single driver call per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    _call(driver)
    assert len(driver.calls) == 1


def test_the_verb_passes_the_argument_through_unchanged_false() -> None:
    """A ``turn_flag=False`` reaches the driver: wave in place.

    The verb's contract is a pass-through: it does not translate
    argument names, does not synthesize defaults the driver did
    not ask for, and does not compose the ``SetTaskId`` payload
    itself.  This cell fixes that ``turn_flag=False`` reaches the
    driver method verbatim - the wave-in-place variant the neon
    bundle observed, and the ``composed_task_id`` the driver's
    own write path composes from it.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, turn_flag=False)

    assert result["status"] == "success"
    assert driver.calls == [False]


def test_the_verb_passes_the_argument_through_unchanged_true() -> None:
    """A ``turn_flag=True`` reaches the driver: wave and turn around.

    The neon bundle observed the wave-and-turn-around variant as
    the second admitted target of ``LocoClient.WaveHand``.  This
    cell pins that the verb reaches the driver
    with ``True`` verbatim (no substitution to the in-place
    default), so a caller upgrading from the neon bundle reaches
    the same behaviour.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, turn_flag=True)

    assert result["status"] == "success"
    assert driver.calls == [True]


def test_the_wrong_shape_driver_is_not_called() -> None:
    """A wrong-shape driver is refused before the accessor path.

    The shared
    :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard is called first; a numeric handle is refused with an
    envelope and the driver's method is never reached.  This cell
    holds the ordering by grading a handle that would raise if
    called (``int`` has no ``wave_hand_loco`` attribute) and
    observing that the refusal envelope has the four invariants
    without an exception in flight.
    """
    result = g1_wave_hand_loco(driver=42, turn_flag=False)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_wave_hand_loco" in text
    assert "'int'" in text


def test_a_missing_turn_flag_is_refused_with_a_message_naming_the_parameter() -> None:
    """A ``None`` ``turn_flag`` is refused before the driver is called.

    ``turn_flag`` is a data parameter the tool schema *does*
    describe, so a model can synthesize the wrong shape here as
    easily as it can reach the verb with the right one.  ``None``
    is the "the model left it out" shape: the two admitted
    variants are the two data points the read-only envelope
    surfaces, and a caller who did not pass one has not decided
    the write.  The verb owes an envelope-shaped refusal naming
    the parameter and the remedy, and this cell fixes that the
    refusal fires before the driver is reached.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_wave_hand_loco(driver=driver, turn_flag=None)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_wave_hand_loco" in text
    assert "turn_flag" in text
    assert driver.calls == []


def test_an_int_turn_flag_is_refused_because_bool_coercion_would_be_silent() -> None:
    """An ``int`` ``turn_flag`` is refused rather than routed through ``bool()``.

    The neon wrapper called ``bool(turn)`` before the SDK saw the
    value, silently transforming ``1`` into ``True`` (wave and
    turn around) and ``0`` into ``False`` (wave in place); a
    caller reaching this verb with ``1`` has not named a
    turn-flag variant on purpose.  Refusing the non-``bool``
    shape explicitly matches the neon verb's
    own ``turn_flag must be bool`` refusal, so both paths render
    the same shape a caller can grep for.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_wave_hand_loco(driver=driver, turn_flag=1)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_wave_hand_loco" in text
    assert "turn_flag" in text
    assert "'int'" in text
    assert driver.calls == []


def test_a_string_turn_flag_is_refused_with_a_message_naming_the_type() -> None:
    """A string ``turn_flag`` is refused rather than routed through ``bool()``.

    A model may synthesize the flag as a truthy string ``"yes"``
    or ``"true"``; Python's ``bool("false")`` is ``True`` (any
    non-empty string is truthy) so a caller writing the word
    would reach the SDK's wave-and-turn dispatcher.  Refusing the
    shape here keeps the refusal decidable at the tool surface
    rather than SDK-version dependent, and matches the envelope
    module's own ``turn_flag must be bool`` refusal shape.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_wave_hand_loco(driver=driver, turn_flag="true")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_wave_hand_loco" in text
    assert "turn_flag" in text
    assert "'str'" in text
    assert driver.calls == []


def test_a_float_turn_flag_is_refused_with_a_message_naming_the_type() -> None:
    """A ``float`` ``turn_flag`` is refused rather than routed through ``bool()``.

    ``bool(0.1)`` is ``True`` and ``bool(0.0)`` is ``False``; a
    caller passing a float is not naming a turn-flag variant on
    purpose (there is no continuous middle ground the SDK admits
    - the two ``SetTaskId`` compositions are the two data
    points).  Refusing the shape here mirrors the envelope's own
    refusal so both paths render the same shape a caller can
    grep for.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_wave_hand_loco(driver=driver, turn_flag=1.0)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_wave_hand_loco" in text
    assert "turn_flag" in text
    assert "'float'" in text
    assert driver.calls == []
