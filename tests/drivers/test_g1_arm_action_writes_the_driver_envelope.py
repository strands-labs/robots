"""``g1_arm_action`` returns exactly what ``G1Driver.arm_action`` gives it.

``g1_arm_action`` is the write-side companion to the neon bundle's
``g1_arm_action`` verb: where the neon verb wraps
``G1ArmActionClient.ExecuteAction(id)`` under a single-writer lock
and holds through a ``time.sleep(hold_seconds)``, this one hands the
gesture request to the driver's own write path and reads back the
envelope the driver produced.  The driver's ``arm_action`` method is
not yet plumbed today (refs strands-labs/robots#358 for the SDK-
facing gate work the write belongs on), so the
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal` grader
refuses a handle without an ``arm_action`` accessor with a message
naming the verb, the ``driver`` parameter and the accessor; these
tests fix the shape the verb passes through for each of the driver's
current and future outcomes (a driver-side refusal, a future success
envelope, and the verb's own ``driver`` and parameter refusals).

The refusal-string tests do not restate the driver's exact prose -
that would trap the verb to the driver's refusal wording of one
gesture, which is exactly what the driver's own release notes say
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

from strands_robots.tools.g1.g1_arm_action import g1_arm_action


class _StubG1Driver:
    """A driver double whose ``arm_action`` returns a fixed envelope.

    ``g1_arm_action`` calls ``driver.arm_action(action, action_id)``
    and returns the envelope verbatim.  This double sits under the
    same interface without pulling the real driver's imports (the
    real class reaches CycloneDDS at construction time in some
    paths), so a test can hand a wired-shape envelope to the verb
    without a bus.  ``calls`` records the arguments each call
    received so a test can pin "the verb hands the driver exactly
    the strings the caller passed" without asking the driver method
    itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls: list[tuple[str, int | None]] = []

    def arm_action(self, action: str, action_id: int | None) -> dict[str, Any]:
        self.calls.append((action, action_id))
        return self._envelope


def _call(
    driver: Any,
    action: str = "",
    action_id: int | None = None,
) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim.  This helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return g1_arm_action(driver=driver, action=action, action_id=action_id)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a submodule
    at import time would break every headless CI runner and Thor
    before an office bring-up.  The driver enforces the same rule
    against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the
    only path that loads the SDK); this cell holds the arm-action
    verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_arm_action")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_arm_action imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that "
        "the SDK loads only inside function bodies "
        "(refs strands-labs/robots#358)."
    )


def test_a_driver_side_refusal_surfaces_verbatim() -> None:
    """The driver's refusal envelope round-trips through the verb unchanged.

    ``G1Driver.arm_action`` will (once landed) refuse an SDK-side
    raise with a named error envelope, and today's driver refuses
    every call because the method is not yet plumbed.  Both shapes
    are ``{"status": "error", "content": [{"text": ...}]}`` and the
    verb passes either through; a wording drift on the driver side
    moves this verb with it (refs strands-labs/robots#2874).
    """
    refusal_text = "arm_action: ExecuteAction(17) raised: RPC_CLIENT_API_TIMEOUT"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, action="clap")

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_future_success_envelope_round_trips_verbatim() -> None:
    """Once ``arm_action`` lands, the success envelope surfaces verbatim.

    When the driver's method is plumbed, ``G1Driver.arm_action``
    will surface the SDK's ``rc`` inside a ``{"status": "success",
    "content": [{"json": {"action": ..., "action_id": ..., "rc": 0,
    "message": ...}}]}`` envelope.  The verb does not reshape it -
    a future field the driver adds reaches a caller the moment the
    driver writes it, and this cell holds that pass-through
    explicit so the verb is ready the moment the write path lands.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "action": "clap",
                    "action_id": 17,
                    "rc": 0,
                    "message": "Arm action rc=0 (OK)",
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, action="clap")

    assert result is envelope or result == envelope
    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["rc"] == 0
    assert payload["action"] == "clap"


def test_the_verb_passes_action_and_id_verbatim() -> None:
    """The verb hands ``driver.arm_action`` exactly what it received.

    The driver's method owns name-to-id resolution and any refusal
    for an unknown gesture; this verb does not pre-flight against
    the map.  Fixing "verb passes the strings through" is what
    keeps a future rename on the driver side (a new gesture, a
    remapped id) from silently dropping on the tool layer.
    """
    envelope = {
        "status": "success",
        "content": [{"json": {"rc": 0}}],
    }
    driver = _StubG1Driver(envelope=envelope)

    _call(driver, action="heart", action_id=None)
    _call(driver, action="", action_id=20)
    _call(driver, action="clap", action_id=17)

    assert driver.calls == [("heart", None), ("", 20), ("clap", 17)]


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
    result = g1_arm_action(driver=None, action="clap")

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_arm_action" in text
    assert "driver" in text


def test_a_string_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot-name string is refused before the accessor is called.

    A caller who reached for a *name* instead of a live handle (a
    common shape when the orchestrator forgot to bind the driver)
    is refused with the same envelope shape, naming the type it
    received so the caller can trace the mistake back to the bind
    site.  The shared ``live_handle_refusal`` guard produces the
    refusal; this cell fixes that a non-``G1Driver`` is refused
    before the accessor path.
    """
    result = g1_arm_action(driver="g1-01", action="clap")

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_arm_action" in text
    assert "str" in text


def test_a_handle_without_arm_action_is_refused_naming_the_accessor() -> None:
    """A live object without ``arm_action`` refuses naming the accessor.

    Today's driver does not plumb ``arm_action``, so every call the
    verb receives lands on the accessor-missing branch of
    ``live_handle_refusal``.  The refusal names the verb, the
    ``driver`` parameter, and the accessor the verb read for; a
    caller upgrading past this PR can grep the accessor name to
    know when the driver's write path is what wire the verb calls
    (refs strands-labs/robots#358).
    """

    class _NoArmAction:
        """Live object without an ``arm_action`` method."""

    result = g1_arm_action(driver=_NoArmAction(), action="clap")

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_arm_action" in text
    assert "arm_action" in text


def test_a_call_with_neither_action_nor_id_is_refused() -> None:
    """The verb refuses a call that names no gesture and no id.

    The neon bundle's ``g1_arm_action`` verb accepted a default
    empty ``action=""`` and a default ``action_id=-1``, and would
    reach the SDK's map with the empty string and either refuse
    there or run the wrong gesture depending on the SDK version.
    This port refuses the shape at the tool layer so a caller who
    reached the verb with neither parameter set sees a message
    naming both parameters and the remedy (call
    ``g1_list_arm_actions`` to see the map), rather than a
    driver-side refusal that names only the id it received.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {"rc": 0}}]})
    result = g1_arm_action(driver=driver)

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_arm_action" in text
    assert "action" in text
    assert "action_id" in text
    assert "g1_list_arm_actions" in text
    # And the driver is not called on the refusal branch: fixing
    # the ordering here is what keeps a caller from writing an
    # empty gesture on the SDK when the tool layer refused it.
    assert driver.calls == []


def test_action_id_alone_is_admitted_without_calling_action_first() -> None:
    """Passing ``action_id`` alone reaches the driver with the id.

    The neon bundle's ``g1_arm_action`` verb documented that
    ``action_id`` wins over ``action`` when both are passed; the
    port keeps that precedence so a caller pinning by id ahead of
    a driver-side name rename is not caught by the string being
    wrong.  Fixing "id alone is admitted" here is what keeps the
    caller who wants the numeric handle from being forced to also
    look up the name.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)

    result = _call(driver, action="", action_id=17)

    assert result["status"] == "success"
    assert driver.calls == [("", 17)]


def test_action_alone_is_admitted_without_action_id() -> None:
    """Passing ``action`` alone reaches the driver with the name.

    The mirror shape of the id-only cell: a caller pinning by
    gesture name (the common case) reaches the driver without
    having to also supply the id, and the driver's own name-to-id
    map resolves the name.  Fixing both shapes here is what keeps
    the verb's two acceptance modes symmetric in the docstring and
    on the wire.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)

    result = _call(driver, action="clap", action_id=None)

    assert result["status"] == "success"
    assert driver.calls == [("clap", None)]
