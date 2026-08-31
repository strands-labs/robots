"""``g1_send_action`` returns exactly what ``G1Driver.send_action`` gives it.

``g1_send_action`` is the one-frame companion to ``g1_run_policy``:
``run_policy`` starts the 500 Hz control loop, and this verb is the
single-write shape - one joint-name-keyed dict, one ``LowCmd_`` frame
on ``rt/lowcmd``.  The driver's method
(:meth:`G1Driver.send_action`) already gates through
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` with
scope ``"arm"``, so a caller whose FSM or battery is outside the
admission set gets the driver's own refusal string; the tests here
fix the envelope shape the verb reshapes each of the driver's
outcomes into (a success write, a gate refusal, a publisher-not-
initialised refusal, an action-dict validation refusal, an SDK-
missing refusal, a publish-error refusal).

The success envelope's field names are read here off the driver's
own writer (:meth:`G1Driver.send_action`) rather than being restated:
a rename on the driver side moves both the write path and this
verb together.  What the tests do restate is the SDK-load-hygiene
contract every file under :mod:`strands_robots.tools.g1` carries:
importing the module must not pull any ``unitree_sdk2py``
submodule.

The refusal-string tests do not restate the driver's exact prose -
that would trap the verb to the driver's refusal wording of one
release, which is exactly what the driver's own release notes say
verbatim quotes should not do (refs strands-labs/robots#2874).  They
grade the shape (``status="error"``, an envelope-shaped
``content[0]["text"]``, the verb's own three refusal invariants when
the verb produced them) and pass the driver's own text through
unchanged when the driver produced it.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_send_action import g1_send_action


class _StubG1Driver:
    """A driver double whose ``send_action`` returns a fixed envelope.

    ``g1_send_action`` calls ``driver.send_action(action)`` and returns
    the envelope verbatim.  This double sits under the same interface
    without pulling the real driver's imports (the real class reaches
    CycloneDDS at construction time in some paths), so a test can
    hand a wired-shape envelope to the verb without a bus.  ``calls``
    records ``(action,)`` per invocation so a test can pin "the verb
    writes the driver exactly once" and "the verb passes the action
    dict through unchanged" without asking the driver method itself
    to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls: list[dict[str, Any]] = []

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        del robot_name
        self.calls.append(action)
        return self._envelope


def _call(driver: Any, action: dict[str, Any]) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim.  This helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return g1_send_action(driver=driver, action=action)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a submodule
    at import time would break every headless CI runner and Thor
    before an office bring-up.  The driver enforces the same rule
    against itself (:func:`~strands_robots.tools.g1._g1_common.ensure_dds`
    is the only path that loads the SDK); this cell holds the
    send-action verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_send_action")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_send_action imports pulled SDK submodules: "
        f"{leaked}. The rule for this package is that the SDK loads only "
        "inside function bodies (refs strands-labs/robots#358)."
    )


def test_a_success_envelope_round_trips_verbatim() -> None:
    """The driver's success envelope surfaces verbatim.

    :meth:`G1Driver.send_action` returns
    ``{"status": "success", "content": [{"json": {"topic": "rt/lowcmd",
    "joints": [...], "fsm_id": ..., "mode_machine": ...}}]}`` on a
    write that admitted the arm-SDK gate and reached the wire.  The
    verb does not reshape the envelope - a future field the driver
    adds on this path reaches a caller the moment the driver writes
    it, and this cell holds that pass-through explicit.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "topic": "rt/lowcmd",
                    "joints": ["left_shoulder_pitch_joint"],
                    "fsm_id": 500,
                    "mode_machine": 1,
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, {"left_shoulder_pitch_joint": 0.0})

    assert result is envelope or result == envelope
    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["topic"] == "rt/lowcmd"
    assert payload["joints"] == ["left_shoulder_pitch_joint"]
    assert payload["fsm_id"] == 500
    assert payload["mode_machine"] == 1


def test_a_gate_refusal_from_the_driver_surfaces_verbatim() -> None:
    """The driver's FSM/battery refusal reaches a caller unchanged.

    :meth:`G1Driver._check_motion_gates` is the one gate; when it
    refuses (FSM outside the arm-SDK admission set, battery under
    the driver's floor, motion switcher unavailable), the driver's
    ``send_action`` returns the refusal envelope its own path
    produced.  The verb passes that envelope through - a second
    gate call here would double the read against the same cache
    and fork the source of truth for a rule the driver's own path
    already enforces (refs strands-labs/robots#2916).
    """
    refusal_text = "send_action refused: FSM id 4 outside the arm-SDK admission set {500, 501, 801}"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, {"left_shoulder_pitch_joint": 0.0})

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_missing_driver_is_refused_with_a_message_naming_the_parameter() -> None:
    """A ``None`` driver is refused before the accessor is called.

    ``driver`` is a live Python object typed :class:`~typing.Any`, so
    the tool schema carries no signal that a caller cannot synthesize
    it.  A model that leaves the parameter out reaches the verb with
    ``None``, and the verb owes an envelope-shaped refusal instead of
    an exception the ``@tool`` wrapper cannot format.  The shared
    :func:`live_handle_refusal` guard produces it; this cell fixes
    that the guard is called before the accessor path.
    """
    result = g1_send_action(driver=None, action={"left_shoulder_pitch_joint": 0.0})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_send_action" in text
    assert "driver" in text


def test_a_wrong_shape_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot *name* (string) is refused before the accessor call.

    A model that synthesizes the argument as a robot name reaches the
    verb with a ``str``.  The verb owes an envelope-shaped refusal
    that names the type it received and the remedy - the four
    invariants every ``@tool`` handler in this package holds - and
    this cell fixes the shape.
    """
    result = g1_send_action(driver="unitree_g1", action={"left_shoulder_pitch_joint": 0.0})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_send_action" in text
    assert "'str'" in text


def test_a_missing_action_is_refused_naming_the_parameter_and_the_remedy() -> None:
    """A ``None`` action is refused before the driver is called.

    The tool schema does describe ``action`` (a dict), so a model can
    synthesize the wrong shape as easily as the right one.  The
    verb's ``action`` refusals cover the three shapes the driver's
    own ``_build_lowcmd_from_action`` would surface as an inner
    refusal naming a joint (a missing ``q``), which is a shape a
    caller reading ``action`` alone cannot map back to the parameter.
    Naming ``action`` here keeps the invariants; this cell fixes
    that the driver's ``send_action`` is not called on the ``None``
    shape.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_send_action(driver=driver, action=None)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_send_action" in text
    assert "action" in text
    # The driver was not called - the guard fires before the accessor path.
    assert driver.calls == []


def test_a_non_dict_action_is_refused_with_a_message_naming_the_type() -> None:
    """A list ``action`` is refused with a message naming ``list``.

    A model that reads "action" and reaches for a list is refused
    with the same four invariants: envelope, verb name, parameter
    name, and the type it received.  This cell holds the type name
    in the refusal string.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_send_action(driver=driver, action=[0.0, 0.0])  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_send_action" in text
    assert "'list'" in text
    assert driver.calls == []


def test_an_empty_action_is_refused_before_the_driver_is_called() -> None:
    """An empty dict ``action`` is refused - a no-op wire frame is refused.

    A wire frame that names no joint is a no-op that would still
    consume the arm-SDK gate's admission read; the driver would
    admit the empty frame and publish an empty ``LowCmd_``, wasting
    both the gate call and a bus write.  The verb refuses this shape
    on this side with a message naming the parameter and the remedy.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_send_action(driver=driver, action={})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_send_action" in text
    assert "empty" in text
    assert driver.calls == []


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.send_action``.

    ``send_action`` is a write (one ``LowCmd_`` frame on the wire) so
    a double-call from a wrapper that retried inside the verb would
    double the wire load and the arm-SDK gate's admission read.  This
    cell pins the verb to a single driver write per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {"topic": "rt/lowcmd"}}]})
    _call(driver, {"left_shoulder_pitch_joint": 0.0})
    assert len(driver.calls) == 1


def test_the_verb_passes_the_action_dict_through_unchanged() -> None:
    """The ``action`` dict the driver receives is the one the caller passed.

    A wrapper that mutated or filtered ``action`` between the verb
    and the driver would fork the mapping between joint-name keys
    the caller wrote and the wire frame the driver builds.  The
    driver's :func:`~strands_robots.drivers.g1._build_lowcmd_from_action`
    is the one builder; this verb is a duck-typed pass-through so
    every key the caller wrote reaches the builder, and every key
    the builder refuses (a missing ``q`` inside a per-joint dict)
    is refused with the joint name the caller wrote.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {"topic": "rt/lowcmd"}}]})
    action = {
        "left_shoulder_pitch_joint": 0.1,
        "left_elbow_joint": {"q": 0.2, "kp": 40.0, "kd": 0.5},
    }
    _call(driver, action)
    assert driver.calls == [action]
    # The dict is not copied - the verb hands the reference through, so
    # the driver sees exactly the object the caller wrote and a builder
    # that reads keys after the call reads the caller's keys, not a
    # snapshot the verb took.
    assert driver.calls[0] is action


def test_the_envelope_status_is_reported_verbatim() -> None:
    """A non-success envelope from the driver surfaces on the returned dict.

    :meth:`G1Driver.send_action` returns ``status="error"`` on every
    refusal path (gate, publisher not initialised, action-dict
    validation, SDK missing, publish error), and the verb must
    surface that verbatim rather than flattening it to a success.
    This cell fixes the flow from the envelope's ``status`` to the
    returned dict's ``status`` for one representative refusal
    (publish error), so a caller reading only ``status`` cannot
    read "success" while the payload carries the driver's error
    text.
    """
    refusal_text = "send_action refused: publish returned rc=-1"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, {"left_shoulder_pitch_joint": 0.0})
    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text
