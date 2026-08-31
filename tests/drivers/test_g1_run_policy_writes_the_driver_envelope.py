"""``g1_run_policy`` returns exactly what ``G1Driver.run_policy`` gives it.

``g1_run_policy`` is the 500-Hz-loop companion to ``g1_send_action``:
``send_action`` writes one frame, and this verb starts the control
loop that writes many frames on a dedicated thread.  The driver's
method (:meth:`G1Driver.run_policy`) already gates through
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` with
scope ``"motion"`` on every step, so a caller whose FSM leaves the
arm-SDK admission set mid-rollout gets the driver's own zero-torque
frame and an ``exit_reason="gate"`` snapshot; the tests here fix the
envelope shape the verb passes through for each of the driver's
outcomes (a started rollout, a ``duration`` / ``n_steps`` validation
refusal, a gate refusal, a ``policy_object`` refused by the driver,
a "task already running" refusal).

The success envelope's field names are read here off the driver's
own :meth:`~strands_robots.drivers.g1.G1Driver.run_policy` writer
rather than being restated: a rename on the driver side moves both
the start path and this verb together.  What the tests do restate
is the SDK-load-hygiene contract every file under
:mod:`strands_robots.tools.g1` carries: importing the module must
not pull any ``unitree_sdk2py`` submodule.

The refusal-string tests do not restate the driver's exact prose -
that would trap the verb to the driver's refusal wording of one
release, which is exactly what the driver's own release notes say
verbatim quotes should not do (refs strands-labs/robots#2874).  They
grade the shape (``status="error"``, an envelope-shaped
``content[0]["text"]``, the verb's own four refusal invariants when
the verb produced them) and pass the driver's own text through
unchanged when the driver produced it.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_run_policy import g1_run_policy


class _StubG1Driver:
    """A driver double whose ``run_policy`` returns a fixed envelope.

    ``g1_run_policy`` calls
    ``driver.run_policy(policy_object, instruction=..., duration=...,
    n_steps=...)`` and returns the envelope verbatim.  This double
    sits under the same interface without pulling the real driver's
    imports (the real class reaches CycloneDDS at construction time
    in some paths), so a test can hand a wired-shape envelope to the
    verb without a bus.  ``calls`` records
    ``(policy_object, instruction, duration, n_steps)`` per
    invocation so a test can pin "the verb writes the driver exactly
    once" and "the verb passes the four arguments through unchanged"
    without asking the driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls: list[tuple[Any, str, float, int | None]] = []

    def run_policy(
        self,
        policy_object: Any,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append((policy_object, instruction, duration, n_steps))
        return self._envelope


class _StubPolicy:
    """A policy double with a ``.step`` method the verb's admission accepts.

    The verb accepts a ``.step()``-exposing object *or* a bare callable;
    this class picks the first shape, which is the one every
    :class:`~strands_robots.policies.Policy` subclass exposes in this
    package.  ``step`` returns a fixed joint-name-keyed dict so a caller
    who reached the driver's loop with this policy would get a stable
    action.  The verb's admission checks callable-ness of ``step`` on
    this side (before the driver is called); the tests hand this to the
    verb to fix the admission's success path.
    """

    def step(self, obs: Any) -> dict[str, Any]:
        del obs
        return {"left_shoulder_pitch_joint": 0.0}


def _call(
    driver: Any,
    policy_object: Any,
    *,
    instruction: str = "",
    duration: float = 30.0,
    n_steps: int | None = None,
) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim.  This helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return g1_run_policy(
        driver=driver,
        policy_object=policy_object,
        instruction=instruction,
        duration=duration,
        n_steps=n_steps,
    )


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a submodule
    at import time would break every headless CI runner and Thor
    before an office bring-up.  The driver enforces the same rule
    against itself (:func:`~strands_robots.tools.g1._g1_common.ensure_dds`
    is the only path that loads the SDK); this cell holds the
    run-policy verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_run_policy")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_run_policy imports pulled SDK submodules: "
        f"{leaked}. The rule for this package is that the SDK loads only "
        "inside function bodies (refs strands-labs/robots#358)."
    )


def test_a_success_envelope_round_trips_verbatim() -> None:
    """The driver's success envelope surfaces verbatim.

    :meth:`G1Driver.run_policy` returns
    ``{"status": "success", "content": [{"json": {"tool_name": ...,
    "task_running": True, "duration": ..., "n_steps": ..., "hz":
    500}}]}`` on a start that admitted the arm-SDK gate and spawned
    the control loop.  The verb does not reshape the envelope - a
    future field the driver adds on this path reaches a caller the
    moment the driver writes it, and this cell holds that
    pass-through explicit.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "tool_name": "g1_run_policy",
                    "task_running": True,
                    "duration": 5.0,
                    "n_steps": 100,
                    "hz": 500,
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, _StubPolicy(), duration=5.0, n_steps=100)

    assert result is envelope or result == envelope
    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["tool_name"] == "g1_run_policy"
    assert payload["task_running"] is True
    assert payload["duration"] == 5.0
    assert payload["n_steps"] == 100
    assert payload["hz"] == 500


def test_a_gate_refusal_from_the_driver_surfaces_verbatim() -> None:
    """The driver's FSM/battery refusal reaches a caller unchanged.

    :meth:`G1Driver._check_motion_gates` is the one gate; when it
    refuses (FSM outside the arm-SDK admission set, battery under
    the driver's floor, motion switcher unavailable), the driver's
    ``run_policy`` returns the refusal envelope its own path
    produced.  The verb passes that envelope through - a second
    gate call here would double the read against the same cache
    and fork the source of truth for a rule the driver's own path
    already enforces (refs strands-labs/robots#2916).
    """
    refusal_text = "run_policy refused: FSM id 4 outside the motion admission set"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, _StubPolicy())

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_task_already_running_refusal_surfaces_verbatim() -> None:
    """The driver's "task already running" refusal reaches a caller unchanged.

    :meth:`G1Driver.run_policy` holds the ``_task_admission`` lock
    across the ``is_running`` check, the loop reference assignment
    and the loop's ``start()``, so a second ``run_policy`` call
    against a running loop refuses with a fixed message.  The verb
    surfaces that message verbatim; a second admission check here
    would double the lock and re-fork the refusal wording.
    """
    refusal_text = "run_policy: a task is already running; call stop_task first"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, _StubPolicy())

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
    result = g1_run_policy(driver=None, policy_object=_StubPolicy())
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_run_policy" in text
    assert "driver" in text


def test_a_wrong_shape_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot *name* (string) is refused before the accessor call.

    A model that synthesizes the argument as a robot name reaches the
    verb with a ``str``.  The verb owes an envelope-shaped refusal
    that names the type it received and the remedy - the four
    invariants every ``@tool`` handler in this package holds - and
    this cell fixes the shape.
    """
    result = g1_run_policy(driver="unitree_g1", policy_object=_StubPolicy())
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_run_policy" in text
    assert "'str'" in text


def test_a_missing_policy_object_is_refused_naming_the_parameter_and_the_remedy() -> None:
    """A ``None`` policy is refused before the driver is called.

    The tool schema does describe ``policy_object`` (a callable or a
    ``.step()``-exposing object), so a model can synthesize the
    wrong shape as easily as the right one.  The verb refuses the
    ``None`` shape here rather than surfacing the driver's own
    refusal through a call site the caller cannot map back to the
    parameter.  Naming ``policy_object`` here keeps the invariants;
    this cell fixes that the driver's ``run_policy`` is not called
    on the ``None`` shape.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_run_policy(driver=driver, policy_object=None)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_run_policy" in text
    assert "policy_object" in text
    # The driver was not called - the guard fires before the accessor path.
    assert driver.calls == []


def test_a_non_callable_policy_object_is_refused_with_a_message_naming_the_type() -> None:
    """A dict ``policy_object`` is refused with a message naming ``dict``.

    A model that reads "policy_object" and reaches for a dict is
    refused with the same four invariants: envelope, verb name,
    parameter name, and the type it received.  The dict has neither
    ``__call__`` nor a ``.step`` attribute, so both admission checks
    fail; this cell holds the type name in the refusal string.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_run_policy(driver=driver, policy_object={"policy": "foo"})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_run_policy" in text
    assert "'dict'" in text
    assert driver.calls == []


def test_a_bare_callable_policy_is_admitted() -> None:
    """A bare callable (no ``.step``) is admitted by the verb.

    The driver's own admission accepts a callable *or* a
    ``.step()``-exposing object, so the verb's admission does the
    same or it would refuse a shape the driver accepts.  This cell
    holds the verb to the driver's admission set.
    """

    def bare_policy(obs: Any) -> dict[str, Any]:
        del obs
        return {"left_shoulder_pitch_joint": 0.0}

    envelope = {"status": "success", "content": [{"json": {"task_running": True}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, bare_policy)

    assert result["status"] == "success"
    # The driver was called - the verb admitted the bare callable.
    assert len(driver.calls) == 1
    assert driver.calls[0][0] is bare_policy


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.run_policy``.

    ``run_policy`` starts a 500 Hz control loop on a dedicated
    thread, so a double-call from a wrapper that retried inside the
    verb would spawn two loops against the same admission lock -
    and the second would refuse with "a task is already running".
    This cell pins the verb to a single driver start per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {"task_running": True}}]})
    _call(driver, _StubPolicy())
    assert len(driver.calls) == 1


def test_the_verb_passes_the_arguments_through_unchanged() -> None:
    """The four arguments the driver receives are the ones the caller passed.

    A wrapper that mutated or defaulted any of ``policy_object``,
    ``instruction``, ``duration``, or ``n_steps`` between the verb
    and the driver would fork the shape the caller wrote from the
    shape the driver observed.  The driver's own
    :meth:`~strands_robots.drivers.g1.G1Driver.run_policy` is the one
    validator for ``duration`` and ``n_steps``; this verb is a
    duck-typed pass-through so every value the caller wrote reaches
    the driver, and every value the driver refuses reaches the caller
    verbatim.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    policy = _StubPolicy()
    _call(driver, policy, instruction="pick up the cube", duration=12.5, n_steps=250)

    assert driver.calls == [(policy, "pick up the cube", 12.5, 250)]
    # The policy object is not copied - the verb hands the reference
    # through, so the driver sees exactly the object the caller wrote
    # and the loop's ``policy_object.step`` call reaches the caller's
    # object, not a snapshot the verb took.
    assert driver.calls[0][0] is policy


def test_the_envelope_status_is_reported_verbatim() -> None:
    """A non-success envelope from the driver surfaces on the returned dict.

    :meth:`G1Driver.run_policy` returns ``status="error"`` on every
    refusal path (``duration`` / ``n_steps`` validation, gate,
    non-callable ``policy_object``, task already running), and the
    verb must surface that verbatim rather than flattening it to a
    success.  This cell fixes the flow from the envelope's
    ``status`` to the returned dict's ``status`` for one
    representative refusal (duration validation), so a caller
    reading only ``status`` cannot read "success" while the payload
    carries the driver's error text.
    """
    refusal_text = "run_policy: duration must be a positive finite number"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, _StubPolicy())
    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_the_default_duration_reaches_the_driver_when_omitted() -> None:
    """A caller who omits ``duration`` reaches the driver with 30.0.

    The verb's signature default (``duration: float = 30.0``) matches
    the driver's own default; a caller who reaches the verb with no
    ``duration`` sees a 30-second rollout the driver would have run
    anyway.  This cell holds the two defaults aligned - a drift here
    would silently cap a rollout at a different wall-clock than the
    driver documents.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    g1_run_policy(driver=driver, policy_object=_StubPolicy())

    assert len(driver.calls) == 1
    _, _, duration, n_steps = driver.calls[0]
    assert duration == 30.0
    assert n_steps is None


def test_the_default_n_steps_of_none_reaches_the_driver_when_omitted() -> None:
    """A caller who omits ``n_steps`` reaches the driver with ``None``.

    The driver's ``run_policy`` uses ``None`` to mean "no step cap"
    (only the ``duration`` deadline applies); a wrapper that
    defaulted ``n_steps`` to any integer here would silently cap
    rollouts at a step count the caller never named.  This cell
    holds the verb's default aligned with the driver's semantics.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    g1_run_policy(driver=driver, policy_object=_StubPolicy(), duration=5.0)

    assert len(driver.calls) == 1
    _, _, duration, n_steps = driver.calls[0]
    assert duration == 5.0
    assert n_steps is None


def test_the_verb_accepts_the_zero_arg_call_pattern_the_universal_discovery_test_uses() -> None:
    """A single-arg call reaches the verb's ``policy_object`` refusal.

    The universal auto-discovery test in this package calls every
    verb with ``verb(driver=driver)`` alone, so every parameter after
    ``driver`` must have a default value; the verb then reaches its
    own admission for the omitted parameter and returns the refusal
    envelope naming that parameter.  This cell fixes the
    single-arg-call reach for this verb: ``policy_object`` defaults
    to ``None``, the admission refuses ``None`` before the driver is
    called, and the returned envelope names ``policy_object``.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_run_policy(driver=driver)  # type: ignore[call-arg]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_run_policy" in text
    assert "policy_object" in text
    assert driver.calls == []
