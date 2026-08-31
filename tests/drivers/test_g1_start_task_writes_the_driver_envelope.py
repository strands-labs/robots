"""``g1_start_task`` returns exactly what ``G1Driver.start_task`` gives it.

``g1_start_task`` is the provider-registry entry point to the driver's
500 Hz control loop, sitting alongside ``g1_run_policy`` (which starts
the loop against an already-built policy) and ``g1_send_action``
(which writes one arm-SDK frame).  The provider registry is not yet
plumbed to :class:`~strands_robots.drivers.g1.G1Driver`, so the
driver's method refuses with a fixed message after the FSM/battery
gate admits; the tests here fix the envelope shape the verb passes
through for each of the driver's current and future outcomes (the
today-shape refusal, a gate refusal, a future success envelope
matching :meth:`run_policy`, and the verb's own ``driver`` refusals).

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

from strands_robots.tools.g1.g1_start_task import g1_start_task


class _StubG1Driver:
    """A driver double whose ``start_task`` returns a fixed envelope.

    ``g1_start_task`` calls
    ``driver.start_task(instruction, policy_port=..., policy_host=...,
    policy_provider=..., duration=...)`` and returns the envelope
    verbatim.  This double sits under the same interface without
    pulling the real driver's imports (the real class reaches
    CycloneDDS at construction time in some paths), so a test can
    hand a wired-shape envelope to the verb without a bus.  ``calls``
    records ``(instruction, policy_port, policy_host,
    policy_provider, duration)`` per invocation so a test can pin
    "the verb writes the driver exactly once" and "the verb passes
    the five arguments through unchanged" without asking the driver
    method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls: list[tuple[str, int | None, str, str, float]] = []

    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        **_policy_kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((instruction, policy_port, policy_host, policy_provider, duration))
        return self._envelope


def _call(
    driver: Any,
    *,
    instruction: str = "",
    policy_port: int | None = None,
    policy_host: str = "localhost",
    policy_provider: str = "groot",
    duration: float = 30.0,
) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim.  This helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return g1_start_task(
        driver=driver,
        instruction=instruction,
        policy_port=policy_port,
        policy_host=policy_host,
        policy_provider=policy_provider,
        duration=duration,
    )


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a submodule
    at import time would break every headless CI runner and Thor
    before an office bring-up.  The driver enforces the same rule
    against itself (:func:`~strands_robots.tools.g1._g1_common.ensure_dds`
    is the only path that loads the SDK); this cell holds the
    start-task verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_start_task")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_start_task imports pulled SDK submodules: "
        f"{leaked}. The rule for this package is that the SDK loads only "
        "inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_registry_not_wired_refusal_from_the_driver_surfaces_verbatim() -> None:
    """Today's driver refusal round-trips through the verb unchanged.

    :meth:`G1Driver.start_task` runs the FSM/battery gate and then
    refuses with a fixed ``start_task: provider registry not wired
    yet; use run_policy(policy_object=...) to drive the control loop
    today`` message because the registry in
    :mod:`strands_robots.policies` is not yet plumbed to this
    driver.  The verb passes that envelope through; a wording drift
    on the driver side moves this verb with it (refs
    strands-labs/robots#2874).
    """
    refusal_text = (
        "start_task: provider registry not wired yet; use run_policy(policy_object=...) to drive the control loop today"
    )
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_gate_refusal_from_the_driver_surfaces_verbatim() -> None:
    """The driver's FSM/battery refusal reaches a caller unchanged.

    :meth:`G1Driver._check_motion_gates` is the one gate; when it
    refuses (FSM outside the arm-SDK admission set, battery under
    the driver's floor, motion switcher unavailable), the driver's
    ``start_task`` returns the refusal envelope its own path
    produced *before* the registry-not-wired refusal fires.  The
    verb passes that envelope through - a second gate call here
    would double the read against the same cache and fork the source
    of truth for a rule the driver's own path already enforces
    (refs strands-labs/robots#2916).
    """
    refusal_text = "start_task refused: FSM id 4 outside the motion admission set"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_future_success_envelope_round_trips_verbatim() -> None:
    """Once the registry lands, the loop-start envelope surfaces verbatim.

    When the provider registry in
    :mod:`strands_robots.policies` is plumbed to the driver,
    :meth:`~strands_robots.drivers.g1.G1Driver.start_task` forwards
    to the same loop path :meth:`run_policy` uses today and returns
    that method's start envelope.  The verb does not reshape it -
    a future field the driver adds reaches a caller the moment the
    driver writes it, and this cell holds that pass-through
    explicit so the verb is ready the moment the registry lands.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "tool_name": "g1_start_task",
                    "task_running": True,
                    "duration": 5.0,
                    "n_steps": None,
                    "hz": 500,
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, duration=5.0)

    assert result is envelope or result == envelope
    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["task_running"] is True
    assert payload["duration"] == 5.0
    assert payload["hz"] == 500


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
    result = g1_start_task(driver=None)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_start_task" in text
    assert "driver" in text


def test_a_wrong_shape_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot *name* (string) is refused before the accessor call.

    A model that synthesizes the argument as a robot name reaches the
    verb with a ``str``.  The verb owes an envelope-shaped refusal
    that names the type it received and the remedy - the four
    invariants every ``@tool`` handler in this package holds - and
    this cell fixes the shape.
    """
    result = g1_start_task(driver="unitree_g1")
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_start_task" in text
    assert "'str'" in text


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.start_task``.

    ``start_task`` (once the registry lands) starts a 500 Hz control
    loop on a dedicated thread, so a double-call from a wrapper that
    retried inside the verb would spawn two loops against the same
    admission lock - and the second would refuse with "a task is
    already running" once the loop path is reachable.  This cell
    pins the verb to a single driver call per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "error", "content": [{"text": "registry-not-wired"}]})
    _call(driver)
    assert len(driver.calls) == 1


def test_the_verb_passes_the_arguments_through_unchanged() -> None:
    """The five arguments the driver receives are the ones the caller passed.

    The verb's contract is a pass-through: it does not translate
    argument names, does not synthesize defaults the driver did not
    ask for, and does not rearrange keyword order.  This cell fixes
    that ``instruction``, ``policy_port``, ``policy_host``,
    ``policy_provider`` and ``duration`` reach the driver method
    verbatim - a rename or reorder on either side is a driver-level
    contract change, not a silent verb-side translation.
    """
    envelope = {"status": "error", "content": [{"text": "registry-not-wired"}]}
    driver = _StubG1Driver(envelope=envelope)
    _call(
        driver,
        instruction="pick up the red cube",
        policy_port=8082,
        policy_host="10.10.4.42",
        policy_provider="groot",
        duration=12.5,
    )
    assert driver.calls == [
        ("pick up the red cube", 8082, "10.10.4.42", "groot", 12.5),
    ]


def test_a_default_call_passes_the_signature_defaults_to_the_driver() -> None:
    """A caller omitting the knobs still reaches the driver with the defaults.

    The verb's parameter defaults (``instruction=""``,
    ``policy_port=None``, ``policy_host="localhost"``,
    ``policy_provider="groot"``, ``duration=30.0``) match the
    driver's method signature.  A caller that names only ``driver``
    reaches the driver with the same defaults the driver would have
    filled in on its own; this cell holds the parity so a driver-side
    default change surfaces on the verb without a silent divergence.
    """
    envelope = {"status": "error", "content": [{"text": "registry-not-wired"}]}
    driver = _StubG1Driver(envelope=envelope)
    result = g1_start_task(driver=driver)

    assert result["status"] == "error"
    assert driver.calls == [("", None, "localhost", "groot", 30.0)]


def test_the_wrong_shape_driver_is_not_called() -> None:
    """A wrong-shape driver is refused before the accessor path.

    The shared :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard is called first; a ``str`` handle is refused with an
    envelope and the driver's method is never reached.  This cell
    holds the ordering by grading a handle that would raise if
    called (``str`` has no ``start_task`` attribute) and observing
    that the refusal envelope has the four invariants without an
    exception in flight.
    """
    result = g1_start_task(driver=42)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_start_task" in text
    assert "'int'" in text
