"""``g1_balance_stand`` returns exactly what ``G1Driver.balance_stand`` gives it.

``g1_balance_stand`` is the write-side companion to the neon
bundle's ``g1_balance_stand`` verb: where the neon verb wraps
``LocoClient.BalanceStand`` (which internally reaches the SDK's
``SetBalanceMode`` handler) under a single-writer lock, this one
hands a target balance-mode id to the driver's own write path and
reads back the envelope the driver produced. The driver's
``G1Driver.balance_stand`` method is not yet plumbed today
(refs strands-labs/robots#358 for the SDK-facing gate work the
write belongs on), so the :func:`live_handle_refusal` grader
refuses a handle without a ``balance_stand`` accessor with a
message naming the verb, the ``driver`` parameter and the
accessor; these tests fix the shape the verb passes through for
each of the driver's current and future outcomes (a driver-side
refusal, a future success envelope, and the verb's own ``driver``
/ ``balance_mode`` refusals).

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

from strands_robots.tools.g1.g1_balance_stand import g1_balance_stand


class _StubG1Driver:
    """A driver double whose ``balance_stand`` returns a fixed envelope.

    ``g1_balance_stand`` calls ``driver.balance_stand(balance_mode)``
    and returns the envelope verbatim. This double sits under the
    same interface without pulling the real driver's imports (the
    real class reaches CycloneDDS at construction time in some
    paths), so a test can hand a wired-shape envelope to the verb
    without a bus. ``calls`` records the ``balance_mode`` per
    invocation so a test can pin "the verb writes the driver
    exactly once" and "the verb passes the argument through
    unchanged" without asking the driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls: list[int] = []

    def balance_stand(self, balance_mode: int) -> dict[str, Any]:
        self.calls.append(balance_mode)
        return self._envelope


def _call(
    driver: Any,
    *,
    balance_mode: int | None = 0,
) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim. This helper is where a shape
    drift would surface once, rather than at every call site. The
    default ``balance_mode=0`` is the static-balance mode the neon
    bundle documented as the default, so a call that omits the
    value here still reaches the driver with a target the
    controller admits.
    """
    return g1_balance_stand(driver=driver, balance_mode=balance_mode)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a
    submodule at import time would break every headless CI runner
    and Thor before an office bring-up. The driver enforces the
    same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the
    only path that loads the SDK); this cell holds the balance-
    stand verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_balance_stand")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_balance_stand imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        "loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_side_refusal_surfaces_verbatim() -> None:
    """The driver's refusal envelope round-trips through the verb unchanged.

    ``G1Driver.balance_stand`` will (once landed) refuse an
    SDK-side raise with a named error envelope, and today's driver
    refuses every call because the method is not yet plumbed. Both
    shapes are ``{"status": "error", "content": [{"text": ...}]}``
    and the verb passes either through; a wording drift on the
    driver side moves this verb with it (refs
    strands-labs/robots#2874).
    """
    refusal_text = "balance_stand: BalanceStand(mode=0) raised: RPC_CLIENT_API_TIMEOUT"
    envelope = {"status": "error", "content": [{"text": refusal_text}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, balance_mode=0)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == refusal_text


def test_a_future_success_envelope_round_trips_verbatim() -> None:
    """Once ``balance_stand`` lands, the success envelope surfaces verbatim.

    When the driver's method is plumbed, ``G1Driver.balance_stand``
    will surface the SDK's outcome inside a ``{"status":
    "success", "content": [{"json": {"rc": 0, "message":
    "BalanceStand(mode=0) dispatched"}}]}`` envelope. The verb
    does not reshape it - a future field the driver adds reaches
    a caller the moment the driver writes it, and this cell holds
    that pass-through explicit so the verb is ready the moment the
    write path lands.
    """
    envelope = {
        "status": "success",
        "content": [
            {
                "json": {
                    "rc": 0,
                    "message": "BalanceStand(mode=0) dispatched",
                }
            }
        ],
    }
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, balance_mode=0)

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
    result = g1_balance_stand(driver=None, balance_mode=0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_balance_stand" in text
    assert "driver" in text


def test_a_wrong_shape_driver_is_refused_with_a_message_naming_the_type() -> None:
    """A robot *name* (string) is refused before the accessor call.

    A model that synthesizes the argument as a robot name reaches
    the verb with a ``str``. The verb owes an envelope-shaped
    refusal that names the type it received and the remedy - the
    four invariants every ``@tool`` handler in this package holds
    - and this cell fixes the shape.
    """
    result = g1_balance_stand(driver="unitree_g1", balance_mode=0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_balance_stand" in text
    assert "'str'" in text


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.balance_stand``.

    A double-call from a wrapper that retried inside the verb
    would issue two ``BalanceStand`` writes on the same admission
    window; the SDK's handler is not re-entrant and neon's own
    bundle held a single-writer lock. This cell pins the verb to
    a single driver call per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    _call(driver)
    assert len(driver.calls) == 1


def test_the_verb_passes_the_argument_through_unchanged() -> None:
    """The balance mode the driver receives is the one the caller passed.

    The verb's contract is a pass-through: it does not translate
    argument names, does not synthesize defaults the driver did
    not ask for, and does not clamp the value. This cell fixes
    that ``balance_mode`` reaches the driver method verbatim - a
    rename or coercion on either side is a driver-level contract
    change, not a silent verb-side translation.
    """
    envelope = {"status": "success", "content": [{"json": {}}]}
    driver = _StubG1Driver(envelope=envelope)
    _call(driver, balance_mode=3)
    assert driver.calls == [3]


def test_the_wrong_shape_driver_is_not_called() -> None:
    """A wrong-shape driver is refused before the accessor path.

    The shared
    :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
    guard is called first; a numeric handle is refused with an
    envelope and the driver's method is never reached. This cell
    holds the ordering by grading a handle that would raise if
    called (``int`` has no ``balance_stand`` attribute) and
    observing that the refusal envelope has the four invariants
    without an exception in flight.
    """
    result = g1_balance_stand(driver=42, balance_mode=0)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_balance_stand" in text
    assert "'int'" in text


def test_a_missing_balance_mode_is_refused_with_a_message_naming_the_parameter() -> None:
    """A ``None`` ``balance_mode`` is refused before the driver is called.

    ``balance_mode`` is a data parameter the tool schema *does*
    describe, so a model can synthesize the wrong shape here as
    easily as it can reach the verb with the right one. ``None``
    is the "the model left it out" shape: the verb owes an
    envelope-shaped refusal naming the parameter and the remedy,
    and this cell fixes that the refusal fires before the driver
    is reached.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_balance_stand(driver=driver, balance_mode=None)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_balance_stand" in text
    assert "balance_mode" in text
    assert driver.calls == []


def test_a_wrong_shape_balance_mode_is_refused_with_a_message_naming_the_parameter() -> None:
    """A non-integer ``balance_mode`` is refused before the driver is called.

    A model may synthesize the mode id as a string ``"0"`` or a
    float ``0.0``; the SDK's handler is int-only (the neon
    bundle's own wrapper coerced through :class:`int` before
    dispatch, so a float or string reaching that path would either
    be silently truncated or raise on the wire). The verb refuses
    the shape here so the envelope names ``balance_mode`` and the
    remedy is decidable rather than SDK-version dependent.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_balance_stand(driver=driver, balance_mode="0")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_balance_stand" in text
    assert "balance_mode" in text
    assert driver.calls == []


def test_a_float_balance_mode_is_refused_before_the_driver_is_called() -> None:
    """A ``float`` ``balance_mode`` is refused rather than truncated.

    ``LocoClient.BalanceStand`` takes an :class:`int` and the neon
    bundle coerced through ``int(balance_mode)`` before dispatch,
    which would silently truncate ``0.9`` to ``0`` (a valid mode
    id the caller did not name) or ``3.5`` to ``3`` (also a valid
    mode id). Refusing the shape here rather than truncating keeps
    the verb from silently transitioning to a balance mode the
    caller writing ``0.9`` did not name; a caller who wants a
    specific mode reaches the verb with an integer explicitly.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_balance_stand(driver=driver, balance_mode=0.9)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_balance_stand" in text
    assert "balance_mode" in text
    assert driver.calls == []


def test_a_boolean_balance_mode_is_refused_because_it_would_act_as_a_silent_int() -> None:
    """A ``True`` payload is refused rather than coerced to ``1``.

    ``bool`` is an ``int`` subclass, so a caller passing ``True``
    would reach the SDK's ``BalanceStand(1)`` handler - a mode id
    outside the neon-bundle-observed admitted set ``{0, 3}`` that
    the controller silently accepts and ignores - through a
    signature that names none of that; ``False`` would collapse
    to ``BalanceStand(0)`` (the static-balance default) silently.
    Refusing ``bool`` explicitly rather than silently transitioning
    matches the shape refusal every other numeric verb in this
    package renders on the same subclass hazard; a caller who
    wants ``0`` or ``3`` reaches the verb with the integer
    explicitly.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": {}}]})
    result = g1_balance_stand(driver=driver, balance_mode=True)  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "g1_balance_stand" in text
    assert "balance_mode" in text
    assert driver.calls == []


def test_a_zero_balance_mode_is_admitted_because_it_is_the_static_balance_mode() -> None:
    """A ``balance_mode=0`` reaches the driver: 0 is the static-balance default.

    Unlike a positive-only knob, ``balance_mode`` at ``0`` is a
    caller-facing value the neon bundle's own wrapper did not
    reject: it is the static-balance default the neon bundle
    documented, and an admitted mode id. This cell pins that a
    caller who wants static balance reaches the driver with the
    target verbatim.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, balance_mode=0)

    assert result["status"] == "success"
    assert driver.calls == [0]


def test_a_dynamic_balance_mode_reaches_the_driver_unchanged() -> None:
    """A ``balance_mode=3`` reaches the driver: 3 is the dynamic-balance mode.

    The neon bundle observed two walkable modes, of which ``3``
    is the dynamic-balance mode. This cell pins that
    the verb reaches the driver with ``3`` verbatim (no
    substitution to the static default), so a caller upgrading
    from the neon bundle reaches the same behaviour.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, balance_mode=3)

    assert result["status"] == "success"
    assert driver.calls == [3]


def test_a_balance_mode_outside_the_admitted_set_reaches_the_driver_unchanged() -> None:
    """A ``balance_mode=7`` reaches the driver unchanged - the verb does not domain-refuse.

    The neon-bundle-observed admitted set is ``{0, 3}``; the
    module docstring names "does not refuse a balance_mode outside the
    admitted set" as one of the things this verb does not do.
    Refusing an unlisted mode here would fork the neon bundle's
    admission set into a second source of truth this module would
    then have to keep in sync with the envelope lookup. This cell
    pins that a caller who passes ``7`` reaches the driver's own
    refusal (or the SDK's silent-accept-and-ignore handler)
    through the verb's pass-through, rather than being intercepted
    here.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, balance_mode=7)

    assert result["status"] == "success"
    assert driver.calls == [7]


def test_a_negative_balance_mode_reaches_the_driver_unchanged() -> None:
    """A negative ``balance_mode`` is passed to the driver, not clamped in the verb.

    The neon bundle's own wrapper coerced through
    ``int(balance_mode)`` and dispatched the result unchanged; a
    negative id is not a shape violation
    (:class:`int` accepts it) but a domain violation the SDK's
    handler will either silently ignore or refuse. The module
    docstring names "does not refuse a balance_mode outside the
    admitted set" as one of the things this verb does not do. This
    cell pins that the verb does not refuse or transform a
    negative value on its own - a rewording of the driver's
    refusal would then reach a caller through the envelope, not
    through a verb-side interception.
    """
    envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
    driver = _StubG1Driver(envelope=envelope)
    result = _call(driver, balance_mode=-1)

    assert result["status"] == "success"
    assert driver.calls == [-1]
