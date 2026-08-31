"""Shared DDS init and safety helpers for the G1 hardware layer.

Two things need to be true across the driver and (later, issue #358) the
agent tools:

* ``ChannelFactoryInitialize`` is called exactly once per process, with a
  known network interface. Calling it never leaves subscribers with no bus;
  calling it a second time is *not* an error the SDK reports -
  ``ChannelFactory.Init`` short-circuits on ``if __initialized: return True``
  and ignores its ``networkInterface`` argument, so a re-init is a silent
  no-op that is indistinguishable from a successful bind.
  :func:`ensure_dds` runs it under a lock, records the interface it actually
  bound, and is idempotent for callers with the same choice.
* A ``ChannelSubscriber`` and a ``ChannelPublisher`` cannot be constructed
  concurrently: the CycloneDDS bindings segfault. :data:`_DDS_INIT_LOCK` is
  the *shared* lock the driver and the tools (issue #358) both hold while
  creating readers or writers. One lock; two consumers.

The ``unitree_sdk2py`` import is lazy - the module attribute stays ``None`` and
is loaded inside :func:`ensure_dds` on first call - so importing this module
without the SDK installed succeeds. That is what lets every test in this repo
mock the bus and skips the SDK entirely on Thor.
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error decoder shared with agent tools (issue #358) so the same numeric code
# renders the same text everywhere. Values sourced from ``unitree_sdk2py``
# response codes observed against the real G1; ``0`` is the SDK's success
# marker and always present.
# ---------------------------------------------------------------------------
ERR_CODES: dict[int, str] = {
    0: "OK",
    3102: "RPC_CLIENT_SEND fail",
    3103: "RPC_CLIENT_API_NOT_REG",
    3104: "RPC_CLIENT_API_TIMEOUT",
    7301: "LocoState not available",
    7302: "Invalid FSM id (loco)",
    7303: "Invalid task id (loco)",
    7400: "rt/armsdk topic is occupied",
    7401: "Arm is holding - release first (id=99)",
    7402: "Invalid action id",
    7404: "Invalid FSM id - need FSM in {500, 501, 801}",
}

#: FSM ids where arm-SDK-shaped commands are honoured. Outside this set the
#: arm refuses to move at all - so
#: :meth:`~strands_robots.drivers.g1.G1Driver.send_action` checks membership
#: before publishing its ``LowCmd_`` on ``rt/lowcmd``. The set is defined
#: against the motion-switcher FSM (arm-SDK, sit, sit-down), not against a
#: topic name; the ``rt/armsdk`` topic itself is future work for the
#: ``g1_tools`` client (issue #358) and this driver does not write it.
HANDSHAKE_FSMS: frozenset[int] = frozenset({500, 501, 801})

#: FSM ids where locomotion velocity commands are honoured. Narrower than
#: :data:`HANDSHAKE_FSMS` because sitting (500) accepts arm gestures but not
#: walking.
WALK_FSMS: frozenset[int] = frozenset({501, 801})


def decode_code(code: object) -> str:
    """Render a Unitree SDK response code as ``"NNNN (meaning)"``.

    Falls back to a bare stringified value for anything the table does not
    know, so a new error code from a firmware update surfaces at least as its
    integer instead of vanishing into ``"OK"``. Non-integer inputs (``None``,
    the ``ret`` field a mocked client may leave as a string) render as their
    ``repr`` for the same reason: a caller reading the log should see what the
    driver actually received.
    """
    if isinstance(code, int):
        return f"{code} ({ERR_CODES.get(code, 'unknown')})"
    return f"{code!r}"


# ---------------------------------------------------------------------------
# DDS init lock - held by both driver and agent tools (issue #358) around
# every subscriber/publisher construction, because the CycloneDDS bindings
# segfault under concurrent creation. One lock, shared.
# ---------------------------------------------------------------------------
_DDS_INIT_LOCK: threading.Lock = threading.Lock()

#: The attribute ``ChannelFactory`` records "the bus is up" on. Nothing public
#: reports it: ``Init`` returns ``True`` both when it binds an interface and
#: when it short-circuits, and ``ChannelFactoryInitialize`` discards that bool.
#: Reading the attribute is the only way to tell a bind from a no-op, and it is
#: a narrower coupling to the SDK than matching its exception text, which
#: :func:`ensure_dds` already does below.
_SDK_FACTORY_BOUND_ATTR = "_ChannelFactory__initialized"


def _sdk_factory_already_bound(channel_module: object) -> bool | None:
    """Whether the SDK's channel factory is already bound to an interface.

    Args:
        channel_module: The imported ``unitree_sdk2py.core.channel`` module.

    Returns:
        ``True`` or ``False`` when this SDK build reports its factory state;
        ``None`` when it does not expose it, so a caller keeps the behaviour
        it had before rather than guessing which way to fail.
    """
    factory = getattr(channel_module, "ChannelFactory", None)
    bound = getattr(factory, _SDK_FACTORY_BOUND_ATTR, None)
    return bound if isinstance(bound, bool) else None


def _bound_elsewhere_error(network_interface: str) -> str:
    """The reason a bus this process did not bind cannot be confirmed.

    Args:
        network_interface: The interface the caller asked to bind.

    Returns:
        A named reason, in the voice the caller can act on.
    """
    return (
        "the DDS channel factory was already initialised outside ensure_dds, so a "
        f"bind to {network_interface!r} cannot be confirmed; let the driver initialise "
        "the bus, or drop the ChannelFactoryInitialize call that runs before it"
    )


# Recorded once by :func:`ensure_dds` for a running process. A second call
# with a different interface is a bug worth catching, not a silent no-op.
_dds_state: dict[str, object] = {"initialized": False, "interface": None}


def ensure_dds(network_interface: str = "eth0") -> str | None:
    """Initialise the DDS channel factory, at most once per process.

    Returns ``None`` on success. Returns a named error string when the SDK is
    missing or the underlying initialise raised - the caller decides whether
    that is fatal (a real bring-up on hardware) or expected (a headless test
    that mocks the bus). The lock covers both the SDK import and the
    ``ChannelFactoryInitialize`` call so two threads racing on the first
    connection cannot double-initialise.

    Args:
        network_interface: The interface to bind CycloneDDS to.

    Returns:
        ``None`` on success. An error string otherwise; never raises.
    """
    with _DDS_INIT_LOCK:
        if _dds_state["initialized"]:
            recorded = _dds_state["interface"]
            if recorded != network_interface:
                # A silent re-bind would attach subscribers to the wrong NIC
                # and produce empty topics with no obvious cause. Refuse.
                return (
                    f"ChannelFactoryInitialize was called with interface "
                    f"{recorded!r}; refusing to re-initialise on {network_interface!r}"
                )
            return None
        try:
            # Lazy import - the SDK is only touched here. On a machine that
            # does not have it (Thor, CI, every unit test) the ImportError is
            # returned as a named string rather than raised, and the caller
            # can decide whether to proceed with a mocked bus.
            sdk_channel = importlib.import_module("unitree_sdk2py.core.channel")
        except ImportError as exc:  # pragma: no cover - exercised on hardware
            return f"unitree_sdk2py is not installed: {exc}"
        if _sdk_factory_already_bound(sdk_channel):
            # Something bound the bus without coming through here, so the
            # interface it is on is not this process's to know. Calling
            # ChannelFactoryInitialize now would short-circuit and return
            # normally, and recording ``network_interface`` off the back of
            # that would attach every later subscriber to whichever NIC the
            # first caller chose while reporting the one this caller asked
            # for - the silent re-bind the refusal below exists to prevent.
            return _bound_elsewhere_error(network_interface)
        try:
            sdk_channel.ChannelFactoryInitialize(0, network_interface)
        except Exception as exc:  # noqa: BLE001 - SDK raises bare Exception
            message = str(exc).lower()
            if "already" in message or "initialized" in message:
                # An SDK build that refuses a second init instead of
                # short-circuiting: the same situation as the probe above, so
                # the same answer. The bus is up, but on an interface this
                # process did not choose, so reporting success here would
                # record a NIC nobody bound.
                return (
                    "ChannelFactoryInitialize reports the factory was already "
                    f"initialised, so a bind to {network_interface!r} cannot be "
                    f"confirmed: {exc}"
                )
            return f"ChannelFactoryInitialize failed: {exc}"
        _dds_state["initialized"] = True
        _dds_state["interface"] = network_interface
        logger.info("ChannelFactoryInitialize(0, %r) succeeded", network_interface)
        return None


def reset_dds_state() -> None:
    """Reset the recorded DDS init state - test hook only.

    Nothing in production calls this. The CycloneDDS factory has no
    ``Shutdown`` in ``unitree_sdk2py``, so a real reset requires a fresh
    process; this only lets a test pretend the factory was never initialised.
    """
    with _DDS_INIT_LOCK:
        _dds_state["initialized"] = False
        _dds_state["interface"] = None


def _handle_refusal_envelope(text: str) -> dict[str, Any]:
    """Wrap a refusal sentence in the envelope every ``@tool`` owes a caller."""
    return {"status": "error", "content": [{"text": text}]}


def live_handle_refusal(
    verb: str,
    driver: Any,
    *,
    accessor: str,
    reads: str,
    expected: str,
) -> dict[str, Any] | None:
    """Return the refusal envelope for an unusable live-handle ``driver``, or ``None``.

    Every verb in this package that takes a wired driver reads it through one
    accessor, and the handle is a **live Python object** rather than a value an
    agent can synthesize: it is typed :class:`~typing.Any` so this package stays
    out of the import cycle the driver's reach into :func:`ensure_dds` would
    close, and that annotation carries no type into the generated tool schema.
    A caller therefore reaches these verbs with whatever it has, and a verb whose
    first statement is the accessor call surfaces ``None``, a robot *name*, or
    any object without the accessor as ``AttributeError`` naming a private
    attribute rather than as a refusal naming the parameter.

    This is the one implementation of that judgement, and it keeps four
    invariants for every verb in the package: the answer is an error *envelope*
    and never an exception, it names the verb, it names ``driver``, and it names
    the type it received. The prose differs per accessor because the remedy does - which
    object to pass, and what that object has to answer - so the wording is the
    caller's parameter and the invariants are not.

    Args:
        verb: The tool name, used to open the message so a caller reading a
            transcript can tell which verb refused.
        driver: The handle to judge. Answers ``None`` only when the object can
            answer ``accessor``; a wrong handle is refused rather than coerced.
        accessor: The attribute the verb reads the handle through. It must be
            *callable* on the handle, not merely present - a namespace built
            from a cache dump carries the name as data, and a guard testing
            only for presence would accept it and then fail on the call.
        reads: Why an agent cannot supply the handle, completing the sentence
            "an agent cannot synthesize it, because ...".
        expected: What the handle failed to expose and what to pass instead,
            completing the sentence "of type 'str' does not expose ...".

    Returns:
        ``None`` when ``driver`` exposes a callable ``accessor``, otherwise the
        ``{"status": "error", "content": [...]}`` envelope every ``@tool`` owes
        a caller instead of an exception.
    """
    if driver is None:
        return _handle_refusal_envelope(
            f"{verb}: `driver` is required. Pass the live G1Driver "
            "handle the orchestrator constructed - an agent cannot "
            f"synthesize it, because {reads}."
        )
    if not callable(getattr(driver, accessor, None)):
        return _handle_refusal_envelope(
            f"{verb}: `driver` of type {type(driver).__name__!r} does not expose {expected}"
        )
    return None


def snapshot_handle_refusal(verb: str, driver: Any) -> dict[str, Any] | None:
    """Return the refusal envelope for an unusable ``driver`` handle, or ``None``.

    The sensor verbs in this package read a cache the driver's own DDS
    subscriber writes, through the driver's ``_snapshot`` accessor. This is
    :func:`live_handle_refusal` bound to that accessor, shared because five
    verbs need it rather than restated in each.

    ``strands_robots.tools.run_policy`` is the one ``@tool`` outside this
    package whose parameter is a live handle typed :class:`~typing.Any`, and it
    refuses both shapes with a message naming the parameter and the remedy.

    Args:
        verb: The tool name, used to open the message so a caller reading a
            transcript can tell which verb refused.
        driver: The handle to judge. Answers ``None`` only when the object can
            answer the accessor; a wrong handle is refused rather than coerced.

    Returns:
        ``None`` when ``driver`` exposes a callable ``_snapshot``, otherwise the
        ``{"status": "error", "content": [...]}`` envelope every ``@tool`` owes
        a caller instead of an exception.
    """
    return live_handle_refusal(
        verb,
        driver,
        accessor="_snapshot",
        reads="the verb reads a cache the driver's own DDS subscriber writes",
        expected=(
            "the cached-snapshot accessor this verb reads. Pass a "
            "strands_robots G1Driver, or an object with a callable "
            "`_snapshot(attr)` answering the cache dict the driver's decoder "
            "writes."
        ),
    )
