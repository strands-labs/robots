"""Opening a motion-switcher client holds the shared DDS lock, in every driver.

``_g1_common``'s module docstring states the contract graded here: "A
``ChannelSubscriber`` and a ``ChannelPublisher`` cannot be constructed
concurrently: the CycloneDDS bindings segfault. ``_DDS_INIT_LOCK`` is the
*shared* lock the driver and the tools both hold while creating readers or
writers. One lock; two consumers."

An RPC client's ``Init()`` is an endpoint construction - it builds the client's
DDS request/response channels - so it belongs on the same lock. Both native
Unitree drivers opened one without it while
:class:`~strands_robots.tools.g1._dds_engine.DDSSubscriberSet` constructed
subscribers under it on the *same driver instance's* streaming, rollout and
mesh-telemetry threads. A shared-device lock is only a guarantee where every
caller takes it, and the caller that loses the race is the one holding nothing,
so the unconverted side is the side that breaks. The loss is a native segfault,
which no ``except`` boundary can turn into an error envelope: the process dies,
possibly while the robot stands under its own controller.

These cells grade the *behaviour* - was the shared lock held while the client
was built - rather than the source. That matters here, because the same
violation is invisible to a source rule when the call is laundered through a
lower-case local binding (``init = getattr(client, "Init"); init()``), which is
exactly how the G1 driver spelled it. A behavioural pin cannot be evaded by
renaming the callee.

:func:`test_the_open_waits_for_a_competing_endpoint_construction` is the cell
that makes the pin specific: a driver-private lock would satisfy "some lock was
held" while excluding nothing the engine does, so the pin insists the open
actually blocks behind a competing holder of the shared lock.

No test here needs a robot, a DDS bus or ``unitree_sdk2py``: the SDK client is a
recorder staged on :mod:`sys.modules` (the G1's lazy-import path) or behind the
loader the Go2 imports.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import Any

import pytest

from strands_robots.drivers.g1 import G1Driver
from strands_robots.drivers.go2 import Go2Driver
from strands_robots.tools.g1 import _motion_switcher
from strands_robots.tools.g1._g1_common import _DDS_INIT_LOCK

#: How long a cell waits for something that should already have happened, and
#: how long it waits to conclude that something is correctly still blocked.
_TIMEOUT_S = 5.0
_BLOCKED_S = 0.2

_SDK_MODULE = "unitree_sdk2py.comm.motion_switcher.motion_switcher_client"


class _RecordingClient:
    """An SDK-shaped client that records whether the shared lock was held.

    Both moments are recorded separately because they are separate endpoint
    touches: the constructor and ``Init``, which is what actually builds the
    request/response channels.
    """

    def __init__(self) -> None:
        self.held_at_construct = _DDS_INIT_LOCK.locked()
        self.held_at_init: bool | None = None

    def SetTimeout(self, timeout: float) -> None:  # noqa: N802 - the SDK's spelling
        """Accept the timeout the Go2 driver sets; the value is not the subject."""

    def Init(self) -> None:  # noqa: N802 - the SDK's spelling
        """Record the lock state at the moment the endpoints are built."""
        self.held_at_init = _DDS_INIT_LOCK.locked()


def _open_g1_over_the_sdk(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Open a G1 client through the driver's lazy ``importlib`` path."""
    for name in (
        "unitree_sdk2py",
        "unitree_sdk2py.comm",
        "unitree_sdk2py.comm.motion_switcher",
        _SDK_MODULE,
    ):
        module = types.ModuleType(name)
        # A stand-in package: ``importlib.import_module`` walks the parents of a
        # dotted name, and a parent without ``__path__`` is not importable.
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys.modules[_SDK_MODULE], "MotionSwitcherClient", _RecordingClient, raising=False)
    driver = G1Driver()
    with driver._motion_switcher_lock:
        return driver._open_motion_switcher_client()


def _open_go2_over_the_sdk(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Open a Go2 client through the loader the driver imports."""
    monkeypatch.setattr(_motion_switcher, "_load_motion_switcher_client", lambda: _RecordingClient)
    return Go2Driver()._open_motion_switcher_client()


def _open_g1_over_a_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Open a G1 client through the injected factory seam."""
    driver = G1Driver(motion_switcher_client_factory=lambda _nic: _RecordingClient())
    with driver._motion_switcher_lock:
        return driver._open_motion_switcher_client()


def _open_go2_over_a_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Open a Go2 client through the injected factory seam."""
    return Go2Driver(motion_switcher_client_factory=lambda _nic: _RecordingClient())._open_motion_switcher_client()


#: Every way either driver can open a motion-switcher client. The factory rows
#: are graded too: the driver cannot know whether an injected factory builds a
#: real client, and one that does owes the same serialisation.
_OPENERS = {
    "g1-over-the-sdk": _open_g1_over_the_sdk,
    "g1-over-a-factory": _open_g1_over_a_factory,
    "go2-over-the-sdk": _open_go2_over_the_sdk,
    "go2-over-a-factory": _open_go2_over_a_factory,
}


class TestTheOpenHoldsTheSharedLock:
    """Every open path constructs its client under the shared DDS lock."""

    @pytest.mark.parametrize("opener", sorted(_OPENERS))
    def test_the_client_is_built_under_the_shared_lock(self, opener: str, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _OPENERS[opener](monkeypatch)
        assert isinstance(client, _RecordingClient), (
            f"{opener}: the open returned {client!r} instead of the recorder, so this cell graded nothing"
        )
        assert client.held_at_construct, (
            f"{opener}: the client was constructed without holding the shared "
            "DDS lock, so it races every subscriber the engine builds on this "
            "driver's own threads and the loss is a native segfault"
        )

    def test_the_sdk_init_that_builds_the_endpoints_is_under_the_lock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``Init`` is the endpoint construction, so it is the moment that counts."""
        for opener in ("g1-over-the-sdk", "go2-over-the-sdk"):
            client = _OPENERS[opener](monkeypatch)
            assert client.held_at_init is True, (
                f"{opener}: Init() ran with the shared lock state "
                f"{client.held_at_init!r}; Init builds the client's DDS "
                "request/response endpoints, which is exactly what must not "
                "overlap another endpoint construction"
            )

    def test_the_lock_is_released_once_the_client_is_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The lock is a critical section, not something the driver keeps."""
        _open_go2_over_a_factory(monkeypatch)
        assert not _DDS_INIT_LOCK.locked(), (
            "the shared lock is still held after the open returned, which would "
            "stall every later subscriber construction in the process"
        )


class TestTheLockIsTheSharedOneAndNotAPrivateOne:
    """A driver-private lock would exclude nothing the engine does."""

    @pytest.mark.parametrize("opener", ["g1-over-a-factory", "go2-over-a-factory"])
    def test_the_open_waits_for_a_competing_endpoint_construction(
        self, opener: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened = threading.Event()
        release = threading.Event()

        def hold_the_lock() -> None:
            with _DDS_INIT_LOCK:
                release.wait(_TIMEOUT_S)

        holder = threading.Thread(target=hold_the_lock, daemon=True)
        holder.start()
        try:
            # Wait for the competing holder to actually own the lock, so the
            # assertion below cannot pass just because the race was not run.
            deadline = threading.Event()
            while not _DDS_INIT_LOCK.locked():
                deadline.wait(0.01)

            def open_the_client() -> None:
                _OPENERS[opener](monkeypatch)
                opened.set()

            opener_thread = threading.Thread(target=open_the_client, daemon=True)
            opener_thread.start()
            assert not opened.wait(_BLOCKED_S), (
                f"{opener}: the open completed while another thread held the "
                "shared DDS lock, so it is not serialised against the engine's "
                "endpoint construction - a private lock excludes nothing"
            )
        finally:
            release.set()
            holder.join(_TIMEOUT_S)
        assert opened.wait(_TIMEOUT_S), (
            f"{opener}: the open never completed after the shared lock was "
            "released, so it is blocked on something other than that lock"
        )
