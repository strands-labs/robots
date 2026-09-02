"""A hardware-shaped ``cflib`` stand-in, so the whole driver runs with no radio.

The fake is shaped like the SDK rather than like the driver: it exposes
``commander``, ``high_level_commander``, ``platform`` and ``log`` with the method
names ``cflib`` actually has, and it records **every call in order** on one
shared list. Ordering is the point - the driver's load-bearing sequences are
orderings (arm before any setpoint, ``send_notify_setpoint_stop`` before
``land``), and a per-object call list cannot see an ordering that spans two
objects.

The link is **asynchronous**, like the real one. ``Crazyflie.open_link`` returns
as soon as the request is queued and delivers the outcome later on its own link
thread, as a ``connected`` or ``connection_failed`` callback; it never raises,
and until ``connected`` has fired ``log.add_config`` cannot find a variable in
the TOC and ``send_packet`` silently discards every packet. A fake whose
``open_link`` connected synchronously would hide all of that, so this one
answers on a thread and refuses the pre-TOC calls the firmware refuses.

Nothing here subclasses or imports ``cflib``: the tests must pass on a machine
that has never had a Crazyradio plugged in, which is every CI runner.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest


class _Recorder:
    """One shared, ordered log of ``(target.method, args)`` across every stub."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._lock = threading.Lock()
        self._counts_reached = threading.Event()
        self._await_name: str | None = None
        self._await_count = 0

    def record(self, name: str, args: tuple[Any, ...]) -> None:
        with self._lock:
            self.calls.append((name, args))
            if self._await_name is not None and self.count(self._await_name) >= self._await_count:
                self._counts_reached.set()

    def count(self, name: str) -> int:
        """How many times ``name`` was called."""
        return sum(1 for called, _ in self.calls if called == name)

    def names(self) -> list[str]:
        """Just the call names, in order."""
        return [name for name, _ in self.calls]

    def args_of(self, name: str) -> tuple[Any, ...]:
        """Arguments of the first ``name`` call."""
        for called, args in self.calls:
            if called == name:
                return args
        raise AssertionError(f"{name} was never called; calls were {self.names()}")

    def wait_for(self, name: str, count: int, timeout: float = 5.0) -> bool:
        """Block until ``name`` has been called ``count`` times, or time out.

        Bounded, and returns whether the count was reached, so a test asserts on
        the outcome rather than on a sleep long enough to be flaky.
        """
        with self._lock:
            self._await_name, self._await_count = name, count
            if self.count(name) >= count:
                return True
            self._counts_reached.clear()
        return self._counts_reached.wait(timeout)


class _Stub:
    """Anything whose methods only need recording. Accepts any call."""

    def __init__(self, recorder: _Recorder, prefix: str, raises: BaseException | None = None) -> None:
        self._recorder = recorder
        self._prefix = prefix
        self._raises = raises

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> None:
            del kwargs
            self._recorder.record(f"{self._prefix}.{name}", args)
            if self._raises is not None:
                raise self._raises

        return call


class _FakeLogConfig:
    """Shaped like ``cflib.crazyflie.log.LogConfig``.

    ``data_received_cb.add_callback`` is how the driver subscribes, and holding
    the callback is what lets a test deliver a telemetry frame by hand.
    """

    def __init__(self, recorder: _Recorder, name: str = "", period_in_ms: int = 0) -> None:
        self._recorder = recorder
        self.name = name
        self.period_in_ms = period_in_ms
        self.variables: list[tuple[str, str]] = []
        self.callbacks: list[Any] = []
        self.data_received_cb = _Callbacks(self.callbacks)

    def add_variable(self, name: str, fetch_as: str) -> None:
        self.variables.append((name, fetch_as))

    def start(self) -> None:
        self._recorder.record("log.start", ())

    def stop(self) -> None:
        self._recorder.record("log.stop", ())

    def deliver(self, data: dict[str, Any]) -> None:
        """Push one telemetry frame through, as the link thread would."""
        for callback in self.callbacks:
            callback(0, data, self)


class _Callbacks:
    def __init__(self, sink: list[Any]) -> None:
        self._sink = sink

    def add_callback(self, callback: Any) -> None:
        self._sink.append(callback)


class _Caller:
    """Shaped like ``cflib.utils.callbacks.Caller``: subscribe, then fan out."""

    def __init__(self) -> None:
        self.callbacks: list[Any] = []

    def add_callback(self, callback: Any) -> None:
        self.callbacks.append(callback)

    def call(self, *args: Any) -> None:
        for callback in list(self.callbacks):
            callback(*args)


class FakeCrazyflie:
    """Shaped like ``cflib.crazyflie.Crazyflie``, including the async link.

    ``outcome`` chooses what the link thread reports, which is the whole point
    of the fake:

    * ``"connected"`` - the TOCs come down and ``connected`` fires. The only
      state in which the aircraft is really reachable.
    * ``"failed"`` - ``connection_failed`` fires with ``failure``. What a missing
      Crazyradio, a switched-off aircraft or a malformed URI actually produce:
      ``open_link`` still returns normally and ``link`` stays ``None``.
    * ``"silent"`` - nothing ever fires. A dongle that answered the USB probe and
      then went quiet; only a bounded wait reports it.

    ``settle_delay`` holds the outcome back for that many seconds, so a test can
    grade *ordering* against the link coming up rather than against a thread
    race: with a delay, a driver that does not wait provably reaches the wire
    first, and one that waits provably does not.
    """

    def __init__(
        self,
        recorder: _Recorder,
        *,
        arming: BaseException | None = None,
        outcome: str = "connected",
        failure: str = "Cannot find a Crazyradio Dongle",
        settle_delay: float = 0.0,
    ) -> None:
        self.recorder = recorder
        self.commander = _Stub(recorder, "commander")
        self.high_level_commander = _Stub(recorder, "high_level")
        self.platform = _Stub(recorder, "platform", raises=arming)
        self.log = _FakeLink(recorder)
        self.uri: str | None = None
        self.connected = _Caller()
        self.connection_failed = _Caller()
        #: ``None`` until the link is up, exactly like ``cflib``: while it is
        #: ``None`` the real ``send_packet`` discards every packet in silence.
        self.link: object | None = None
        self._outcome = outcome
        self._failure = failure
        self._settle_delay = settle_delay

    def open_link(self, uri: str) -> None:
        """Queue the connection and return, as ``cflib`` does. Never raises."""
        self.uri = uri
        self.recorder.record("open_link", (uri,))
        threading.Thread(target=self._settle, name="fake-cf-link", daemon=True).start()

    def _settle(self) -> None:
        """The link thread reporting the outcome, after ``open_link`` returned."""
        if self._settle_delay:
            time.sleep(self._settle_delay)
        if self._outcome == "connected":
            self.link = object()
            self.log.toc_ready = True
            self.recorder.record("connected", (self.uri,))
            self.connected.call(self.uri)
        elif self._outcome == "failed":
            self.recorder.record("connection_failed", (self.uri, self._failure))
            self.connection_failed.call(self.uri, self._failure)

    def close_link(self) -> None:
        self.link = None
        self.recorder.record("close_link", ())


class _FakeLink:
    """The ``cf.log`` surface: holds the one block the driver adds.

    ``add_config`` raises until the TOC is down, as the real one does - it looks
    every variable up in the downloaded TOC and raises ``KeyError`` for a name it
    has not seen yet. That is what makes "arm and log only after ``connected``"
    a testable ordering rather than a comment.
    """

    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder
        self.block: _FakeLogConfig | None = None
        self.toc_ready = False

    def add_config(self, block: _FakeLogConfig) -> None:
        if not self.toc_ready:
            raise KeyError(f"Variable {block.variables[0][0] if block.variables else '?'} not in TOC")
        self.block = block
        self._recorder.record("log.add_config", (block.name,))


@pytest.fixture
def recorder() -> _Recorder:
    """The shared ordered call log every stub writes to."""
    return _Recorder()


@pytest.fixture
def connected(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder):  # type: ignore[no-untyped-def]
    """Build a connected, armed driver over the fake link.

    Returns a factory so a test can choose the constructor keywords (a faster
    ``setpoint_hz``, a failing arming request) and the link ``outcome``, and
    still get the same fake.
    """
    from strands_robots.drivers import crazyflie as module

    def build(  # type: ignore[no-untyped-def]
        *,
        arming: BaseException | None = None,
        outcome: str = "connected",
        failure: str = "Cannot find a Crazyradio Dongle",
        settle_delay: float = 0.0,
        **kwargs: Any,
    ):
        fake = FakeCrazyflie(recorder, arming=arming, outcome=outcome, failure=failure, settle_delay=settle_delay)
        pieces = type(
            "_Pieces",
            (),
            {
                "crtp": _Stub(recorder, "crtp"),
                "Crazyflie": lambda **_: fake,
                "LogConfig": lambda **kw: _FakeLogConfig(recorder, **kw),
            },
        )
        monkeypatch.setattr(module, "_resolve_cflib", lambda: pieces)
        driver = module.CrazyflieDriver(**kwargs)
        reason = driver.connect_eagerly()
        return driver, fake, reason

    return build
