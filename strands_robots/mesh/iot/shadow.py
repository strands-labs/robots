"""Device shadows for a mounted device (MAPPING rows 1.1–1.5).

Three named shadows per thing: ``manifest`` (``reported`` = the compact device document of ``manifest.py``:
identity, control ranges, units, rpc and event names — it fits every registry robot under the 8 KB cap, and names
the mesh key that carries the full record; written once at mount, deleted at unmount, so a dashboard that has never
met the device draws it from one GET), ``state`` (``reported`` = the live state at ``shadow_hz`` ≤ 1, the last-known
view; the live stream stays on the ``state`` key) and ``control`` (``desired.<control>`` = an operator's target,
which arrives as a delta and goes through ``robot.confirm_act`` like every other write; the outcome is written back
to ``reported``, a refusal verbatim under ``reported.refused``).

An e-stop or resume as a shadow field is refused: the shadow is eventually consistent and persists, a latch is
immediate and must never replay. The lockout is *reported* into ``state`` so a late operator sees it.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from ._reply import control_writes
from .manifest import manifest_doc

MAX_SHADOW_HZ = 1.0
"""Above 1 Hz a shadow is telemetry (measured: the per-shadow throttle sits at 20–30 updates/s, 429 beyond; the
mesh ``state`` key carries 10–500 Hz for the same bytes)."""
SHADOW_DOC_CAP = 8 * 1024
"""IoT rejects a reported document over 8,192 bytes (413, nothing stored)."""
NAMES = ("manifest", "state", "control")
_RPC_FIELDS = ("estop", "resume", "hold", "stop")
STAMP_RESOLUTION_S = 1.0
"""AWS writes ``metadata.<key>.timestamp`` as integer epoch seconds: a write time is truncated by up to 999 ms before
the delta leaves the broker, so no staleness bound under one second can be told apart from that truncation."""
MIN_STALE_S = STAMP_RESOLUTION_S


def shadow_topic(thing: str, name: str, op: str) -> str:
    """``$aws/things/<thing>/shadow/name/<name>/<op>`` (``op``: update, update/delta, update/accepted, get, delete)."""
    if name not in NAMES:
        raise ValueError(f"unknown shadow {name!r}; shadows are {list(NAMES)}")
    return f"$aws/things/{thing}/shadow/name/{name}/{op}"


def _plain(value: Any) -> Any:
    """JSON-safe scalars only (a shadow is a document, not an array carrier)."""
    if hasattr(value, "item") and not hasattr(value, "shape") or getattr(value, "shape", None) == ():
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class ShadowMirror:
    """Keep the three shadows of one mounted device in step with the robot, over an open IoT ``Link``."""

    def __init__(
        self,
        link: Any,
        robot: Any,
        record: Mapping[str, Any],
        *,
        thing: str | None = None,
        shadow_hz: float = 0.5,
        stale_s: float | None = 2.0,
        lockout: Callable[[], Mapping[str, Any]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """``link`` publishes/subscribes MQTT topics (``AwsLink`` or ``FakeLink``); ``thing`` defaults to the device
        id (the thing name IS the device id, MAPPING 3.1). ``lockout()`` returns the device's lockout as a dict so the
        ``state`` shadow reports it. A desired target whose ``metadata`` stamp is older than ``stale_s`` at delivery
        is dropped as stale (``None``: never). The bound is the shadow's own, not the mesh watchdog: a shadow is
        eventually consistent and its stamps resolve to whole seconds, a watchdog is a 500 ms liveness check."""
        if not 0 < shadow_hz <= MAX_SHADOW_HZ:
            raise ValueError(
                f"shadow_hz={shadow_hz} is over the {MAX_SHADOW_HZ:g} Hz shadow limit (20 req/s per-thing quota, "
                f"8 KB documents); subscribe to the mesh state stream for live data"
            )
        if stale_s is not None and not stale_s >= MIN_STALE_S:
            raise ValueError(
                f"stale_s={stale_s} is under the {MIN_STALE_S:g} s resolution of a shadow timestamp (AWS stamps whole "
                f"seconds); use stale_s >= {MIN_STALE_S:g}, or None to apply every delta"
            )
        self.link, self.robot, self.record = link, robot, dict(record)
        self.thing = thing or str(record["id"])
        self.shadow_hz, self.stale_s, self.clock = float(shadow_hz), stale_s, clock
        self._lockout = lockout
        self._controls = {str(c["name"]) for c in (self.record.get("manifest") or {}).get("controls") or ()}
        self._undo: list[Callable[[], None]] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.updates = {"manifest": 0, "state": 0, "control": 0}
        self.applied: list[dict[str, float]] = []
        # A delta is APPLIED on the mirror's own worker, never on the link's dispatcher thread: that thread also
        # delivers the e-stop, which must never wait behind a confirm_act (an operator gate can hold it for seconds).
        # Deltas received while one is being applied coalesce: the latest desired wins.
        self._pending: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
        self._busy = False
        self._wake = threading.Condition()
        self._worker = threading.Thread(target=self._drain, daemon=True, name=f"shadow-{self.thing}")

    def start(self) -> ShadowMirror:
        """Write ``manifest``, start reporting ``state`` at ``shadow_hz``, listen for ``control`` deltas."""
        self._update("manifest", {"reported": manifest_doc(self.record)})
        self._worker.start()
        self._undo.append(self.link.subscribe(shadow_topic(self.thing, "control", "update/delta"), self._on_delta))
        if hasattr(self.robot, "subscribe"):
            self._undo.append(self.robot.subscribe("state", self._on_state, hz=self.shadow_hz))
        return self

    def stop(self) -> None:
        """Delete the ``manifest`` shadow (the device is gone, not stale), stop reporting, and drop any delta still
        queued (a target for a device that is leaving is not applied). Idempotent."""
        if self._stop.is_set():
            return
        self._stop.set()
        with self._wake:
            self._pending = None
            self._wake.notify()
        for u in reversed(self._undo):
            u()
        self._publish(shadow_topic(self.thing, "manifest", "delete"), {})

    def drain(self, timeout: float = 5.0) -> None:
        """Wait until every received delta has been applied (a clean stop, tests)."""
        end = time.monotonic() + timeout
        while (self._pending is not None or self._busy) and time.monotonic() < end:
            time.sleep(0.005)

    def _drain(self) -> None:
        while not self._stop.is_set():
            with self._wake:
                while self._pending is None and not self._stop.is_set():
                    self._wake.wait(0.5)
                job, self._pending = self._pending, None
                self._busy = job is not None
            if job is not None:
                try:
                    self.apply(*job)
                except Exception as exc:  # an operator's input never kills the control path: say why, stay alive
                    self._report_failure(job[0], exc)
                finally:
                    self._busy = False

    def _report_failure(self, desired: Mapping[str, Any], exc: Exception) -> None:
        word = f"{len(desired)} desired keys not applied: {type(exc).__name__}: {exc}"[:512]
        try:
            self._write_control({"refused": {"*": word}}, {str(k): None for k in desired})
        except Exception:  # the link itself is gone; the next delta will try again
            return

    def _on_state(self, payload: Mapping[str, Any]) -> None:
        doc = _plain(payload)
        if self._lockout is not None:
            doc["lockout"] = _plain(self._lockout())
        self._update("state", {"reported": doc})

    def _update(self, name: str, state: Mapping[str, Any]) -> None:
        body = {"state": dict(state)}
        raw = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > SHADOW_DOC_CAP:
            raise ValueError(f"shadow {name!r} document is {len(raw)} bytes; IoT stores at most {SHADOW_DOC_CAP}")
        self.link.publish(shadow_topic(self.thing, name, "update"), raw)
        with self._lock:
            self.updates[name] += 1

    def _publish(self, topic: str, doc: Mapping[str, Any]) -> None:
        self.link.publish(topic, json.dumps(doc, separators=(",", ":")).encode())

    def _on_delta(self, _topic: str, raw: bytes) -> None:
        try:
            msg = json.loads(raw)
            desired = msg.get("state") or {}
            stamps = msg.get("metadata") or {}
        except (ValueError, AttributeError):
            return
        if not isinstance(desired, Mapping):
            return
        with self._wake:  # returns at once: the link's dispatcher is free for the next message (an e-stop)
            if self._pending is not None:
                prev_d, prev_s = self._pending
                desired, stamps = {**prev_d, **desired}, {**prev_s, **stamps}
            self._pending = (desired, stamps)
            self._wake.notify()

    def apply(self, desired: Mapping[str, Any], stamps: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """One delta: split into targets for ``confirm_act`` and refusals; write the outcome to ``control``.
        Returns the reported document."""
        action: dict[str, float] = {}
        refused: dict[str, str] = {}
        clear: dict[str, None] = {}
        for key, value in desired.items():
            key = str(key)
            if key in _RPC_FIELDS:
                refused[key] = f"{key} is an rpc, not a shadow field"
            elif key not in self._controls:
                refused[key] = f"{self.thing} has no control {key!r}"
            elif self._stale(key, stamps):
                refused[key] = f"{key}: target is older than stale_s={self.stale_s:g}; dropped as stale"
            else:
                try:
                    action[key.rsplit(".pos", 1)[0]] = float(value)
                except (TypeError, ValueError):
                    refused[key] = f"{key}: {value!r} is not a number"
            if key in refused:
                clear[key] = None
        reported: dict[str, Any] = {}
        if action:
            try:
                self.robot.confirm_act(action)
                self.applied.append(action)
                reported.update({f"{k}.pos": v for k, v in action.items()})
            except Exception as exc:  # the safety chain's own words travel verbatim
                for k in action:
                    refused[f"{k}.pos"] = str(exc)
                    clear[f"{k}.pos"] = None
        reported["refused"] = refused or None
        self._write_control(reported, clear)
        return reported

    def _write_control(self, reported: dict[str, Any], clear: dict[str, None]) -> None:
        for state in control_writes(reported, clear, SHADOW_DOC_CAP):  # the reply fits the cap whatever the delta was
            self._update("control", state)

    def _stale(self, key: str, stamps: Mapping[str, Any] | None) -> bool:
        """An integer stamp means "written somewhere in [ts, ts + 1 s)": the target is judged by the youngest
        write that stamp can stand for, so truncation alone never makes a fresh target stale."""
        if self.stale_s is None or not stamps:
            return False
        ts = (stamps.get("desired") or stamps).get(key)
        ts = ts.get("timestamp") if isinstance(ts, Mapping) else ts
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            return False
        youngest = float(ts) + (STAMP_RESOLUTION_S if isinstance(ts, int) else 0.0)
        return self.clock() - youngest > self.stale_s
