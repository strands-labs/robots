"""Device shadows for a mounted device (MAPPING rows 1.1–1.5).

Three named shadows per thing: ``manifest`` (``reported`` = the device's record, written once at mount, deleted at
unmount: a dashboard that has never met the device draws it from one GET), ``state`` (``reported`` = the live state
at ``shadow_hz`` ≤ 1, the last-known view; the live stream stays on the ``state`` key) and ``control``
(``desired.<control>`` = an operator's target, which arrives as a delta and goes through ``robot.confirm_act`` like
every other write; the outcome is written back to ``reported``, a refusal verbatim under ``reported.refused``).

An e-stop or resume as a shadow field is refused: the shadow is eventually consistent and persists, a latch is
immediate and must never replay. The lockout is *reported* into ``state`` so a late operator sees it.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

MAX_SHADOW_HZ = 1.0
"""Above 1 Hz a shadow is telemetry (measured: the per-shadow throttle sits at 20–30 updates/s, 429 beyond; the
mesh ``state`` key carries 10–500 Hz for the same bytes)."""
SHADOW_DOC_CAP = 8 * 1024
"""IoT rejects a reported document over 8,192 bytes (413, nothing stored)."""
NAMES = ("manifest", "state", "control")
_RPC_FIELDS = ("estop", "resume", "hold", "stop")


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
        watchdog_ms: float | None = 500.0,
        lockout: Callable[[], Mapping[str, Any]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """``link`` publishes/subscribes MQTT topics (``AwsLink`` or ``FakeLink``); ``thing`` defaults to the device
        id (the thing name IS the device id, MAPPING 3.1). ``lockout()`` returns the device's lockout as a dict so the
        ``state`` shadow reports it. A delta older than ``watchdog_ms`` is dropped as stale (``None``: never)."""
        if not 0 < shadow_hz <= MAX_SHADOW_HZ:
            raise ValueError(
                f"shadow_hz={shadow_hz} is over the {MAX_SHADOW_HZ:g} Hz shadow limit (20 req/s per-thing quota, "
                f"8 KB documents); subscribe to the mesh state stream for live data"
            )
        self.link, self.robot, self.record = link, robot, dict(record)
        self.thing = thing or str(record["id"])
        self.shadow_hz, self.watchdog_ms, self.clock = float(shadow_hz), watchdog_ms, clock
        self._lockout = lockout
        self._controls = {str(c["name"]) for c in (self.record.get("manifest") or {}).get("controls") or ()}
        self._undo: list[Callable[[], None]] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.updates = {"manifest": 0, "state": 0, "control": 0}
        self.applied: list[dict[str, float]] = []

    def start(self) -> ShadowMirror:
        """Write ``manifest``, start reporting ``state`` at ``shadow_hz``, listen for ``control`` deltas."""
        self._update("manifest", {"reported": self.record})
        self._undo.append(self.link.subscribe(shadow_topic(self.thing, "control", "update/delta"), self._on_delta))
        if hasattr(self.robot, "subscribe"):
            self._undo.append(self.robot.subscribe("state", self._on_state, hz=self.shadow_hz))
        return self

    def stop(self) -> None:
        """Delete the ``manifest`` shadow (the device is gone, not stale) and stop reporting. Idempotent."""
        if self._stop.is_set():
            return
        self._stop.set()
        for u in reversed(self._undo):
            u()
        self._publish(shadow_topic(self.thing, "manifest", "delete"), {})

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
        self.apply(desired, stamps)

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
                refused[key] = f"{key}: target is older than watchdog_ms={self.watchdog_ms:g}; dropped as stale"
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
        self._update("control", {"reported": reported, **({"desired": clear} if clear else {})})
        return reported

    def _stale(self, key: str, stamps: Mapping[str, Any] | None) -> bool:
        if self.watchdog_ms is None or not stamps:
            return False
        ts = (stamps.get("desired") or stamps).get(key)
        ts = ts.get("timestamp") if isinstance(ts, Mapping) else ts
        return isinstance(ts, (int, float)) and (self.clock() - float(ts)) * 1000.0 > self.watchdog_ms
