"""Make a socket's DEATH visible - the missing half of the camera stream's story."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


def _mb(n: int) -> float:
    return n / (1024 * 1024)


def close_verdict(
    *,
    frames_sent: int,
    lifetime_s: float,
    publishing: bool,
    bytes_sent: int = 0,
) -> str:
    """One sentence naming what this closed socket actually was."""
    if frames_sent > 0:
        rate = f", {_mb(bytes_sent) / lifetime_s:.2f} MB/s" if bytes_sent and lifetime_s >= 0.05 else ""
        size = f" / {_mb(bytes_sent):.1f} MB" if bytes_sent else ""
        fps = f" ({frames_sent / lifetime_s:.1f} fps{rate})" if lifetime_s >= 0.05 else ""
        return f"streamed {frames_sent} frames{size} over {lifetime_s:.1f}s{fps}"
    if not publishing:
        return (
            f"sent nothing in {lifetime_s:.1f}s - that camera is not publishing to the mesh; "
            "the robot may not be running, or it could not open the device"
        )
    if lifetime_s < 2.0:
        return (
            f"sent nothing and closed after {lifetime_s:.1f}s - the client hung up before a "
            "frame was due (a page reload, or a retry loop with no backoff)"
        )
    return f"sent nothing in {lifetime_s:.0f}s although the camera is publishing - frames are not reaching this socket"


@dataclass
class _Seen:
    last_logged_at: float
    suppressed: int = 0


class CloseLogThrottle:
    """Log the first close for a key, then at most one per ``window_s``."""

    def __init__(self, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._window_s = float(window_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: dict[str, _Seen] = {}
        # a bound, for the same reason ttl_cache exists: keys are peer/camera pairs and
        # a spawn loop can invent them
        self._max_keys = 256

    def should_log(self, key: str) -> tuple[bool, int]:
        """Return (log_now, suppressed_since_last_line)."""
        now = self._clock()
        with self._lock:
            entry = self._seen.get(key)
            if entry is None:
                if len(self._seen) >= self._max_keys:
                    self._seen.pop(next(iter(self._seen)))
                self._seen[key] = _Seen(last_logged_at=now)
                return True, 0
            if now - entry.last_logged_at >= self._window_s:
                suppressed = entry.suppressed
                entry.last_logged_at = now
                entry.suppressed = 0
                return True, suppressed
            entry.suppressed += 1
            return False, entry.suppressed


def close_line(*, peer_id: str, cam: str, verdict: str, suppressed: int) -> str:
    """The log line itself, with the storm count when there is one."""
    tail = f" [+{suppressed} more closes suppressed in the last minute]" if suppressed else ""
    return f"camera socket {peer_id}/{cam} closed: {verdict}{tail}"


MAX_CAP_FPS = 30.0

MIN_CAP_FPS = 0.1


def fps_cap(raw: str | None) -> float | None:
    """Parse a viewer's requested frame rate. ``None`` means "as fast as frames arrive"."""
    if raw is None or raw == "":
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or v <= 0:  # NaN or nonsense
        return None
    return max(MIN_CAP_FPS, min(MAX_CAP_FPS, v))


def cap_note(cap: float | None) -> str:
    """What the close verdict says about the rate this socket agreed to."""
    return "" if cap is None else f" [viewer capped at {cap:g} fps]"
