"""A defence against a viewer that reopens the same camera forever."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

#: Opens per minute, per viewer per camera, above which this is churn and not a human.
CHURN_OPENS_PER_MIN = 20
# : The rate a churning viewer is served at.
CHURN_CAP_FPS = 2.0
#: Bound on remembered identities, so the guard cannot become the leak it prevents.
MAX_TRACKED = 512


@dataclass(frozen=True)
class ChurnVerdict:
    """What the server decided about this socket, and why - in words."""

    opens_in_window: int
    cap_fps: float | None
    reason: str | None

    @property
    def throttled(self) -> bool:
        """True when this verdict caps the stream's frame rate."""
        return self.cap_fps is not None


class ChurnGuard:
    """Counts recent opens per (viewer, camera) and caps a storm's frame rate."""

    def __init__(
        self,
        *,
        opens_per_min: int = CHURN_OPENS_PER_MIN,
        cap_fps: float = CHURN_CAP_FPS,
        max_tracked: int = MAX_TRACKED,
    ) -> None:
        self._opens_per_min = opens_per_min
        self._cap_fps = cap_fps
        self._max_tracked = max_tracked
        self._seen: dict[str, deque[float]] = {}

    def note_open(self, identity: str, *, now: float | None = None) -> ChurnVerdict:
        """Record one socket opening and judge the viewer behind it."""
        now = time.monotonic() if now is None else now
        window = self._seen.setdefault(identity, deque())
        window.append(now)
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(self._seen) > self._max_tracked:
            for key in sorted(self._seen, key=lambda k: len(self._seen[k]))[: len(self._seen) - self._max_tracked]:
                if key != identity:
                    self._seen.pop(key, None)
        count = len(window)
        if count <= self._opens_per_min:
            return ChurnVerdict(opens_in_window=count, cap_fps=None, reason=None)
        return ChurnVerdict(
            opens_in_window=count,
            cap_fps=self._cap_fps,
            reason=(
                f"this viewer opened this camera {count} times in the last minute, so the "
                f"server is pacing it at {self._cap_fps:g} fps until that settles - the "
                f"tile keeps updating, it just stops saturating the link. Reload the page "
                f"to pick up the client-side fix for the reconnect loop."
            ),
        )

    def forget(self, identity: str) -> None:
        """Drop an identity (tests, and an operator who wants a clean measurement)."""
        self._seen.pop(identity, None)


def viewer_identity(*, subject: str | None, host: str | None, peer_id: str, cam: str) -> str:
    """Who is watching what - the key the guard counts."""
    who = subject or host or "unknown"
    return f"{who}|{peer_id}|{cam}"


def effective_cap(requested: float | None, churn: float | None) -> float | None:
    """The rate actually served: the LOWER of what the viewer asked for and what the server will give a
    storm.
    """
    caps = [c for c in (requested, churn) if c is not None]
    return min(caps) if caps else None
