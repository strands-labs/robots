"""Count refused handshakes per client so a retry storm reads as a storm, not as noise."""

from __future__ import annotations

from dataclasses import dataclass, field

#: How far back "recent" reaches. Five minutes is short enough that a fixed client stops being
#: reported almost immediately, and long enough that a 30s-backoff loop still shows up.
WINDOW_S = 300.0

# : Refusals from ONE client inside the window that mean "this is a loop, not a person typing
# a : wrong token".
STORM_THRESHOLD = 10

# : Distinct clients tracked.
MAX_CLIENTS = 64


@dataclass
class RefusalTally:
    """Every refused /api or /ws handshake, in memory, bounded."""

    total: int = 0
    #: (client, path, kind) -> ring of epoch seconds, newest last
    _recent: dict[tuple[str, str, str], list[float]] = field(default_factory=dict)
    #: refusals dropped because MAX_CLIENTS was reached (counted, never hidden)
    untracked: int = 0

    def record(self, *, client: str, path: str, now: float, kind: str = "credential") -> None:
        """Note one refusal of *client* on *path* at *now*."""
        self.total += 1
        key = (client or "?", path or "?", kind or "credential")
        seen = self._recent.get(key)
        if seen is None:
            if len(self._recent) >= MAX_CLIENTS:
                self.untracked += 1
                return
            seen = []
            self._recent[key] = seen
        seen.append(now)
        # Trim on write: the read path is /api/health, which must stay cheap even mid-storm.
        cutoff = now - WINDOW_S
        if seen[0] < cutoff:
            self._recent[key] = [t for t in seen if t >= cutoff]

    def summary(self, now: float, *, detailed: bool = True) -> dict[str, object] | None:
        """What /api/health should say, or ``None`` when there is nothing to report."""
        cutoff = now - WINDOW_S
        live = {k: [t for t in v if t >= cutoff] for k, v in self._recent.items()}
        live = {k: v for k, v in live.items() if v}
        recent = sum(len(v) for v in live.values())
        if not self.total:
            return None
        out: dict[str, object] = {
            "total": self.total,
            "recent": recent,
            "window_s": int(WINDOW_S),
            "clients": len(live),
        }
        if self.untracked:
            out["untracked"] = self.untracked
        if not live:
            # Refused in the past, quiet now. Worth saying, because "the storm stopped" is the
            # answer to "did my fix work?" - and it says it without inventing a culprit.
            out["text"] = (
                f"{self.total} handshake(s) refused since start, none in the last {int(WINDOW_S / 60)} minutes."
            )
            return out
        (client, path, kind), stamps = max(live.items(), key=lambda kv: len(kv[1]))
        worst = len(stamps)
        if not detailed:
            # Say THAT it is happening and how hard, never who or where.
            out["storm"] = worst >= STORM_THRESHOLD
            out["text"] = (
                f"{recent} handshake(s) refused in the last {int(WINDOW_S / 60)} minutes. "
                "Sign in to see which client and which path."
            )
            return out
        out["worst"] = {"client": client, "path": path, "kind": kind, "count": worst}
        if worst >= STORM_THRESHOLD:
            span = max(stamps) - min(stamps)
            rate = f"{worst / span * 60:.0f}/min" if span > 1 else f"{worst} in a burst"
            out["storm"] = True
            # The sentence names the cause the evidence supports (a retry loop with a credential this
            # server will not accept), the client, and the ONE action that ends it.
            cause = (
                "that page is holding an expired or wrong sign-in - reload it and sign in again"
                if kind == "credential"
                else "its Origin is not allow-listed - add it to security.cors_origins, or open "
                "the dashboard on its own address"
            )
            out["text"] = (
                f"{client} is retrying {path} and being refused ({kind}) {worst} times in the "
                f"last {int(WINDOW_S / 60)} minutes, ~{rate}. It will not recover by itself: "
                f"{cause}. Nothing is wrong with the robots."
            )
        else:
            out["storm"] = False
            out["text"] = (
                f"{recent} handshake(s) refused in the last {int(WINDOW_S / 60)} minutes "
                f"({len(live)} client(s), most from {client} on {path}, {kind}). Normal if someone is "
                f"signing in; a loop would show tens."
            )
        return out
