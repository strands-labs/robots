"""Is the arm being recorded actually MOVING, or is the dataset a still life?"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = ["EPSILON_DEG", "MIN_SAMPLES", "WINDOW_S", "joint_positions", "prune", "motion_verdict"]

# : How far a joint must travel inside the window to count as movement, in DEGREES. : A still
# but powered SO-101 jitters by ~0.1-0.3 deg on this rig; hand guiding moves : tens of
# degrees. 0.5 sits between those with room on both sides.
EPSILON_DEG = 0.5

#: No verdict before this many samples: two frames that happen to match are not evidence,
#: and the first tick of an episode has nothing to compare against at all.
MIN_SAMPLES = 10

# : How far back to look.
WINDOW_S = 8.0


def joint_positions(obs: Mapping[str, Any] | None) -> dict[str, float]:
    """The finite numeric joint positions of one observation."""
    out: dict[str, float] = {}
    for key, value in (obs or {}).items():
        if not isinstance(key, str) or not key.endswith(".pos"):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        v = float(value)
        if math.isfinite(v):
            out[key] = v
    return out


def prune(
    samples: Sequence[tuple[float, Mapping[str, float]]], now: float, window_s: float = WINDOW_S
) -> list[tuple[float, Mapping[str, float]]]:
    """The samples still inside the window, oldest first."""
    if not samples:
        return []
    cutoff = now - max(0.0, float(window_s))
    return [(t, p) for t, p in samples if t >= cutoff]


def _travel(samples: Iterable[tuple[float, Mapping[str, float]]]) -> tuple[str | None, float]:
    """The joint that moved most inside these samples, and by how much."""
    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    for _t, positions in samples:
        for joint, raw in positions.items():
            # Defensive on purpose: the caller extracts positions with ``joint_positions``, but a future
            # caller handing over a whole observation would otherwise crash the record loop on comparing
            # two camera frames - and a crashed tick is a lost frame, which is worse than a missing
            # notice.
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            value = float(raw)
            if not math.isfinite(value):
                continue
            if joint not in lo or value < lo[joint]:
                lo[joint] = value
            if joint not in hi or value > hi[joint]:
                hi[joint] = value
    best_joint: str | None = None
    best = 0.0
    for joint, high in hi.items():
        span = high - lo[joint]
        if best_joint is None or span > best:
            best_joint, best = joint, span
    return best_joint, best


def motion_verdict(
    samples: Sequence[tuple[float, Mapping[str, float]]],
    *,
    now: float,
    frames: int | None = None,
    window_s: float = WINDOW_S,
    epsilon_deg: float = EPSILON_DEG,
    min_samples: int = MIN_SAMPLES,
) -> dict[str, Any] | None:
    """What to tell the operator about the follower's motion, or ``None``."""
    # COVERAGE is judged on the whole history handed over, TRAVEL only on the window.
    if not samples:
        return None
    reach_s = now - samples[0][0]
    if reach_s < window_s:
        # Less history than the window: a hold this short is ordinary.
        return None
    window = prune(samples, now, window_s)
    if len(window) < max(2, int(min_samples)):
        return None
    span_s = now - window[0][0]
    # A stream that STOPPED is a different defect (frames are not being recorded at all, and the
    # fps panel already says so).
    if now - window[-1][0] > window_s / 2:
        return None
    joint, travelled = _travel(window)
    if joint is None:
        return None  # no joint positions in any sample - not our subject
    if travelled >= epsilon_deg:
        return None

    counted = f"{frames} frames of " if frames else ""
    return {
        "still": True,
        "seconds": round(span_s, 1),
        "samples": len(window),
        "max_travel_deg": round(travelled, 3),
        "quietest_joint": joint,
        "message": (
            f"the follower has not moved for {span_s:.0f}s - {counted}one unchanging pose "
            f"(largest joint travel {travelled:.2f} deg). If that is deliberate, ignore this. "
            "If it is not: a Feetech bus still REPORTS positions from the USB logic rail "
            "when the 12V supply is off, so a tripped supply looks exactly like this - "
            "valid numbers, full frame rate, an arm that never moves. Check the follower's "
            "power before collecting more, and redo this episode."
        ),
    }
