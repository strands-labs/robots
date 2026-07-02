"""Grasp confirmation: did the claw close on FABRIC, or on NOTHING (an empty grab)?

This is the pull-failure detector's blind spot and the live-demo killer. `bed_fail_detect.py`
catches an ANCHORED stall ("pulled something that won't move" — grabbed the fixed cover). It does
NOT catch the opposite failure: the claw closes on AIR, the arm "draws" nothing, following-error and
tau read perfectly FREE, and the orchestrator reports SUCCESS while the sheet never moved. A confident
false-success is worse for the Device Connect story than an honest stall — so the grasp must confirm
fabric-in-hand BEFORE the draw, and emit an explicit empty-grab signal the orchestrator escalates on
(exactly like /tmp/pull_failed).

PEAK ≠ CAPTURE (hardware-observed 2026-06-19, anchored taut sheet). The first cut keyed the verdict on
the PEAK touch-force rise across the close — and that produced a real false-success: the claw brushed a
taut, tucked sheet (a finger's force rose to 10, over the threshold), but with no slack to pinch the
sheet SLIPPED OFF, force collapsed to 0 by full close, the hand drew on air and read FREE. Peak fired
"FABRIC"; the hand was empty. The fix: the verdict keys on fabric STILL LOADED while HOLDING the closed
grip (a sustained reading over a short hold), not a transient peak. A peak that decays to ~0 by the held
close = touched-but-didn't-capture = EMPTY. Drive the hold with `begin_hold()` + a few `update()` calls
at the held closed position (bed_grip_v1.py does this after the close); the sustained value is the median
over that held window, so a single dropout can't flip it. Peak is kept for diagnostics (peak-high +
sustained-low is the slip signature).

Signals — both already in the brainco_bridge `get` telemetry, no harness change:

  * touch_force rise — per-finger contact force above the open-claw baseline (u16). Fabric in the claw
    loads the fingertip; air does not. CAVEAT: thin fabric reads LOW — observed grabs are ~9-39 u16.
  * proximity deviation — per-finger proximity change from the open-claw baseline (u16). Fabric in the
    claw sits in front of the fingertip sensors. Sign is firmware-dependent so we key on ABSOLUTE
    deviation. (DEAD on this G1 — the bridge never publishes left_proximity — so the verdict runs
    touch-only here; prox stays a first-pass guess for any G1 whose bridge does publish it.)

Verdict = FABRIC PRESENT if (SUSTAINED touch rise >= force_rise) OR (sustained proximity deviation >=
prox_dev) on at least `fingers_needed` fingers. Otherwise EMPTY GRAB.

CALIBRATION STATUS (this G1): the TOUCH threshold (force_rise) is hardware-calibrated 2026-06-19 to 6
(empty floor ≤1 over n=7 closes; single-layer sheet 9-39 over n=6, min 9; 2-layer 81). Re-validate the
SUSTAINED verdict on hardware (empty / real captured grab / brush-slip). The original procedure (capture
~5 empty + ~5 fabric closes, set the threshold between the clusters) is automated in
bed_grasp_calibrate.py. Re-run it if the fabric or hand changes.

Bias on purpose: when the two signals disagree or proximity reads garbage, prefer to DECLARE EMPTY
(escalate to the human) over claiming a grab — a false "I have the sheet" is the failure we are here to
kill. (Proximity degrades to touch-only if it reads all-zero/garbage, mirroring read_arm_tau.)
"""
from __future__ import annotations

import statistics
from collections import deque


def _as_list(v, n):
    """Coerce a bridge sensor field to a length-n list of floats (missing/short → zeros)."""
    out = [0.0] * n
    if not v:
        return out
    for i in range(min(n, len(v))):
        try:
            out[i] = float(v[i])
        except (TypeError, ValueError):
            out[i] = 0.0
    return out


class GraspConfirmMonitor:
    """Capture an open-claw baseline, watch the close, decide fabric-vs-empty on the HELD grip.

    Usage: baseline(reading) once with the claw open (before the digits close), update(reading) each
    sensor poll during the close, then — while HOLDING the closed grip — begin_hold() once and a few
    more update(reading) calls so the verdict keys on fabric still loaded (not a transient peak). Read
    verdict()/summary() after.
    """

    def __init__(self, *, n_fingers=5, force_rise=4.0, prox_dev=150.0, fingers_needed=1, settle_samples=4,
                 use_proximity=False, fabric_indices=None):
        self.n = n_fingers
        # Which touch indices VOTE for fabric. Touch order is [thumb, index, middle, ring, pinky]; the
        # THUMB pad (index 0) is EXCLUDED by default: in the thumb-clamp-last close the thumb presses
        # straight onto the closed fingers, so its pad reads high (~600) on an EMPTY close (self-contact,
        # not fabric — observed 2026-06-19). Fabric is sensed on the 4 finger pads (1-4), where every real
        # grab registered. Pass fabric_indices explicitly to override.
        self.fabric_indices = list(range(1, n_fingers)) if fabric_indices is None else list(fabric_indices)
        # force_rise HARDWARE-CALIBRATED 2026-06-19 (touch-only; proximity dead on this G1). With the FIXED
        # sequenced close, real grabs read STRONG on the finger pads (index ~50-110); the empty/slip floor
        # is ≤2.5. 4 sits above that floor and well below a real grab — and is the single source of truth
        # with bed_grip_v1.py's --confirm-force-rise default. The HELD-grip verdict + mid-draw monitor are
        # the real backstops.
        self.force_rise = force_rise            # u16 SUSTAINED touch rise that counts as fabric (LOW: soft fabric)
        self.prox_dev = prox_dev                # u16 |proximity - baseline| that counts as something-in-claw
        self.fingers_needed = fingers_needed
        # Proximity is OPT-IN (default OFF). The canonical bridge DOES publish left_proximity, but closing
        # the claw swings the fingertips toward the palm/each other, so |prox - open_baseline| spikes
        # ~30000 on an EMPTY close (geometry, not fabric) → false-FABRIC. It needs a closed-empty baseline
        # before it can vote; until that calibration exists the verdict is touch-only. (2026-06-19)
        self.use_proximity = use_proximity
        self.base_force = [0.0] * n_fingers
        self.base_prox = [0.0] * n_fingers
        self.peak_force_rise = [0.0] * n_fingers
        self.peak_prox_dev = [0.0] * n_fingers
        # SUSTAINED window: the recent force/prox rises, dominated by the HELD closed grip (begin_hold
        # clears it so dynamic-close transients don't leak in). median over it = the verdict basis.
        self._recent_force_rise = deque(maxlen=settle_samples)
        self._recent_prox_dev = deque(maxlen=settle_samples)
        self._prox_seen_nonzero = False         # did proximity ever report a usable (non-zero) value?
        self._based = False

    @staticmethod
    def _force(reading):
        return reading.get("left_touch_force") if isinstance(reading, dict) else None

    @staticmethod
    def _prox(reading):
        return reading.get("left_proximity") if isinstance(reading, dict) else None

    def baseline(self, reading) -> None:
        self.base_force = _as_list(self._force(reading), self.n)
        self.base_prox = _as_list(self._prox(reading), self.n)
        self._based = True

    def begin_hold(self) -> None:
        """Mark the start of the HELD-grip phase: drop the dynamic-close window so the sustained verdict
        is computed only from readings taken while holding the closed grip."""
        self._recent_force_rise.clear()
        self._recent_prox_dev.clear()

    def update(self, reading) -> dict:
        if not self._based:                     # be forgiving: first update doubles as the baseline
            self.baseline(reading)
        force = _as_list(self._force(reading), self.n)
        prox = _as_list(self._prox(reading), self.n)
        if any(p != 0.0 for p in prox):
            self._prox_seen_nonzero = True
        fr = [force[i] - self.base_force[i] for i in range(self.n)]
        pd = [abs(prox[i] - self.base_prox[i]) for i in range(self.n)]
        self._recent_force_rise.append(fr)
        self._recent_prox_dev.append(pd)
        for i in range(self.n):
            self.peak_force_rise[i] = max(self.peak_force_rise[i], fr[i])
            self.peak_prox_dev[i] = max(self.peak_prox_dev[i], pd[i])
        return {"force": force, "prox": prox, "force_rise": fr, "prox_dev": pd}

    def _sustained(self, recent):
        """Elementwise median over the recent (held) window — robust to a single dropout."""
        if not recent:
            return [0.0] * self.n
        return [float(statistics.median(col)) for col in zip(*recent)]

    @property
    def sustained_force_rise(self):
        return self._sustained(self._recent_force_rise)

    @property
    def sustained_prox_dev(self):
        return self._sustained(self._recent_prox_dev)

    def _touch_fingers(self):
        s = self.sustained_force_rise
        return [i for i in self.fabric_indices if s[i] >= self.force_rise]

    def _prox_fingers(self):
        if not self.use_proximity or not self._prox_seen_nonzero:   # OFF by default (false-fires on close
            return []                                               # geometry) / dead → don't let it vote
        s = self.sustained_prox_dev
        return [i for i in self.fabric_indices if s[i] >= self.prox_dev]

    def fabric_present(self) -> bool:
        fingers = set(self._touch_fingers()) | set(self._prox_fingers())
        return len(fingers) >= self.fingers_needed

    def _slipped(self) -> bool:
        """Peak fired but the sustained held grip did not → touched-but-didn't-capture (a slip)."""
        s = self.sustained_force_rise
        peak_fired = any(self.peak_force_rise[i] >= self.force_rise for i in self.fabric_indices)
        held = any(s[i] >= self.force_rise for i in self.fabric_indices)
        return peak_fired and not held

    def verdict(self) -> dict:
        return {
            "fabric_present": self.fabric_present(),
            "touch_fingers": self._touch_fingers(),
            "prox_fingers": self._prox_fingers(),
            "sustained_force_rise": [round(v, 1) for v in self.sustained_force_rise],
            "sustained_prox_dev": [round(v, 1) for v in self.sustained_prox_dev],
            "peak_force_rise": [round(v, 1) for v in self.peak_force_rise],
            "peak_prox_dev": [round(v, 1) for v in self.peak_prox_dev],
            "prox_usable": self._prox_seen_nonzero,
            "slipped": self._slipped(),
        }

    def summary(self) -> str:
        v = self.verdict()
        prox_note = "" if (self.use_proximity and v["prox_usable"]) else " [touch-only verdict]"
        if v["fabric_present"]:
            result = "FABRIC in hand"
        elif v["slipped"]:
            result = "EMPTY GRAB — touched fabric but it SLIPPED (peak fired, held grip empty)"
        else:
            result = "EMPTY GRAB — closed on nothing"
        return (f"sustained_force_rise={v['sustained_force_rise']} (thr {self.force_rise}, fired "
                f"{v['touch_fingers']}) peak_force_rise={v['peak_force_rise']} "
                f"sustained_prox_dev={v['sustained_prox_dev']} (thr {self.prox_dev}, fired "
                f"{v['prox_fingers']}){prox_note}  ->  {result}")
