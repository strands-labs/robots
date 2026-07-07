"""Pull-failure detector: did the draw move a FREE sheet, or stall against an ANCHORED load
(the classic failure = the claw caught the fixed mattress cover, not just the sheet)?

Two robust signals, both from the existing public RobotState — no extra telemetry, no harness change:

  * follow_err — max |commanded - actual| over the arm joints (rad). Under the PD arm-overlay a joint
    held back by an immovable load lags its command (torque ≈ Kp·follow_err), so a stalled arm shows a
    large, SUSTAINED following error while a free sweep tracks closely. This is our torque proxy.
  * waist_yaw torque - the PRIMARY signal in the firm-grasp era (2026-06-19). A firm anchored grasp no
    longer lags the arm; the reaction moment dumps into an upper-torso twist that the vendor balancer
    resists by driving the waist_yaw motor hard, so an anchored load shows a large waist_yaw tau
    deviation. See the FailDetector class docstring for the full rationale.
  * arm tau_est - direct arm motor torque, a SECONDARY verdict signal. It reads 0/garbage on some G1
    EDU firmware (cf. power_v=0.0), so its threshold is loose.

Also computed but LOGGED-ONLY (not in the verdict): base IMU yaw_drift (the balancer holds the pelvis
still while only the torso twists, so the base misses the load) and per-tick dq.

Verdict = FAILED if the waist_yaw torque deviation (PRIMARY, once its threshold is calibrated) OR
follow_err OR arm tau_est stays above threshold for >= sustain_s (sustained, not a transient spike at
the start of the sweep). Thresholds are first-pass - calibrate against one free pull and one anchored
pull, then tighten and enable --abort-on-stall.
"""
from __future__ import annotations

import math


def yaw_deg_from_quat(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny, cosy))


def _ang_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


class DrawResistanceMonitor:
    """Watch the draw and decide free-vs-anchored. Call start() at draw start, update() every tick.

    PRIMARY signal (2026-06-19, firm-grasp era): waist_yaw TORQUE. With the firm grasp, an anchored
    pull no longer lags the ARM (follow stays free-like ~0.12 even on a hard stall) — instead the
    reaction moment dumps into an upper-torso TWIST (~25° observed, sheet held fast after ~8"), which
    the vendor balancer resists by driving the waist_yaw motor hard. The arm joints still track their
    commands (low follow), so the load is invisible to follow_err/arm-tau; it shows up as a big
    waist_yaw tau_est DEVIATION (and joint deflection). The base IMU misses it too — it reads the
    PELVIS, which the balancer holds still while only the torso (waist_yaw joint) twists. So the waist
    torque is the discriminator the operator pointed us to. follow/arm-tau are kept as a SECONDARY OR
    (they did catch the 2026-06-18 arm-lag anchored case), but their thresholds are now loose because
    a firm FREE pull's onset transient nicks the old 0.28 follow line.
    """

    def __init__(self, arm_joints, dt, *, follow_thresh_rad=0.28, tau_thresh=11.0, yaw_thresh_deg=6.0,
                 waist_tau_thresh=None, sustain_s=0.5, warmup_s=0.6):
        self._arm = list(arm_joints)
        self._dt = dt
        self.follow_thresh = follow_thresh_rad
        self.tau_thresh = tau_thresh
        self.yaw_thresh = yaw_thresh_deg
        self.waist_tau_thresh = waist_tau_thresh        # None = measure-only (log, don't vote)
        self._sustain_ticks = max(1, int(sustain_s / dt))
        self._warmup_ticks = max(0, int(warmup_s / dt))   # ignore the draw's onset acceleration transient
        self._yaw0 = None
        self._waist_tau0 = {}          # per-joint waist tau/pos at draw start — signals are DEVIATIONS from here
        self._waist_q0 = {}
        self._tick = 0
        self._over = 0
        self.peak_follow = 0.0          # steady-state peaks (post-warmup) — what the verdict uses
        self.peak_yaw = 0.0
        self.peak_tau = 0.0
        self.peak_waist_dtau = 0.0      # peak max-over-waist-joints |tau_est - baseline| (PRIMARY anchored signal)
        self.peak_waist_dyaw = 0.0      # peak max-over-waist-joints |joint deflection| (deg) — fallback if tau dead
        self.min_dq = float("inf")
        self.tripped = False

    def start(self, state) -> None:
        self._yaw0 = yaw_deg_from_quat(state.quat_wxyz)
        self._waist_tau0 = {}
        self._waist_q0 = {}
        self._tick = 0
        self._over = 0
        self.peak_follow = self.peak_yaw = self.peak_tau = 0.0
        self.peak_waist_dtau = self.peak_waist_dyaw = 0.0
        self.min_dq = float("inf")
        self.tripped = False

    def update(self, state, cmdq, tau=None, waist_taus=None, waist_qs=None) -> dict:
        self._tick += 1
        armed = self._tick > self._warmup_ticks            # past the startup transient?
        follow = max(abs(cmdq[j] - state.q[j]) for j in self._arm)
        yaw = _ang_diff_deg(yaw_deg_from_quat(state.quat_wxyz), self._yaw0) if self._yaw0 is not None else 0.0
        dq = sum(abs(state.dq[j]) for j in self._arm) / len(self._arm)   # mean |arm velocity| (logged only)
        tau_max = max(abs(v) for v in tau.values()) if tau else 0.0
        # Waist torque/deflection as DEVIATION from the draw-start baseline (robust to a resting posture
        # offset), taken as the MAX over the configured waist joints (1 on a 23-DOF G1 = waist_yaw; 3 on
        # a 29-DOF G1). Empty dict (firmware lacks the channel / no waist joints) → degrades to 0.
        waist_dtau = 0.0
        if waist_taus:
            for n, v in waist_taus.items():
                self._waist_tau0.setdefault(n, v)
                waist_dtau = max(waist_dtau, abs(v - self._waist_tau0[n]))
        waist_dyaw = 0.0
        if waist_qs:
            for n, v in waist_qs.items():
                self._waist_q0.setdefault(n, v)
                waist_dyaw = max(waist_dyaw, math.degrees(abs(v - self._waist_q0[n])))
        if armed:                                          # peaks for the VERDICT exclude the transient
            self.peak_follow = max(self.peak_follow, follow)
            self.peak_yaw = max(self.peak_yaw, yaw)
            self.peak_tau = max(self.peak_tau, tau_max)
            self.peak_waist_dtau = max(self.peak_waist_dtau, waist_dtau)
            self.peak_waist_dyaw = max(self.peak_waist_dyaw, waist_dyaw)
            self.min_dq = min(self.min_dq, dq)
        # ANCHORED: PRIMARY = waist_yaw torque deviation (the torso twist the balancer resists); the
        # arm signals (follow_err, arm-tau) are a SECONDARY OR for the arm-lag mode. waist_tau_thresh
        # is None until calibrated → measure-only (logged, not voted). dq/base-yaw are dead (logged).
        waist_over = (self.waist_tau_thresh is not None and waist_dtau > self.waist_tau_thresh)
        over = armed and (follow > self.follow_thresh or tau_max > self.tau_thresh or waist_over)
        self._over = self._over + 1 if over else 0
        if self._over >= self._sustain_ticks:
            self.tripped = True
        return {"follow": follow, "yaw": yaw, "dq": dq, "tau": tau_max,
                "waist_dtau": waist_dtau, "waist_dyaw": waist_dyaw,
                "armed": armed, "over": over, "tripped": self.tripped}

    def summary(self) -> str:
        min_dq = self.min_dq if self.min_dq != float("inf") else 0.0
        verdict = "ANCHORED LOAD — pull FAILED" if self.tripped else "free pull — OK"
        wthr = "measure-only" if self.waist_tau_thresh is None else f"thr {self.waist_tau_thresh}"
        return (f"peak_waist_dtau={self.peak_waist_dtau:.2f} ({wthr}) "
                f"peak_waist_dyaw={self.peak_waist_dyaw:.1f}deg "
                f"peak_follow={self.peak_follow:.3f}rad (thr {self.follow_thresh}) "
                f"peak_tau={self.peak_tau:.2f} (thr {self.tau_thresh}) "
                f"[min_dq={min_dq:.3f} base_yaw={self.peak_yaw:.1f}deg — logged only]  ->  {verdict}")


def read_arm_tau(io, arm_joints):
    """Best-effort estimated joint torque from the latest lowstate (None if unavailable). Private
    RobotIO internals; wrapped so a firmware that lacks tau_est just degrades to the two main signals."""
    try:
        ms = io._ls.motor_state
        sdk = io._sdk
        return {n: float(ms[sdk[n]].tau_est) for n in arm_joints}
    except Exception:
        return None


# G1 motor table (the EDU is indexed like the 29-DOF G1): legs 0–11, then the WAIST. A 23-DOF G1 has
# ONLY waist_yaw (idx 12); a 29-DOF G1 adds waist_roll (13) + waist_pitch (14). Index 12 is waist_yaw
# on BOTH, so it is always safe to read; 13/14 are present ONLY on a 29-DOF (reading them on a 23-DOF
# returns dead/garbage slots — never enable them there).
WAIST_SDK_INDEX = {"waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14}
# Monitored set by DOF. DEFAULT = waist_yaw only: a lateral headward draw twists the torso in YAW,
# which the waist_yaw joint carries on BOTH variants, so this is correct and safe everywhere and never
# touches a 29-DOF-only slot. On a 29-DOF G1, use WAIST_JOINTS[3] to ALSO catch off-axis (roll/pitch)
# load. (Index 12 == waist_yaw regardless, so the 23-DOF default transfers unchanged to a 29-DOF.)
WAIST_JOINTS = {1: ["waist_yaw_joint"],
                3: ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]}


def read_waist(io, waist_joints=("waist_yaw_joint",)):
    """Best-effort waist tau_est + position per configured waist joint (two dicts name→value, or
    ({},{}) if the lowstate is unavailable). The anchored-load torso TWIST that the arm joints and the
    base IMU both miss shows up HERE: the vendor balancer drives the waist motor hard to resist it.
    DOF-aware — reads ONLY the joints you pass, so the 23-DOF default (waist_yaw) never indexes a
    29-DOF-only slot. Pass WAIST_JOINTS[3] on a 29-DOF G1 to include waist_roll/pitch."""
    try:
        ms = io._ls.motor_state
        taus = {n: float(ms[WAIST_SDK_INDEX[n]].tau_est) for n in waist_joints}
        qs = {n: float(ms[WAIST_SDK_INDEX[n]].q) for n in waist_joints}
        return taus, qs
    except Exception:
        return {}, {}
