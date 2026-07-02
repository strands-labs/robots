"""Unit tests for DrawResistanceMonitor — the waist-torque anchored-stall signal (2026-06-19).

Pure logic, no robot/torch. Run:  python3 test_bed_fail_detect.py

Covers the firm-grasp finding: an anchored pull can leave arm follow/tau FREE-like (the load dumps
into the torso twist), so the verdict must trip on waist-torque DEVIATION — while a free pull (low on
every signal) and an empty/air draw must NOT trip. Also covers DOF-awareness (max over 1 or 3 waist
joints), baseline-relative measurement, graceful degrade when the waist channel is absent, and that the
onset transient inside the warmup window does not trip.
"""
from bed_fail_detect import DrawResistanceMonitor

DT = 0.02  # 50 Hz; defaults: warmup 0.6s = 30 ticks, sustain 0.5s = 25 ticks


class FakeState:
    def __init__(self, q, dq, quat=(1.0, 0.0, 0.0, 0.0)):
        self.q = q
        self.dq = dq
        self.quat_wxyz = quat


def run(monitor, *, ticks, follow=0.0, arm_tau=0.0, waist_taus=None, waist_qs=None,
        waist_base_taus=None, waist_base_qs=None, warmup_spike_follow=None):
    """Drive `ticks` updates. `follow` is injected as a cmd/actual gap on the one arm joint;
    `warmup_spike_follow` (if set) applies only during the warmup window (tick<30). The waist signal is
    a DEVIATION from the draw-start baseline, so tests pass a low `waist_base_*` for tick 0 (captured as
    the baseline) and the elevated `waist_*` for the rest (the held-fast torque rise)."""
    cmd = {"a": 0.0}
    for t in range(ticks):
        f = follow
        if warmup_spike_follow is not None and t < 30:
            f = warmup_spike_follow
        st = FakeState(q={"a": -f}, dq={"a": 0.0})            # follow = |cmd - q| = f
        tau = {"a": arm_tau}
        wt = waist_base_taus if (t == 0 and waist_base_taus is not None) else waist_taus
        wq = waist_base_qs if (t == 0 and waist_base_qs is not None) else waist_qs
        monitor.update(st, cmd, tau, waist_taus=wt, waist_qs=wq)
    return monitor.tripped


def mk(**kw):
    m = DrawResistanceMonitor(["a"], DT, **kw)
    m.start(FakeState(q={"a": 0.0}, dq={"a": 0.0}))
    return m


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    assert cond, name


print("waist-signal voting:")
# Anchored-by-twist: arm signals free-like, but waist torque deviates hard & sustained → TRIP.
m = mk(waist_tau_thresh=6.0)
check("anchored twist (waist_dtau~12 sustained) trips",
      run(m, ticks=70, follow=0.12, arm_tau=4.5,
          waist_base_taus={"waist_yaw_joint": 1.0}, waist_taus={"waist_yaw_joint": 13.0}) is True)
check("  ...and peak_waist_dtau recorded (~12 = 13-baseline 1)", abs(m.peak_waist_dtau - 12.0) < 1e-6)

# Free pull: low on every signal → no trip.
m = mk(waist_tau_thresh=6.0)
check("free pull (all low) does NOT trip",
      run(m, ticks=70, follow=0.12, arm_tau=4.5, waist_taus={"waist_yaw_joint": 1.0}) is False)

# Empty/air draw: no waist load at all → no trip.
m = mk(waist_tau_thresh=6.0)
check("empty/air draw does NOT trip",
      run(m, ticks=70, follow=0.05, arm_tau=2.0, waist_taus={"waist_yaw_joint": 1.2}) is False)

print("baseline-relative (deviation, not absolute):")
# Big CONSTANT waist torque offset but no change from baseline → no trip (it's the deviation that counts).
m = mk(waist_tau_thresh=6.0)
check("constant high waist torque (no deviation) does NOT trip",
      run(m, ticks=70, waist_taus={"waist_yaw_joint": 40.0}) is False)

print("measure-only (thresh None):")
# Waist deviates hugely but thresh None → waist does not vote; arm signals also low → no trip.
m = mk(waist_tau_thresh=None)
check("measure-only: huge waist_dtau alone does NOT trip",
      run(m, ticks=70, follow=0.12, arm_tau=4.5,
          waist_base_taus={"waist_yaw_joint": 1.0}, waist_taus={"waist_yaw_joint": 99.0}) is False)
check("  ...but peak_waist_dtau still measured (~98)", abs(m.peak_waist_dtau - 98.0) < 1e-6)
# Secondary OR still alive in measure-only: a real arm-lag stall (high follow) trips on follow.
m = mk(waist_tau_thresh=None, follow_thresh_rad=0.28)
check("measure-only: arm-lag (follow 0.4 sustained) still trips on follow",
      run(m, ticks=70, follow=0.4) is True)

print("DOF-awareness (max over waist joints):")
# 3-DOF: only waist_roll spikes — max-over-joints must catch it.
m = mk(waist_tau_thresh=6.0)
trip = run(m, ticks=70, follow=0.1, arm_tau=4.0,
           waist_base_taus={"waist_yaw_joint": 1.0, "waist_roll_joint": 1.0, "waist_pitch_joint": 1.0},
           waist_taus={"waist_yaw_joint": 1.0, "waist_roll_joint": 15.0, "waist_pitch_joint": 1.0})
check("3-DOF: a spike on waist_roll alone trips (max over joints)", trip is True)
# 1-DOF default set: only waist_yaw present, quiet → no trip.
m = mk(waist_tau_thresh=6.0)
check("1-DOF: quiet waist_yaw does NOT trip",
      run(m, ticks=70, follow=0.12, arm_tau=4.5, waist_taus={"waist_yaw_joint": 1.0}) is False)

print("graceful degrade + warmup:")
# No waist data at all (firmware/absent) → waist contributes 0; free arm signals → no trip.
m = mk(waist_tau_thresh=6.0)
check("no waist channel (empty) does NOT trip on a free pull",
      run(m, ticks=70, follow=0.12, arm_tau=4.5, waist_taus={}) is False)
# Onset transient confined to the warmup window must not trip even if it exceeds a threshold.
m = mk(waist_tau_thresh=6.0, follow_thresh_rad=0.28)
check("warmup-only follow spike (0.5) does NOT trip",
      run(m, ticks=70, follow=0.0, warmup_spike_follow=0.5) is False)
# Position deflection fallback recorded (deg) from waist_qs deviation.
m = mk(waist_tau_thresh=6.0)
run(m, ticks=70, waist_taus={"waist_yaw_joint": 1.0},
    waist_base_qs={"waist_yaw_joint": 0.0}, waist_qs={"waist_yaw_joint": 0.4363})  # 0.4363 rad ~ 25 deg
check("waist position deflection logged (~25 deg)", abs(m.peak_waist_dyaw - 25.0) < 0.5)

print("\nALL TESTS PASSED")
