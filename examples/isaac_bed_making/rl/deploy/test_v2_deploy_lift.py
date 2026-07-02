#!/usr/bin/env python3
"""Off-hardware proof that the v2 (82-D, asymmetric-critic) bed-reach policy deploys through the
robotics-connect productized harness (``lib/policy_deploy.py`` + the G1 ``RobotIO`` binding).

This is the "lift v2 through robotics-connect" check from robots#3 — it runs with NO robot:
it loads the REAL exported ``policy.pt`` and the REAL ``deploy_contract_v2.json``, drives them
through ``PolicyDeploy`` against a ``MockRobotIO``, and asserts the contract + obs + action +
gains are all transfer-correct. It guards three things that silently break a real deploy:

  1. **obs parity** — the productized term-major ObsBuilder builds exactly the 82-D obs the v2
     actor expects, and (critically) WITHOUT ``base_lin_vel`` (the term v2 dropped for the real
     G1). A stale 85-D builder would feed the wrong shape.
  2. **action validity** — the real policy returns a finite, bounded 23-D action that maps onto
     all 23 named joints.
  3. **the gains footgun** — the contract must carry non-zero PD gains for every joint, else the
     productized ``run_whole`` publishes kp=0 → zero torque → the robot collapses. (The bespoke
     ``g1_bedreach_deploy.py`` hid this by hardcoding gains; the productized path reads the
     contract, so the contract must carry them.) We additionally run the whole-body ladder on the
     MockRobotIO and assert every published leg command carried the trained gains.

Run (needs torch + numpy; use the Isaac python which has both):
  cd ~/workspaces/git/IsaacLab && ./_isaac_sim/python.sh \
    ~/workspaces/git/robots-issue3/examples/isaac_bed_making/rl/deploy/test_v2_deploy_lift.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)                                   # .../isaac_bed_making/rl
RC_LIB = "/home/aifabric/workspaces/git/robotics-connect/lib"

CONTRACT = os.path.join(RL, "deploy_contract_v2.json")
DEFAULT_POLICY = os.path.join(
    RL, "logs/bed_reach_g1edu/2026-06-13_00-40-39_transfer_v2_critic/exported/policy.pt"
)
# A bedside reach target (base frame: +x fwd, +y left, +z up), inside the trained command range
# (pos_x 0.18..0.55, pos_y -0.40..0.40, pos_z -0.16..0.10): a right-side, low reach + identity quat.
CMD = [0.45, -0.25, -0.05, 1.0, 0.0, 0.0, 0.0]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# Load the robotics-connect harness by file path (unique name; never bare `import policy_deploy`).
_load("safe_stop", os.path.join(RC_LIB, "safe_stop.py"))
pd = _load("robotics_connect_policy_deploy", os.path.join(RC_LIB, "policy_deploy.py"))


class _GainRecordingIO(pd.MockRobotIO):
    """MockRobotIO that also records the per-publish gains, so we can prove the whole-body ladder
    commands non-zero PD (the zero-torque footgun)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pub_gains = []

    def publish_targets(self, targets, gains, weight=None):
        self.pub_gains.append(dict(gains))
        super().publish_targets(targets, gains, weight)


def main(policy_path: str) -> None:
    assert os.path.exists(CONTRACT), f"missing {CONTRACT}"
    assert os.path.exists(policy_path), f"missing policy {policy_path}"

    c = pd.DeployContract.load(CONTRACT)

    # 1) contract shape + the v2 deployability invariant
    assert c.n == 23, f"action dim {c.n} != 23"
    assert c.obs_total_dim == 82, f"obs total_dim {c.obs_total_dim} != 82"
    assert "base_lin_vel" not in c.obs_term_order, \
        "v2 actor must NOT observe base_lin_vel (it is unobservable on the real G1)"
    print(f"  ok  contract: {c.n} joints, {c.obs_total_dim}-D obs, term_order={c.obs_term_order}")

    # 2) the gains footgun guard — contract must carry positive PD for every action joint
    assert c.gains, "contract carries NO gains -> productized run_whole would be zero-torque"
    for n in c.action_joint_names:
        assert n in c.gains, f"no gain for {n}"
        kp, kd = c.gains[n]
        assert kp > 0 and kd > 0, f"{n} has non-positive gain {(kp, kd)}"
    print(f"  ok  gains: all {len(c.gains)} joints have positive PD "
          f"(knee={c.gains['left_knee_joint']}, waist={c.gains['waist_yaw_joint']})")

    # 3) build the productized obs from a standing state — must be 82-D, upright, no base_lin_vel
    io = _GainRecordingIO(c.action_joint_names, c.action_offset)   # seed q = the standing crouch
    dep = pd.PolicyDeploy(c, policy_path, io)                      # loads the REAL TorchScript policy
    s0 = io.read_state()
    obs, parts = dep.obs.build(s0, CMD, np.zeros(c.n))
    assert obs.shape[0] == 82, f"built obs dim {obs.shape[0]} != 82"
    assert np.allclose(parts["projected_gravity"], [0, 0, -1], atol=1e-6)
    # the v2 actor drops base_lin_vel: the built obs must be INVARIANT to it (a stale 85-D builder
    # would leak it in and the obs would change here).
    s1 = pd.RobotState(q=s0.q, dq=s0.dq, quat_wxyz=s0.quat_wxyz, gyro=s0.gyro,
                       base_lin_vel=(3.0, -2.0, 1.0))
    obs_alt, _ = dep.obs.build(s1, CMD, np.zeros(c.n))
    assert np.array_equal(obs, obs_alt), "v2 obs must be invariant to base_lin_vel (actor drops it)"
    print(f"  ok  obs builder: {obs.shape[0]}-D, gravity={np.round(parts['projected_gravity'],3).tolist()}, "
          f"invariant to base_lin_vel")

    # 4) the real policy returns a finite, bounded 23-D action mapping onto all 23 joints
    out = dep.run_offline(CMD, steps=3, log=lambda *_: None)
    a, tq = out[0]
    assert a.shape[0] == 23, f"action dim {a.shape[0]} != 23"
    assert np.isfinite(a).all(), "policy emitted non-finite action"
    assert np.abs(a).max() < 10.0, f"action unreasonably large: |a|max={np.abs(a).max()}"
    assert set(tq) == set(c.action_joint_names), "targets must cover every action joint"
    print(f"  ok  policy(offline): |a|max={np.abs(a).max():.3f}, finite, "
          f"target_q in [{min(tq.values()):+.3f}, {max(tq.values()):+.3f}]")

    # 5) run the whole-body ladder on the mock and prove the trained PD reaches the joints with the
    #    damping-first gain ramp (move-to-default ramps kp floor→nominal; kd stays nominal), and that
    #    the policy phase commands the FULL trained gain (i.e. run_whole is NOT zero-torque).
    pd._sleep = lambda _dt: None                                  # don't actually sleep in the test
    dep.run_whole(CMD, seconds=0.04, to_default_s=0.04, settle_s=0.02, blend_s=0.02,
                  cmd_ramp_s=0.02, release_s=0.04, log=lambda *_: None)
    assert io.verified and io.published, "run_whole should verify whole-body authority and publish"
    knee = [g.get("left_knee_joint") for g in io.pub_gains]
    assert all(g is not None for g in knee), "every whole-body publish must set the knee gain"
    assert all(g[1] == 4.0 for g in knee), \
        f"the nominal (damping-first) knee kd=4.0 must be held on EVERY publish: {set(map(tuple, knee))}"
    assert (150.0, 4.0) in [tuple(g) for g in knee], \
        "the policy phase must command the full trained knee PD (150.0, 4.0)"
    assert max(g[0] for g in knee) == 150.0 and min(g[0] for g in knee) == 0.0, \
        f"kp must ramp up to 150 (move-to-default/policy) and ease to 0 (gentle release): " \
        f"range [{min(g[0] for g in knee)}, {max(g[0] for g in knee)}]"
    assert io.damps > 0, "run_whole must damp on exit (SafeStop)"
    print(f"  ok  run_whole: {len(io.published)} commands, knee kp ramps 0→150→0, kd held 4.0, damped on exit")

    print("\nv2 deploy lift: 5/5 checks passed — the 82-D policy deploys cleanly through "
          "robotics-connect PolicyDeploy + the G1 RobotIO contract.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_POLICY)
