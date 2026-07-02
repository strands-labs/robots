"""On-robot deploy harness for the 23-DOF whole-body bed-reach policy (issue #3).

De-risk ladder (run in order; each stage gates the next):
  --stage offline   read-only: build the 85-D obs from live DDS, run the policy, print
                    obs + actions + resulting joint targets. NO commands published. Proves
                    obs construction + sim<->SDK joint-order parity before anything moves.
  --stage arms      run the policy at 50 Hz but apply ONLY the arm joint targets via
                    rt/arm_sdk; the legs stay on the vendor BalanceStand controller, so the
                    robot CANNOT fall. Motion-blended takeover. Validates the policy's arm
                    response to the bed target on hardware.
  --stage whole     full whole-body: all 23 joints via rt/lowcmd, vendor balance OFF.
                    REQUIRES the robot supported. (Built last, run last.)

The deploy contract (joint order, scales, defaults) is read from deploy_contract.json,
which is dumped from the EXACT training env by dump_deploy_contract.py — no guessing.

Frames: obs base_lin_vel/base_ang_vel/projected_gravity are body-frame; hand_target is a
base-frame pose (pos3 + quat4 wxyz). joint_pos obs = q - default_q in the sim action order.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

# ── Unitree SDK motor index for each joint name (standard G1 layout; verified live
#    against the EDU motor table — knee@3/9, elbow@18/25, etc.). The EDU lacks
#    13,14,20,21,27,28 (present-but-zero), which are never in the 23-joint action set.
SDK_INDEX = {
    "left_hip_pitch_joint": 0, "left_hip_roll_joint": 1, "left_hip_yaw_joint": 2,
    "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6, "right_hip_roll_joint": 7, "right_hip_yaw_joint": 8,
    "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16, "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18, "left_wrist_roll_joint": 19, "left_wrist_pitch_joint": 20, "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23, "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25, "right_wrist_roll_joint": 26, "right_wrist_pitch_joint": 27, "right_wrist_yaw_joint": 28,
}

# Arm joints only (for the fall-safe --stage arms): these go through rt/arm_sdk.
ARM_JOINT_NAMES = {
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
}

# Conservative arm joint clamps (rad) — a backstop; the rate limit is the primary smoother.
ARM_LIMITS = {
    "left_shoulder_pitch_joint": (-2.6, 2.6), "right_shoulder_pitch_joint": (-2.6, 2.6),
    "left_shoulder_roll_joint": (-1.5, 2.2), "right_shoulder_roll_joint": (-2.2, 1.5),
    "left_shoulder_yaw_joint": (-2.5, 2.5), "right_shoulder_yaw_joint": (-2.5, 2.5),
    "left_elbow_joint": (-1.0, 2.0), "right_elbow_joint": (-1.0, 2.0),
    "left_wrist_roll_joint": (-1.9, 1.9), "right_wrist_roll_joint": (-1.9, 1.9),
}
ARM_SDK_WEIGHT_IDX = 29  # G1JointIndex.kNotUsedJoint — motor_cmd[29].q is the arm_sdk blend weight


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _load_remote():
    """Load the verified G1Remote controller-abort watcher (shares the DDS channel)."""
    import importlib.util
    p = "/home/unitree/robotics-connect/unitree/g1/controller/g1_remote.py"
    spec = importlib.util.spec_from_file_location("g1_remote_dep", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.G1Remote(init_dds=False)


def quat_rotate_inverse(q_wxyz, v):
    """Rotate vector v from world into body frame given body quaternion q (wxyz)."""
    w, x, y, z = q_wxyz
    # R^T @ v ; build rotation matrix from quaternion, transpose-apply.
    vx, vy, vz = v
    # body = R^T world. Use the standard formula.
    t0 = 2.0 * (w * w + x * x) - 1.0
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])
    return R.T @ np.asarray(v, dtype=float)


class DeployState:
    """Live DDS state: joint q/dq + IMU (rt/lowstate, HG) and base vel (rt/odommodestate)."""

    def __init__(self, iface="eth0"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        ChannelFactoryInitialize(0, iface)
        self._ls = None
        self._sms = None
        ChannelSubscriber("rt/lowstate", LowState_).Init(self._on_ls, 10)
        ChannelSubscriber("rt/odommodestate", SportModeState_).Init(self._on_sms, 10)
        t = time.time()
        while (self._ls is None or self._sms is None) and time.time() - t < 4.0:
            time.sleep(0.02)
        if self._ls is None:
            raise RuntimeError("no rt/lowstate — robot on? iface correct?")
        if self._sms is None:
            raise RuntimeError("no rt/odommodestate — robot on?")

    def _on_ls(self, m):
        self._ls = m

    def _on_sms(self, m):
        self._sms = m

    def joints(self):
        ms = self._ls.motor_state
        q = np.array([float(ms[i].q) for i in range(35)])
        dq = np.array([float(ms[i].dq) for i in range(35)])
        return q, dq

    def imu(self):
        im = self._ls.imu_state
        quat = [float(x) for x in im.quaternion]      # wxyz
        gyro = [float(x) for x in im.gyroscope]        # body rad/s
        rpy = [float(x) for x in im.rpy]
        return quat, gyro, rpy

    def base_lin_vel(self):
        # SportModeState velocity is the body-frame base velocity (vx fwd, vy left).
        return np.array([float(x) for x in self._sms.velocity])


class BedReachDeploy:
    def __init__(self, contract_path, policy_path, iface="eth0"):
        with open(contract_path) as f:
            self.c = json.load(f)
        self.act_names = self.c["action"]["joint_names_in_order"]
        self.scale = np.array(self.c["action"]["scale"], dtype=float)
        self.offset = np.array(self.c["action"]["offset"], dtype=float)   # == default_q (action order)
        self.obs_jp_names = self.c["obs"]["joint_pos_names_in_order"]
        self.obs_jp_default = np.array(self.c["obs"]["joint_pos_offset_default"], dtype=float)
        assert self.obs_jp_names == self.act_names, "obs/action joint order mismatch!"
        self.sdk_idx = [SDK_INDEX[n] for n in self.act_names]
        self.n = len(self.act_names)
        self.last_action = np.zeros(self.n, dtype=float)
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        self.state = DeployState(iface)

    def build_obs(self, hand_target_pose):
        q, dq = self.state.joints()
        quat, gyro, rpy = self.state.imu()
        base_lin = self.state.base_lin_vel()
        base_ang = np.array(gyro, dtype=float)
        proj_g = quat_rotate_inverse(quat, [0.0, 0.0, -1.0])
        # joints in sim action order, via the SDK index map
        q_sim = np.array([q[i] for i in self.sdk_idx])
        dq_sim = np.array([dq[i] for i in self.sdk_idx])
        joint_pos_rel = q_sim - self.obs_jp_default
        joint_vel = dq_sim
        ht = np.asarray(hand_target_pose, dtype=float)  # 7: pos3 + quat4 wxyz
        obs = np.concatenate([base_lin, base_ang, proj_g, ht,
                              joint_pos_rel, joint_vel, self.last_action]).astype(np.float32)
        parts = {"base_lin_vel": base_lin, "base_ang_vel": base_ang, "projected_gravity": proj_g,
                 "hand_target": ht, "joint_pos_rel": joint_pos_rel, "joint_vel": joint_vel,
                 "last_action": self.last_action.copy(), "q_sim": q_sim, "proj_g": proj_g, "rpy": rpy}
        return obs, parts

    def infer(self, obs):
        with torch.inference_mode():
            a = self.policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        self.last_action = a.astype(float)
        return a

    def action_to_targets(self, action):
        return self.offset + self.scale * np.asarray(action, dtype=float)


def benign_target():
    """A near-zero reach target (pos ~ mid front, identity quat) — minimal motion for the
    offline sanity step. pos in base frame; quat identity (wxyz)."""
    return [0.30, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def stage_offline(dep, target, steps):
    print("\n=== STAGE 0: OFFLINE (read-only, NO commands) ===")
    print(f"hand_target = {np.round(target,3).tolist()}")
    for k in range(steps):
        obs, p = dep.build_obs(target)
        a = dep.infer(obs)
        tq = dep.action_to_targets(a)
        if k == 0:
            print(f"\nobs dim = {obs.shape[0]} (expect {dep.c['obs']['total_dim']})")
            print(f"projected_gravity = {np.round(p['projected_gravity'],3).tolist()}  "
                  f"(upright ~ [0,0,-1])")
            print(f"base_lin_vel = {np.round(p['base_lin_vel'],3).tolist()}  "
                  f"base_ang_vel = {np.round(p['base_ang_vel'],3).tolist()}")
            print(f"rpy(deg) = {[round(math.degrees(x),1) for x in p['rpy']]}")
            print("\nper-joint  q_sim    default  joint_pos_rel   action   target_q   |  SDK#")
            for i, nm in enumerate(dep.act_names):
                print(f"  {nm:26s} {p['q_sim'][i]:+6.3f}  {dep.obs_jp_default[i]:+6.3f}   "
                      f"{p['joint_pos_rel'][i]:+6.3f}      {a[i]:+6.3f}  {tq[i]:+6.3f}   |  {dep.sdk_idx[i]}")
            print(f"\naction: min={a.min():+.3f} max={a.max():+.3f} "
                  f"|a|max={np.abs(a).max():.3f}  finite={np.isfinite(a).all()}")
            print(f"target_q: min={tq.min():+.3f} max={tq.max():+.3f}")
        else:
            print(f"  step {k}: |a|max={np.abs(a).max():.3f} "
                  f"action[0:4]={np.round(a[:4],3).tolist()}")
    print("\n[offline] done — NO motion was commanded.")


# Per-joint PD gains MATCHED to the sim actuators (robot_cfg_g1edu) — the policy was
# trained at these, so transfer requires them. Keyed by joint-name substring.
SIM_GAINS = [
    ("_hip_", (100.0, 2.0)), ("_knee_", (150.0, 4.0)), ("_ankle_", (40.0, 2.0)),
    ("waist_yaw", (200.0, 5.0)),
    ("_shoulder_", (40.0, 10.0)), ("_elbow_", (40.0, 10.0)), ("_wrist_roll_", (40.0, 10.0)),
]


def gains_for(name):
    for key, kpkd in SIM_GAINS:
        if key in name:
            return kpkd
    return (0.0, 0.0)


def stage_whole(dep, target, run_s, settle_s=1.5, blend_s=2.5, dry=False):
    """Full whole-body: all 23 joints via rt/lowcmd, vendor balance OFF (MotionSwitcher).
    Phased motion-blended handover: hold the captured crouch -> blend to policy -> full policy.
    REQUIRES the robot supported (gantry). Abort / end -> low-level damp (gantry holds it).

    --dry builds and PRINTS the exact lowcmd it would send (no publish, no ReleaseMode):
    a read-only validation of the command construction + gains before any motion.
    """
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    MODE_PR = 0
    sdk = dep.sdk_idx
    gains = [gains_for(n) for n in dep.act_names]
    mode_machine = int(dep.state._ls.mode_machine)
    q0, _ = dep.state.joints()
    hold = {s: float(q0[s]) for s in sdk}

    low_cmd = unitree_hg_msg_dds__LowCmd_()
    crc = CRC()

    def fill(targets):
        low_cmd.mode_pr = MODE_PR
        low_cmd.mode_machine = mode_machine
        for n, s, (kp, kd) in zip(dep.act_names, sdk, gains):
            mc = low_cmd.motor_cmd[s]
            mc.mode = 1
            mc.q = float(targets[s]); mc.dq = 0.0; mc.kp = float(kp); mc.kd = float(kd); mc.tau = 0.0
        low_cmd.crc = crc.Crc(low_cmd)

    if dry:
        print("\n=== STAGE 2 DRY-CHECK (read-only — no ReleaseMode, no publish) ===")
        print(f"mode_machine={mode_machine}  mode_pr={MODE_PR}")
        obs, _ = dep.build_obs(target)
        a = dep.infer(obs)
        tq = dep.action_to_targets(a)
        pol = {s: float(tq[i]) for i, s in enumerate(sdk)}
        fill(hold)  # validate construction holding the crouch
        print(" joint                       SDK  hold_q   policy_q   kp     kd")
        for i, n in enumerate(dep.act_names):
            s = sdk[i]
            print(f"  {n:26s} {s:3d}  {hold[s]:+6.3f}  {pol[s]:+6.3f}  {gains[i][0]:6.1f} {gains[i][1]:5.1f}")
        print(f"\npolicy target_q range: [{tq.min():+.3f}, {tq.max():+.3f}]  finite={np.isfinite(tq).all()}")
        print("[dry] OK — construction valid. NO motion, vendor controller untouched.")
        return

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    remote = _load_remote(); remote.connect()

    print("\n=== STAGE 2: FULL WHOLE-BODY (rt/lowcmd, vendor balance OFF) ===")
    print("*** ROBOT MUST BE ON THE GANTRY. A transfer error = fall. ***")
    print("[abort] waiting for controller to arm (release all buttons)…")
    if not remote.wait_until_armed(8.0):
        print("[abort] NOT armed — refusing to run."); return
    reply = input("Type 'whole' to RELEASE the vendor controller and run the policy: ")
    if reply.strip().lower() != "whole":
        print("aborted (no confirmation)."); return
    if remote.aborted():
        print("[whole] abort already tripped — not running."); return

    msc = MotionSwitcherClient(); msc.SetTimeout(5.0); msc.Init()

    def check_mode():
        # CheckMode returns (code, dict) on success, (code, None) on a transient RPC miss.
        for _ in range(12):
            code, data = msc.CheckMode()
            if code == 0:
                return data or {}
            time.sleep(0.3)
        return None  # could not read the mode at all

    data = check_mode()
    if data is None:
        print("[whole] could NOT read vendor mode (RPC) — aborting, NOT taking over."); return
    name = data.get("name")
    print(f"[whole] current vendor mode: {name!r} — releasing …")
    tries = 0
    while name and tries < 12:
        msc.ReleaseMode(); time.sleep(0.5)
        data = check_mode()
        name = (data or {}).get("name")
        tries += 1
    # SAFETY: only take over if the vendor controller is confirmed released. Otherwise the
    # rt/lowcmd loop would fight the vendor balance controller.
    if name:
        print(f"[whole] FAILED to release vendor mode (still {name!r}) — aborting, NOT taking over."); return
    print("[whole] vendor controller released. rt/lowcmd now owns the joints.")
    pub = ChannelPublisher("rt/lowcmd", LowCmd_); pub.Init()

    dt = 1.0 / dep.c["control_hz"]
    def damp_out():
        print("[whole] DAMP — kp->0, compliant; gantry holds the robot.")
        q, _ = dep.state.joints()
        for k in range(int(1.0 / dt)):
            for n, s, _g in zip(dep.act_names, sdk, gains):
                mc = low_cmd.motor_cmd[s]
                mc.mode = 1; mc.q = float(q[s]); mc.dq = 0.0; mc.kp = 0.0; mc.kd = 3.0; mc.tau = 0.0
            low_cmd.mode_pr = MODE_PR; low_cmd.mode_machine = mode_machine
            low_cmd.crc = crc.Crc(low_cmd); pub.Write(low_cmd); time.sleep(dt)

    t0 = time.monotonic()
    phase = None
    while True:
        t = time.monotonic() - t0
        if remote.aborted():
            print("[whole] CONTROLLER ABORT."); damp_out(); break
        obs, _ = dep.build_obs(target)
        a = dep.infer(obs)
        tq = dep.action_to_targets(a)
        pol = {s: float(tq[i]) for i, s in enumerate(sdk)}
        if t < settle_s:
            tgt, ph = hold, "settle(hold-crouch)"
        elif t < settle_s + blend_s:
            al = (t - settle_s) / blend_s
            tgt = {s: (1 - al) * hold[s] + al * pol[s] for s in sdk}
            ph = f"blend({al:.2f})"
        elif t < settle_s + blend_s + run_s:
            tgt, ph = pol, "policy"
        else:
            print("[whole] run complete."); damp_out(); break
        fill(tgt); pub.Write(low_cmd)
        if ph != phase:
            print(f"[whole] t={t:4.1f}s -> {ph}")
            phase = ph
        time.sleep(dt)
    remote.shutdown()


def stage_arms(dep, target, seconds, vmax_rad_s=1.0, kp=40.0, kd=1.5, blend_s=1.2):
    """Fall-safe: run the policy at 50 Hz, apply ONLY arm targets via rt/arm_sdk; legs stay
    on the vendor BalanceStand controller. Motion-blended takeover + rate-limit + clamp + abort.
    """
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    remote = _load_remote()
    remote.connect()
    pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
    pub.Init()
    low_cmd = unitree_hg_msg_dds__LowCmd_()
    crc = CRC()

    arm_ai = [i for i, n in enumerate(dep.act_names) if n in ARM_JOINT_NAMES]
    arm_sdk = [dep.sdk_idx[i] for i in arm_ai]
    arm_nm = [dep.act_names[i] for i in arm_ai]
    dt = 1.0 / dep.c["control_hz"]
    max_step = vmax_rad_s * dt

    q0, _ = dep.state.joints()
    cmd = {s: float(q0[s]) for s in arm_sdk}  # commanded arm pose, starts at the live pose

    def publish(weight):
        low_cmd.motor_cmd[ARM_SDK_WEIGHT_IDX].q = float(weight)
        for s in arm_sdk:
            mc = low_cmd.motor_cmd[s]
            mc.q = float(cmd[s]); mc.dq = 0.0; mc.kp = kp; mc.kd = kd; mc.tau = 0.0
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)

    def release_and_exit(msg):
        print(f"[arms] {msg} — ramping arm_sdk weight 0, returning arms to balance controller")
        steps = int(blend_s / dt)
        for k in range(steps):
            publish(1.0 - (k + 1) / steps)
            time.sleep(dt)
        print("[arms] released. done.")

    print("\n=== STAGE 1: ARM-ONLY (fall-safe — legs on vendor BalanceStand) ===")
    print(f"hand_target={np.round(target,3).tolist()}  arms={arm_nm}")
    print("[abort] waiting for controller to arm (release all buttons)…")
    if not remote.wait_until_armed(8.0):
        print("[abort] NOT armed — refusing to run."); return
    print(f"[abort] ARMED. Any button = release arms + stop. kp={kp} kd={kd} vmax={vmax_rad_s} rad/s")

    # 1) Blend IN: ramp weight 0→1 while holding the live arm pose (no jerk).
    n_in = int(blend_s / dt)
    for k in range(n_in):
        if remote.aborted():
            return release_and_exit("abort during blend-in")
        publish((k + 1) / n_in)
        time.sleep(dt)

    # 2) Policy loop: arms track the (rate-limited, clamped) policy targets.
    t_end = time.monotonic() + seconds
    step = 0
    while time.monotonic() < t_end:
        if remote.aborted():
            return release_and_exit("controller abort")
        obs, _ = dep.build_obs(target)
        a = dep.infer(obs)
        tq = dep.action_to_targets(a)
        for ai, s, nm in zip(arm_ai, arm_sdk, arm_nm):
            lo, hi = ARM_LIMITS[nm]
            goal = _clip(float(tq[ai]), lo, hi)
            cmd[s] += _clip(goal - cmd[s], -max_step, max_step)  # rate-limited blend
        publish(1.0)
        if step % 25 == 0:
            print(f"  [arms] t={time.monotonic()-(t_end-seconds):4.1f}s |a|max={np.abs(a).max():.2f} "
                  f"Lsh_p={cmd[15]:+.2f} Rsh_p={cmd[22]:+.2f} Lelb={cmd[18]:+.2f} Relb={cmd[25]:+.2f}")
        step += 1
        time.sleep(dt)

    # 3) Blend OUT: hand the arms back to the balance controller.
    release_and_exit("run complete")
    remote.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["offline", "arms", "whole"], default="offline")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--contract", default=os.path.join(os.path.dirname(__file__), "deploy_contract.json"))
    ap.add_argument("--policy", required=True, help="path to exported policy.pt (TorchScript)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--target", type=float, nargs=7, default=None,
                    help="hand_target pose: x y z qw qx qy qz (base frame)")
    ap.add_argument("--seconds", type=float, default=5.0, help="arms/whole run length")
    ap.add_argument("--vmax", type=float, default=1.0, help="arm joint rate limit (rad/s)")
    ap.add_argument("--kp", type=float, default=40.0)
    ap.add_argument("--kd", type=float, default=1.5)
    ap.add_argument("--dry", action="store_true", help="whole-body: build+print the lowcmd, no motion")
    ap.add_argument("--settle", type=float, default=1.5, help="whole: hold-crouch seconds")
    ap.add_argument("--blend", type=float, default=2.5, help="whole: blend-to-policy seconds")
    args = ap.parse_args()

    dep = BedReachDeploy(args.contract, args.policy, iface=args.iface)
    target = args.target if args.target is not None else benign_target()

    if args.stage == "offline":
        stage_offline(dep, target, args.steps)
    elif args.stage == "arms":
        stage_arms(dep, target, args.seconds, vmax_rad_s=args.vmax, kp=args.kp, kd=args.kd)
    elif args.stage == "whole":
        stage_whole(dep, target, args.seconds, settle_s=args.settle, blend_s=args.blend, dry=args.dry)


if __name__ == "__main__":
    main()
