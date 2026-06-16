#!/usr/bin/env python3
"""Torque-control deploy harness for WBCPolicy on the Unitree G1.

Reproduces NVIDIA's GR00T Whole-Body-Control reference loop
(``decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py``) closely enough to
drive the real ``GR00T-WholeBodyControl-{Balance,Walk}.onnx`` weights and watch
the G1 try to walk in MuJoCo:

    every physics tick:
        tau = (target_dof_pos - q) * kp + (0 - dq) * kd      # PD -> torque
        data.ctrl[:15] = tau                                  # leg+waist motors
        arms held at default with a stiff PD
        mj_step
        every control_decimation (4) ticks:
            obs    = build the 86-dim frame (whole-body qj/dqj + base IMU + cmd)
            action = WBCPolicy ONNX (Balance if standing, Walk if moving)
            target_dof_pos = action * action_scale + default_angles

The crucial difference from ``sim.run_policy`` is that this applies the policy's
position targets through the upstream TORQUE PD law on a TORQUE-actuated model
(``policy.compute_torques(...)``), not as position-servo ctrl. That is what a
real deployment does, and what a stable gait needs.

This harness is intentionally standalone (not a Simulation AgentTool action) and
self-contained: it converts the MuJoCo Menagerie G1's position-servo actuators
to torque motors in-process, so it needs no extra mesh/XML download beyond the
``robot_descriptions`` G1 the rest of the project already uses.

Usage::

    pip install "strands-robots[wbc,sim-mujoco]"
    # Point at a checkpoint dir with policy.onnx (+ optional walk_policy.onnx,
    # config.json). The real weights live in the upstream GR00T-WBC repo under
    # decoupled_wbc/sim2mujoco/resources/robots/g1/policy/.
    python examples/wbc_g1_torque_deploy.py --checkpoint /path/to/GEAR-SONIC \
        --duration 5 --vx 0.5 [--mp4 /tmp/g1_walk.mp4] [--viewer]

It prints per-second base x/z and a final verdict (advanced / fell / stayed put),
and exits non-zero on a hard error so it can gate CI behind real weights.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _build_torque_g1() -> tuple:
    """Load the Menagerie G1 and convert its actuators to pure-torque motors.

    Returns ``(mujoco_module, model, data, joint_names, leg_waist_idx)`` where
    ``joint_names`` is the 29-DOF order (qpos[7:] / qvel[6:]) and
    ``leg_waist_idx`` are the first 15 actuator indices (the controlled set).
    """
    import mujoco
    from robot_descriptions import g1_mj_description

    spec = mujoco.MjSpec.from_file(g1_mj_description.MJCF_PATH)

    # The robot_descriptions G1 MJCF is the robot alone - no ground plane. The
    # upstream g1_gear_wbc.xml scene includes a floor; without one the robot
    # falls through space (z -> -inf) even under a perfect static hold. Add a
    # static ground plane + a light so the robot has something to stand on.
    if not any(g.name == "wbc_ground" for g in spec.worldbody.geoms):
        spec.worldbody.add_geom(
            name="wbc_ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0.0, 0.0, 0.05],
            pos=[0.0, 0.0, 0.0],
            rgba=[0.4, 0.4, 0.4, 1.0],
        )

    # Convert every actuator to a pure-torque motor (gaintype FIXED, no bias),
    # so writing data.ctrl[i] = tau applies tau directly - the contract the
    # upstream PD loop assumes. The Menagerie model ships position servos.
    for act in spec.actuators:
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.gainprm = [1.0, 0.0, 0.0] + list(act.gainprm[3:])
        act.biasprm = [0.0, 0.0, 0.0] + list(act.biasprm[3:])
        # Open the ctrl range so a torque command isn't clamped to a small
        # position-servo range.
        act.ctrlrange = [-1000.0, 1000.0]
        act.ctrllimited = True
    model = spec.compile()
    data = mujoco.MjData(model)

    # 29-DOF joint order (skip the free/floating base joint).
    joint_names: list[str] = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
    return mujoco, model, data, joint_names


def _set_standing_pose(mujoco, model, data, default_angles: np.ndarray, height: float) -> None:
    """Place the base upright at ``height`` and the legs at the nominal stance."""
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, height]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion (w, x, y, z)
    n = min(len(default_angles), data.qpos.shape[0] - 7)
    data.qpos[7 : 7 + n] = default_angles[:n]
    mujoco.mj_forward(model, data)


def run(args: argparse.Namespace) -> int:
    from strands_robots.policies import create_policy
    from strands_robots.policies.wbc import WBC_G1_ALL_JOINTS, WBCPolicy

    mujoco, model, data, joint_names = _build_torque_g1()
    n_joints = data.qpos.shape[0] - 7

    policy = create_policy("wbc", checkpoint=args.checkpoint, walk=not args.no_walk)
    assert isinstance(policy, WBCPolicy)
    cfg = policy.config
    # Resolve the policy's joint mapping against the model's joint order.
    policy.set_robot_state_keys(joint_names)

    default_angles = np.zeros(n_joints, dtype=np.float64)
    da = np.asarray(cfg.default_angles, dtype=np.float64)
    default_angles[: min(len(da), n_joints)] = da[: min(len(da), n_joints)]

    na = cfg.num_actions
    # The leg+waist PD gains (kps/kds) live in the policy config; the upstream
    # PD law is applied via policy.compute_torques(...), which reads them.

    # Arm-hold PD (upstream uses kp=100, kd=0.5 to zero target for joints > 15).
    arm_kp, arm_kd = 100.0, 0.5

    physics_dt = float(args.physics_dt)
    model.opt.timestep = physics_dt
    decim = int(args.control_decimation)

    _set_standing_pose(mujoco, model, data, default_angles, cfg.height_cmd)
    x0, z0 = float(data.qpos[0]), float(data.qpos[2])

    # WBC emits joint-POSITION targets; seed with the nominal stance.
    target_dof_pos = default_angles.copy()
    command = {"target_velocity": [args.vx, args.vy, args.omega]}

    n_steps = int(args.duration / physics_dt)
    frames: list[np.ndarray] = []
    renderer = None
    if args.mp4:
        renderer = mujoco.Renderer(model, height=480, width=640)

    fell = False
    for step in range(n_steps):
        # --- per physics tick: PD -> torque on the controlled 15 ---
        q_lw = data.qpos[7 : 7 + na].copy()
        dq_lw = data.qvel[6 : 6 + na].copy()
        leg_tau = policy.compute_torques(target_dof_pos[:na], q_lw, dq_lw)
        data.ctrl[:na] = leg_tau
        # Hold the arms (joints na..n_joints) at default with a stiff PD.
        if n_joints > na:
            q_arm = data.qpos[7 + na : 7 + n_joints].copy()
            dq_arm = data.qvel[6 + na : 6 + n_joints].copy()
            arm_target = default_angles[na:n_joints]
            data.ctrl[na:n_joints] = (arm_target - q_arm) * arm_kp + (0.0 - dq_arm) * arm_kd

        mujoco.mj_step(model, data)

        # --- every control_decimation ticks: query the policy ---
        if step % decim == 0:
            obs = _model_observation(data, joint_names, n_joints)
            actions = policy.get_actions_sync(obs, "", **command)
            raw = np.array([actions[0][name] for name in WBC_G1_ALL_JOINTS[:na]], dtype=np.float64)
            # actions are already absolute targets (default + action_scale*raw);
            # WBCPolicy.get_actions returns target_q directly.
            target_dof_pos[:na] = raw

        if renderer is not None and step % max(1, int(1.0 / (physics_dt * args.mp4_fps))) == 0:
            renderer.update_scene(data, camera=-1)
            frames.append(renderer.render())

        z = float(data.qpos[2])
        if z < 0.4 * z0:
            fell = True
            print(f"[step {step} t={step * physics_dt:.2f}s] base height {z:.3f} m < 0.4*{z0:.3f} - FELL")
            break

        if step % int(1.0 / physics_dt) == 0:  # ~once per second
            print(f"[t={step * physics_dt:.1f}s] base x={data.qpos[0]:+.3f} z={data.qpos[2]:.3f}")

    x1, z1 = float(data.qpos[0]), float(data.qpos[2])
    forward = x1 - x0
    print("\n=== WBC G1 torque-deploy result ===")
    print(f"  duration: {args.duration:.1f}s | command vx={args.vx} vy={args.vy} omega={args.omega}")
    print(f"  base x: {x0:+.3f} -> {x1:+.3f}  (forward {forward:+.3f} m)")
    print(f"  base z: {z0:.3f} -> {z1:.3f} m")
    if fell:
        print("  VERDICT: FELL (height collapsed)")
    elif forward >= 0.10:
        print(f"  VERDICT: WALKED FORWARD ({forward:.2f} m)")
    elif abs(forward) < 0.05 and z1 > 0.7 * z0:
        print("  VERDICT: STAYED UPRIGHT (balanced, little forward progress)")
    else:
        print("  VERDICT: MOVED but inconclusive")

    if renderer is not None and frames:
        import imageio

        imageio.mimsave(args.mp4, frames, fps=args.mp4_fps)
        print(f"  video: {args.mp4} ({len(frames)} frames)")
        renderer.close()

    if args.viewer:
        import mujoco.viewer

        print("\n  launching interactive viewer (Ctrl+C to exit)...")
        mujoco.viewer.launch(model, data)
    return 0


def _model_observation(data, joint_names: list[str], n_joints: int) -> dict:
    """Build a per-joint observation dict (positions + .vel + base IMU) the way
    WBCPolicy expects, straight from MuJoCo data - including joint velocities and
    the base angular velocity / quaternion (which sim.run_policy does NOT supply,
    and which a balance controller genuinely needs)."""
    obs: dict = {}
    for i, name in enumerate(joint_names):
        obs[name] = float(data.qpos[7 + i])
        obs[f"{name}.vel"] = float(data.qvel[6 + i])
    obs["base_quat"] = [float(v) for v in data.qpos[3:7]]  # (w, x, y, z)
    obs["base_ang_vel"] = [float(v) for v in data.qvel[3:6]]
    return obs


def main() -> None:
    p = argparse.ArgumentParser(description="Torque-control deploy harness for WBCPolicy on the G1.")
    p.add_argument("--checkpoint", required=True, help="dir with policy.onnx (+ walk_policy.onnx, config.json)")
    p.add_argument("--duration", type=float, default=5.0, help="seconds to simulate")
    p.add_argument("--vx", type=float, default=0.5, help="forward velocity command (m/s)")
    p.add_argument("--vy", type=float, default=0.0, help="lateral velocity command (m/s)")
    p.add_argument("--omega", type=float, default=0.0, help="yaw rate command (rad/s)")
    p.add_argument("--physics-dt", type=float, default=0.005, help="physics timestep (upstream 0.005)")
    p.add_argument("--control-decimation", type=int, default=4, help="physics steps per policy query (upstream 4)")
    p.add_argument("--no-walk", action="store_true", help="load only the main (balance) policy")
    p.add_argument("--mp4", default="", help="write an MP4 of the rollout to this path")
    p.add_argument("--mp4-fps", type=int, default=30)
    p.add_argument("--viewer", action="store_true", help="launch the interactive MuJoCo viewer at the end")
    args = p.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
