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
        --duration 5 --vx 0.5 [--mp4 /tmp/g1_walk.mp4]

It prints per-second base x/z and a final verdict (advanced / fell / stayed put),
and exits non-zero on a hard error so it can gate CI behind real weights.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _build_torque_g1() -> tuple:
    """Load the Menagerie G1 and convert its actuators to pure-torque motors.

    Returns ``(mujoco_module, model, data, joint_names)`` where ``joint_names``
    is the 29-DOF actuated-joint order (qpos[7:] / qvel[6:]); the first 15 are
    the controlled leg+waist set, the rest are the held arms.
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


def simulate_rollout(
    policy,  # type: ignore[no-untyped-def]
    *,
    vx: float = 0.5,
    vy: float = 0.0,
    omega: float = 0.0,
    duration: float = 5.0,
    physics_dt: float = 0.005,
    control_decimation: int = 4,
    height: float | None = None,
    on_step=None,  # type: ignore[no-untyped-def]
    renderer_dims: tuple[int, int] | None = None,
    fps: int = 30,
) -> dict:
    """Run the upstream torque-control loop and return rollout metrics.

    Pure / importable so the CLI AND the test suite drive the identical loop.
    The robot is a torque-actuated G1 (built in :func:`_build_torque_g1`); each
    physics tick applies ``policy.compute_torques`` to the 15 controlled joints
    (PD -> torque), holds the arms with a stiff PD, and every
    ``control_decimation`` ticks re-queries ``policy.get_actions_sync`` for new
    position targets.

    Args:
        policy: A constructed WBCPolicy (real or stubbed). ``set_robot_state_keys``
            is called here against the model's joint order.
        vx, vy, omega: Locomotion command.
        duration: Wall-clock seconds (n_steps = duration / physics_dt).
        height: Initial base height; defaults to ``policy.config.height_cmd``.
        on_step: Optional ``(step:int, data) -> None`` hook (per physics tick).
        renderer_dims: ``(width, height)`` to capture RGB frames, else no render.
        fps: Frame-capture rate when rendering.

    Returns:
        Dict with ``x0/z0/x1/z1/forward/fell/steps/frames`` (frames is a list of
        HxWx3 uint8 arrays, empty unless ``renderer_dims`` is set).
    """
    from strands_robots.policies.wbc import WBC_G1_ALL_JOINTS

    mujoco, model, data, joint_names = _build_torque_g1()
    n_joints = data.qpos.shape[0] - 7
    cfg = policy.config
    policy.set_robot_state_keys(joint_names)

    default_angles = np.zeros(n_joints, dtype=np.float64)
    da = np.asarray(cfg.default_angles, dtype=np.float64)
    default_angles[: min(len(da), n_joints)] = da[: min(len(da), n_joints)]

    na = cfg.num_actions
    # The leg+waist PD gains live in the policy config and are applied via
    # policy.compute_torques(...). Arms held with a stiff PD (upstream kp=100,
    # kd=0.5 to the default pose for joints beyond the controlled set).
    arm_kp, arm_kd = 100.0, 0.5

    model.opt.timestep = physics_dt
    decim = int(control_decimation)
    base_height = float(height) if height is not None else float(cfg.height_cmd)
    _set_standing_pose(mujoco, model, data, default_angles, base_height)
    x0, z0 = float(data.qpos[0]), float(data.qpos[2])

    target_dof_pos = default_angles.copy()  # WBC emits position targets
    command = {"target_velocity": [vx, vy, omega]}

    n_steps = int(duration / physics_dt)
    frames: list[np.ndarray] = []
    renderer = None
    if renderer_dims is not None:
        renderer = mujoco.Renderer(model, height=renderer_dims[1], width=renderer_dims[0])
    render_every = max(1, int(1.0 / (physics_dt * fps)))

    fell = False
    steps_done = 0
    for step in range(n_steps):
        steps_done = step + 1
        # --- per physics tick: PD -> torque on the controlled 15 ---
        q_lw = data.qpos[7 : 7 + na].copy()
        dq_lw = data.qvel[6 : 6 + na].copy()
        data.ctrl[:na] = policy.compute_torques(target_dof_pos[:na], q_lw, dq_lw)
        # Hold the arms at default with a stiff PD.
        if n_joints > na:
            q_arm = data.qpos[7 + na : 7 + n_joints].copy()
            dq_arm = data.qvel[6 + na : 6 + n_joints].copy()
            data.ctrl[na:n_joints] = (default_angles[na:n_joints] - q_arm) * arm_kp + (0.0 - dq_arm) * arm_kd

        mujoco.mj_step(model, data)

        # --- every control_decimation ticks: query the policy ---
        if step % decim == 0:
            obs = _model_observation(data, joint_names, n_joints)
            actions = policy.get_actions_sync(obs, "", **command)
            # WBCPolicy.get_actions returns absolute targets keyed by joint name.
            target_dof_pos[:na] = np.array([actions[0][name] for name in WBC_G1_ALL_JOINTS[:na]], dtype=np.float64)

        if renderer is not None and step % render_every == 0:
            renderer.update_scene(data, camera=-1)
            frames.append(renderer.render())

        if on_step is not None:
            on_step(step, data)

        if float(data.qpos[2]) < 0.4 * z0:
            fell = True
            break

    if renderer is not None:
        renderer.close()
    x1, z1 = float(data.qpos[0]), float(data.qpos[2])
    return {
        "x0": x0,
        "z0": z0,
        "x1": x1,
        "z1": z1,
        "forward": x1 - x0,
        "fell": fell,
        "steps": steps_done,
        "frames": frames,
    }


def run(args: argparse.Namespace) -> int:
    from strands_robots.policies import create_policy
    from strands_robots.policies.wbc import WBCPolicy

    policy = create_policy("wbc", checkpoint=args.checkpoint, walk=not args.no_walk)
    assert isinstance(policy, WBCPolicy)

    physics_dt = float(args.physics_dt)

    def _progress(step: int, data) -> None:  # type: ignore[no-untyped-def]
        if step % int(1.0 / physics_dt) == 0:  # ~once per second
            print(f"[t={step * physics_dt:.1f}s] base x={data.qpos[0]:+.3f} z={data.qpos[2]:.3f}")

    result = simulate_rollout(
        policy,
        vx=args.vx,
        vy=args.vy,
        omega=args.omega,
        duration=args.duration,
        physics_dt=physics_dt,
        control_decimation=args.control_decimation,
        on_step=_progress,
        renderer_dims=(640, 480) if args.mp4 else None,
        fps=args.mp4_fps,
    )

    forward, z0, z1 = result["forward"], result["z0"], result["z1"]
    print("\n=== WBC G1 torque-deploy result ===")
    print(f"  duration: {args.duration:.1f}s | command vx={args.vx} vy={args.vy} omega={args.omega}")
    print(f"  base x: {result['x0']:+.3f} -> {result['x1']:+.3f}  (forward {forward:+.3f} m)")
    print(f"  base z: {z0:.3f} -> {z1:.3f} m")
    if result["fell"]:
        print(f"  VERDICT: FELL (height collapsed at step {result['steps']})")
    elif forward >= 0.10:
        print(f"  VERDICT: WALKED FORWARD ({forward:.2f} m)")
    elif abs(forward) < 0.05 and z1 > 0.7 * z0:
        print("  VERDICT: STAYED UPRIGHT (balanced, little forward progress)")
    else:
        print("  VERDICT: MOVED but inconclusive")

    if args.mp4 and result["frames"]:
        import imageio

        imageio.mimsave(args.mp4, result["frames"], fps=args.mp4_fps)
        print(f"  video: {args.mp4} ({len(result['frames'])} frames)")
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
    args = p.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
