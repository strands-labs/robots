#!/usr/bin/env python3
"""Composite whole-body deploy: WBC legs + a manipulation upper body on the G1.

Demonstrates :class:`strands_robots.policies.composite.CompositePolicy` driving
ONE Unitree G1 with two stacked policies, each tick:

* **lower** = :class:`WBCPolicy` (the real ``GR00T-WholeBodyControl-{Balance,
  Walk}.onnx`` weights) owns the 15 leg+waist joints and walks the robot.
* **upper** = a manipulation policy owns the 14 arm joints. Swap in
  ``create_policy("groot", port=5555)`` / pi0 / MolmoAct for a real VLA; this
  example ships a tiny scripted arm-wave so it runs fully in sim with no server.

The composite queries both children, routes leg+waist targets to the WBC torque
PD law and arm targets to a position PD, and steps MuJoCo. This is the
in-process equivalent of upstream's teleop-upper-body-on-WBC stack.

Usage::

    pip install "strands-robots[wbc,sim-mujoco]" robot_descriptions imageio imageio-ffmpeg
    # checkpoint dir holds policy.onnx (+ walk_policy.onnx, config.json); the
    # real weights live in NVlabs/GR00T-WholeBodyControl under
    # decoupled_wbc/sim2mujoco/resources/robots/g1/policy/.
    python examples/wbc/wbc_g1_composite.py --checkpoint /path/to/grootwbc-g1 \
        --duration 5 --vx 0.4 --mp4 /tmp/g1_composite.mp4
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Reuse the torque-harness building blocks from the sibling example (the G1
# torque model, standing pose, and MuJoCo-faithful observation builder) instead
# of duplicating them.
_TORQUE = importlib.util.spec_from_file_location(
    "wbc_g1_torque_deploy", str(Path(__file__).with_name("wbc_g1_torque_deploy.py"))
)
assert _TORQUE is not None and _TORQUE.loader is not None
_torque_mod = importlib.util.module_from_spec(_TORQUE)
_TORQUE.loader.exec_module(_torque_mod)
_build_torque_g1 = _torque_mod._build_torque_g1
_set_standing_pose = _torque_mod._set_standing_pose
_model_observation = _torque_mod._model_observation

from strands_robots.policies import CompositePolicy, Policy, create_policy  # noqa: E402
from strands_robots.policies.wbc import WBC_G1_ALL_JOINTS, WBC_G1_LEG_WAIST_JOINTS, WBCPolicy  # noqa: E402

# The 14 arm joints are the whole-body order minus the 15 leg+waist DOFs.
ARM_JOINTS: tuple[str, ...] = WBC_G1_ALL_JOINTS[len(WBC_G1_LEG_WAIST_JOINTS) :]


class ScriptedArmWavePolicy(Policy):
    """A zero-dependency upper-body policy that waves both shoulders.

    Stands in for a real manipulation VLA (GR00T / pi0 / MolmoAct) so the
    composite example runs fully offline. Emits absolute position targets for
    the arm joints only; every other joint is left to the lower policy.
    """

    def __init__(self, amplitude: float = 0.6, freq_hz: float = 0.5) -> None:
        self._amp = float(amplitude)
        self._freq = float(freq_hz)
        self._t = 0.0
        self._dt = 0.02

    @property
    def provider_name(self) -> str:
        return "scripted_arm_wave"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        return None

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        phase = math.sin(2 * math.pi * self._freq * self._t)
        self._t += self._dt
        targets: dict[str, float] = {}
        for name in ARM_JOINTS:
            if "shoulder_pitch" in name:
                targets[name] = self._amp * phase
            elif "elbow" in name:
                targets[name] = 0.4 + 0.3 * phase
            else:
                targets[name] = 0.0
        return [targets]


def simulate_composite(
    policy: CompositePolicy,
    *,
    vx: float = 0.4,
    duration: float = 5.0,
    physics_dt: float = 0.005,
    control_decimation: int = 4,
    arm_kp: float = 100.0,
    arm_kd: float = 0.5,
    renderer_dims: tuple[int, int] | None = None,
    fps: int = 30,
) -> dict:
    """Run the composite (WBC legs + scripted/VLA arms) torque loop on the G1.

    Leg+waist targets go through the WBC PD-to-torque law; arm targets from the
    upper policy drive a stiff position PD. Returns the same metrics dict as the
    torque-deploy harness plus the captured frames.
    """
    import mujoco

    wbc = policy.lower
    assert isinstance(wbc, WBCPolicy)
    mj, model, data, joint_names = _build_torque_g1()
    n_joints = data.qpos.shape[0] - 7
    cfg = wbc.config
    policy.set_robot_state_keys(joint_names)

    na = cfg.num_actions  # 15 controlled leg+waist DOFs
    default_angles = np.zeros(n_joints, dtype=np.float64)
    da = np.asarray(cfg.default_angles, dtype=np.float64)
    default_angles[: min(len(da), n_joints)] = da[: min(len(da), n_joints)]

    model.opt.timestep = physics_dt
    decim = int(control_decimation)
    _set_standing_pose(mj, model, data, default_angles, float(cfg.height_cmd))
    x0, z0 = float(data.qpos[0]), float(data.qpos[2])

    target_leg = default_angles[:na].copy()
    arm_target = default_angles[na:n_joints].copy()
    command = {"target_velocity": [vx, 0.0, 0.0]}

    n_steps = int(duration / physics_dt)
    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(model, height=renderer_dims[1], width=renderer_dims[0]) if renderer_dims else None
    render_every = max(1, int(1.0 / (physics_dt * fps)))

    fell = False
    steps_done = 0
    for step in range(n_steps):
        steps_done = step + 1
        # leg+waist: WBC target -> PD torque
        q_lw = data.qpos[7 : 7 + na].copy()
        dq_lw = data.qvel[6 : 6 + na].copy()
        data.ctrl[:na] = wbc.compute_torques(target_leg, q_lw, dq_lw)
        # arms: upper-policy target -> stiff position PD torque
        if n_joints > na:
            q_arm = data.qpos[7 + na : 7 + n_joints].copy()
            dq_arm = data.qvel[6 + na : 6 + n_joints].copy()
            data.ctrl[na:n_joints] = (arm_target - q_arm) * arm_kp + (0.0 - dq_arm) * arm_kd

        mj.mj_step(model, data)

        if step % decim == 0:
            obs = _model_observation(data, joint_names, n_joints)
            merged = policy.get_actions_sync(obs, "", **command)[0]
            target_leg = np.array([merged[name] for name in WBC_G1_LEG_WAIST_JOINTS[:na]], dtype=np.float64)
            arm_target = np.array(
                [merged.get(name, default_angles[na + i]) for i, name in enumerate(ARM_JOINTS)],
                dtype=np.float64,
            )

        if renderer is not None and step % render_every == 0:
            renderer.update_scene(data, camera=-1)
            frames.append(renderer.render())

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
    lower = create_policy("wbc", checkpoint=args.checkpoint, walk=not args.no_walk)
    assert isinstance(lower, WBCPolicy)
    upper: Policy
    if args.upper_port:
        upper = create_policy("groot", port=args.upper_port)
    else:
        upper = ScriptedArmWavePolicy()

    policy = CompositePolicy(
        lower=lower,
        upper=upper,
        lower_joints=WBC_G1_LEG_WAIST_JOINTS,
        upper_joints=ARM_JOINTS,
    )

    result = simulate_composite(
        policy,
        vx=args.vx,
        duration=args.duration,
        renderer_dims=(640, 480) if args.mp4 else None,
        fps=args.mp4_fps,
    )

    forward, z0, z1 = result["forward"], result["z0"], result["z1"]
    print("\n=== WBC G1 composite (legs=WBC, arms=upper) result ===")
    print(f"  upper policy: {upper.provider_name}")
    print(f"  duration: {args.duration:.1f}s | command vx={args.vx}")
    print(f"  base x: {result['x0']:+.3f} -> {result['x1']:+.3f}  (forward {forward:+.3f} m)")
    print(f"  base z: {z0:.3f} -> {z1:.3f} m")
    if result["fell"]:
        print(f"  VERDICT: FELL (height collapsed at step {result['steps']})")
    elif forward >= 0.10:
        print(f"  VERDICT: WALKED FORWARD while moving arms ({forward:.2f} m)")
    else:
        print("  VERDICT: stayed put / inconclusive")

    if args.mp4 and result["frames"]:
        import imageio

        imageio.mimsave(args.mp4, result["frames"], fps=args.mp4_fps)
        print(f"  video: {args.mp4} ({len(result['frames'])} frames)")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Composite WBC-legs + manipulation-arms deploy on the G1.")
    p.add_argument("--checkpoint", required=True, help="WBC dir with policy.onnx (+ walk_policy.onnx, config.json)")
    p.add_argument("--duration", type=float, default=5.0, help="seconds to simulate")
    p.add_argument("--vx", type=float, default=0.4, help="forward velocity command (m/s)")
    p.add_argument("--no-walk", action="store_true", help="load only the main (balance) policy")
    p.add_argument(
        "--upper-port",
        type=int,
        default=0,
        help="if set, use create_policy('groot', port=...) for the arms instead of the scripted wave",
    )
    p.add_argument("--mp4", default="", help="write an MP4 of the rollout to this path")
    p.add_argument("--mp4-fps", type=int, default=30)
    args = p.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
