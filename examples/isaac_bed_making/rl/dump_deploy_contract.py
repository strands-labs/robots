"""Dump the GROUND-TRUTH deploy contract for the 23-DOF bed-reach policy.

The single biggest sim-to-real footgun for the G1 is obs/action joint-ORDER parity:
IsaacLab orders the articulation DOFs interleaved (left/right by tree depth), which is
NOT the Unitree SDK's 0..28 index order. This script boots the exact training env
(BedReachEduEnvCfg_PLAY) and writes the resolved order + scales + defaults to JSON, so
the on-robot deploy harness maps sim<->SDK with zero guessing.

Run (headless on the Spark GPU):
  cd ~/workspaces/git/IsaacLab && export LD_PRELOAD=$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1 \
    && ./isaaclab.sh -p ~/workspaces/git/robots-issue3/examples/isaac_bed_making/rl/dump_deploy_contract.py \
       --headless --num_envs 1
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump the bed-reach deploy contract.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--out", type=str, default=None, help="output JSON path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.bed_reach_env_cfg_g1edu import BedReachEduEnvCfg_PLAY  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cfg = BedReachEduEnvCfg_PLAY()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.eval_cam = None  # no rendering needed for a joint-order dump
    # Deploy runs at the NOMINAL PD gains (you set the real robot's gains; you can't randomize
    # them). The startup actuator-gain DR scales the per-env gains ±20%, so it would corrupt the
    # gains we read back — disable it so the dumped gains are the nominal training values.
    cfg.events.randomize_gains = None
    env = ManagerBasedRLEnv(cfg=cfg)
    robot = env.unwrapped.scene["robot"]

    # 1) Full articulation DOF order (the PhysX/IsaacLab order).
    artic_names = list(robot.joint_names)

    # 2) Action term: the JointPositionAction over the 23 BODY_JOINTS — resolved order,
    #    scale, and offset (use_default_offset=True -> offset is the default joint pos).
    act_term = env.unwrapped.action_manager.get_term("joint_pos")
    act_ids = list(act_term._joint_ids) if not isinstance(act_term._joint_ids, slice) else list(range(len(artic_names)))
    act_names = [artic_names[i] for i in act_ids]
    scale = act_term._scale
    offset = act_term._offset
    def _to_list(x):
        if torch.is_tensor(x):
            return x.flatten().tolist()
        return [float(x)] * len(act_names)
    act_scale = _to_list(scale)
    act_offset = _to_list(offset)

    # 3) Obs joint_pos / joint_vel resolved order (must match the action order).
    #    Pull from the observation term cfg's asset_cfg joint_ids.
    obs_terms = env.unwrapped.observation_manager._group_obs_term_names["policy"]
    def _flatdim(d):
        if isinstance(d, (tuple, list)):
            p = 1
            for x in d:
                p *= int(x)
            return p
        return int(d)
    obs_dims = [_flatdim(d) for d in env.unwrapped.observation_manager._group_obs_term_dim["policy"]]
    jp_cfg = cfg.observations.policy.joint_pos.params["asset_cfg"]
    jp_cfg.resolve(env.unwrapped.scene)
    obs_jp_ids = list(jp_cfg.joint_ids) if not isinstance(jp_cfg.joint_ids, slice) else list(range(len(artic_names)))
    obs_jp_names = [artic_names[i] for i in obs_jp_ids]

    # 4) Default joint pos (articulation order) and per-action-joint default.
    default_q_full = robot.data.default_joint_pos[0].tolist()
    default_by_name = {n: float(default_q_full[i]) for i, n in enumerate(artic_names)}

    # 5) EE / pelvis bodies + body order.
    body_names = list(robot.body_names)

    # 6) Per-joint PD gains (kp, kd) the policy was trained under, in ACTION-joint order. The
    #    whole-body deploy MUST command these — a zero/odd-gain transfer falls. Read the resolved
    #    per-joint gains straight off the articulation (randomize_gains disabled above, so these
    #    are the nominal values), and keep only the 23 action joints.
    def _gain_vec(attr):
        d = robot.data
        for name in (attr, "default_" + attr):
            if hasattr(d, name):
                return getattr(d, name)[0].detach().cpu().tolist()
        raise AttributeError(f"articulation data has neither {attr} nor default_{attr}")
    js = _gain_vec("joint_stiffness")
    jd = _gain_vec("joint_damping")
    gains_by_name = {n: [round(float(js[i]), 6), round(float(jd[i]), 6)] for i, n in enumerate(artic_names)}
    gains = {n: gains_by_name[n] for n in act_names}

    contract = {
        "control_hz": 1.0 / (cfg.sim.dt * cfg.decimation),
        "decimation": cfg.decimation,
        "sim_dt": cfg.sim.dt,
        "articulation_joint_names": artic_names,
        "n_articulation_joints": len(artic_names),
        "action": {
            "joint_names_in_order": act_names,   # policy output i -> this joint
            "n": len(act_names),
            "scale": act_scale,                  # target_q = offset + scale*action
            "offset": act_offset,                # == default_q for these joints
        },
        "obs": {
            "term_order": list(obs_terms),
            "term_dims": obs_dims,
            "total_dim": int(sum(obs_dims)),
            "joint_pos_names_in_order": obs_jp_names,
            "joint_pos_offset_default": [default_by_name[n] for n in obs_jp_names],
        },
        "gains": gains,  # joint-name -> [kp, kd]; nominal PD the policy trained under (deploy needs these)
        "default_joint_pos_by_name": default_by_name,
        "ee_bodies": {"right": "right_wrist_roll_link", "left": "left_wrist_roll_link",
                      "pelvis": "pelvis"},
        "body_names": body_names,
        "command": {
            "name": "hand_target",
            "frame": "base/pelvis-local",
            "pos_x": [0.18, 0.55], "pos_y": [-0.40, 0.40], "pos_z": [-0.16, 0.10],
            "as_obs": "position(3)+quaternion(4) wxyz, in base frame",
        },
    }

    out = args_cli.out or os.path.join(HERE, "deploy_contract.json")
    with open(out, "w") as f:
        json.dump(contract, f, indent=2)

    print("\n==== DEPLOY CONTRACT ====")
    print(f"control_hz = {contract['control_hz']}")
    print(f"articulation order ({len(artic_names)}): {artic_names}")
    print(f"\nACTION joint order ({len(act_names)}):")
    for i, n in enumerate(act_names):
        print(f"  [{i:2d}] {n:28s} scale={act_scale[i]:.3f} offset={act_offset[i]:+.4f}")
    print(f"\nOBS joint_pos order matches action order: "
          f"{obs_jp_names == act_names}")
    print(f"obs term order: {list(obs_terms)}")
    print(f"obs term dims:  {obs_dims}  total={sum(obs_dims)}")
    print("\nPD gains (kp, kd) per action joint:")
    for n in act_names:
        print(f"  {n:28s} kp={gains[n][0]:7.2f} kd={gains[n][1]:6.2f}")
    print(f"\nwrote {out}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
