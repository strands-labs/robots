"""Custom MDP terms for the whole-body bed-reach / bed-pull RL env.

Two ideas live here:

**Station-keeping** (`base_xy_anchor_l2`). The free base must reach by LEANING / SQUATTING with
its feet planted, not by stepping. Beside a bed, a policy free to translate reaches forward by
stepping BACKWARD, walks itself off its stance and topples. Penalizing the base's xy drift from its
spawn anchor (quadratic, so a small lean is nearly free while a backward step is heavily penalized)
forces "lean over and pull without losing balance."

**Ambidexterity** (`same_side_*`, `idle_arm_deviation_l1`, `randomize_ee_load`). The reach is solved
with the hand on the SAME SIDE as the target — left hand for a +y (leftward) target, right hand for
a −y target. Exploiting the G1's bilateral symmetry this way (cf. SYMDEX, arXiv:2505.05287) makes a
sideways drag a natural abduction for whichever hand leads, instead of a cross-body sweep one hand
can't balance. The active hand is read straight from the command's y-sign, so the observation is
unchanged — the policy already sees the target. The idle arm is regularized to hang at its side.
"""

from __future__ import annotations

import math

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms


def base_xy_anchor_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared horizontal (xy) distance of the robot base from its per-env spawn anchor.

    The anchor is the env origin (with the reset xy-noise zeroed, the robot spawns there), so this
    is the base's drift from where it started. Returns (num_envs,)."""
    asset = env.scene[asset_cfg.name]
    xy = asset.data.root_pos_w[:, :2]
    anchor = env.scene.env_origins[:, :2]
    return torch.sum(torch.square(xy - anchor), dim=1)


def _active_hand_pos_w(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """World position of the SAME-SIDE hand for each env: the left hand when the commanded target is
    on the robot's left (base +y), the right hand otherwise. ``asset_cfg.body_ids`` must be
    ``[right_ee_id, left_ee_id]`` (resolve with ``preserve_order=True``). Returns (num_envs, 3)."""
    asset = env.scene[asset_cfg.name]
    right = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    left = asset.data.body_pos_w[:, asset_cfg.body_ids[1]]
    use_left = (env.command_manager.get_command(command_name)[:, 1] >= 0.0).unsqueeze(-1)
    return torch.where(use_left, left, right)


def same_side_position_error(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """L2 distance from the same-side hand to the commanded base-frame target. Returns (num_envs,)."""
    asset = env.scene[asset_cfg.name]
    des_pos_b = env.command_manager.get_command(command_name)[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    return torch.norm(des_pos_w - _active_hand_pos_w(env, command_name, asset_cfg), dim=1)


def same_side_position_error_tanh(env, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Shaped reach reward (1 at the target, decaying over ``std``) for the same-side hand."""
    return 1.0 - torch.tanh(same_side_position_error(env, command_name, asset_cfg) / std)


def idle_arm_deviation_l1(
    env,
    command_name: str,
    right_arm_cfg: SceneEntityCfg,
    left_arm_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize the IDLE arm (the one not reaching) drifting from its at-side default, so it hangs
    naturally instead of contorting into an uncanny counter-pose. The active (same-side) arm is left
    free to reach. Returns (num_envs,)."""
    asset = env.scene[right_arm_cfg.name]
    jp, dp = asset.data.joint_pos, asset.data.default_joint_pos
    right_dev = torch.sum(torch.abs(jp[:, right_arm_cfg.joint_ids] - dp[:, right_arm_cfg.joint_ids]), dim=1)
    left_dev = torch.sum(torch.abs(jp[:, left_arm_cfg.joint_ids] - dp[:, left_arm_cfg.joint_ids]), dim=1)
    # Left target -> left arm active -> penalize the RIGHT (idle) arm, and vice versa.
    use_left = env.command_manager.get_command(command_name)[:, 1] >= 0.0
    return torch.where(use_left, right_dev, left_dev)


def randomize_ee_load(
    env,
    env_ids,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    force_range: tuple[float, float] = (0.0, 35.0),
    slip_prob: float = 0.4,
) -> None:
    """Apply a horizontal grip-slip / sheet-tension force to the ACTIVE (same-side) hand — the load
    the sheet's tension/drag puts on whichever hand is gripping — that RANDOMLY DROPS TO ZERO (grip
    slip / let-go). Run on an interval so the load steps up and down every 1-2.5 s; the policy learns
    to absorb a SUDDEN load change without toppling ("don't fall when the sheet slips or you release
    it"). This is the FALCON-style force disturbance the loco-manipulation literature uses for
    force-adaptive whole-body control. ``asset_cfg.body_ids`` must be ``[right_ee_id, left_ee_id]``.

    The force is a horizontal vector of random magnitude in ``force_range`` and random direction;
    with probability ``slip_prob`` it is zero. It is applied to the active hand only (the idle hand
    carries no load), via the articulation's external-wrench buffer, so PhysX re-applies it every
    step until the next interval resamples it."""
    asset = env.scene[asset_cfg.name]
    right_id, left_id = asset_cfg.body_ids[0], asset_cfg.body_ids[1]
    n = len(env_ids)
    use_left = env.command_manager.get_command(command_name)[env_ids, 1] >= 0.0
    mag = torch.rand(n, device=env.device) * (force_range[1] - force_range[0]) + force_range[0]
    mag = torch.where(torch.rand(n, device=env.device) < slip_prob, torch.zeros_like(mag), mag)
    theta = torch.rand(n, device=env.device) * 2 * math.pi
    fx, fy = mag * torch.cos(theta), mag * torch.sin(theta)
    zero = torch.zeros_like(fx)
    forces = torch.zeros(n, 2, 3, device=env.device)  # body slot 0 = right, 1 = left
    forces[:, 0, 0], forces[:, 0, 1] = torch.where(use_left, zero, fx), torch.where(use_left, zero, fy)
    forces[:, 1, 0], forces[:, 1, 1] = torch.where(use_left, fx, zero), torch.where(use_left, fy, zero)
    asset.set_external_force_and_torque(
        forces, torch.zeros_like(forces), body_ids=[right_id, left_id], env_ids=env_ids
    )
