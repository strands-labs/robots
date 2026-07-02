"""Replay a **real** bed-making motion onto a sim G1.

Instead of scripted reach-and-drag waypoints, this drives each G1's waist + arm
(and optionally Inspire finger) joints directly from a trajectory recorded by a
real teleoperated Unitree G1 making a bed — the ``unitreerobotics/
G1_WBT_Brainco_Make_The_Bed`` LeRobot dataset (see ``tools/extract_trajectory.py``
for how ``data/bed_making_traj.npz`` is produced).

Because the sim uses the actual Unitree G1 model, the recorded joint angles are
applied **1:1** as joint position targets — the arm pose in the sim matches the
real arm pose. The dataset is a *single* robot; the two sim peers replay the same
trajectory, and because robot 1 is spawned rotated 180° about Z (facing the bed
from the opposite long side), the identical joint angles produce a mirrored,
point-symmetric motion — two peers working the bed from both sides at once, as in
the Figure Helix clip. An optional per-robot start ``lag`` desynchronises them so
they look independent rather than mechanically identical.

Only waist + arm joints are replayed; the legs/root are left at their planted
rest pose (the robots are fixed-base — the walk-up is a separate roadmap item).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

DEFAULT_TRAJ = Path(__file__).resolve().parent / "data" / "bed_making_traj.npz"

# Which finger keyword each of the 6 per-hand close fractions drives. The dataset
# (Inspire) order is [index, middle, ring, little, thumb_oc, thumb_lat]; the sim
# Inspire hand names "little" as "pinky" and splits the thumb into a lateral
# "proximal_base" joint (thumb_lat) and the curl joints (thumb_oc).
FINGER_KEYWORDS = ["index", "middle", "ring", "pinky", "thumb_oc", "thumb_lat"]


class TrajectoryReplay:
    """Drives one G1's waist+arm (+hand) joints from a recorded trajectory."""

    def __init__(self, robot, device: str, npz_path=DEFAULT_TRAJ,
                 drive_waist: bool = True, drive_hands: bool = True):
        import torch

        self._torch = torch
        self.robot = robot
        self.device = device
        z = np.load(npz_path, allow_pickle=False)
        self.fps = int(z["fps"])
        self.source_episode = int(z["source_episode"])

        # ── body joints: waist (3) + left arm (7) + right arm (7) ──────────────
        names: List[str] = []
        cols = []
        if drive_waist:
            names += list(z["waist_names"])
            cols.append(z["q_waist"])
        names += list(z["left_arm_names"])
        cols.append(z["q_left_arm"])
        names += list(z["right_arm_names"])
        cols.append(z["q_right_arm"])
        traj = np.concatenate(cols, axis=1)  # (N, J)

        ids: List[int] = []
        keep: List[int] = []
        for k, nm in enumerate(names):
            jid, _ = robot.find_joints([str(nm)], preserve_order=True)
            if jid:
                ids.append(int(jid[0]))
                keep.append(k)
        self.body_ids = ids
        self.body_traj = torch.tensor(traj[:, keep], device=device, dtype=torch.float32)
        self.n = int(self.body_traj.shape[0])
        self.missing = [str(names[k]) for k in range(len(names)) if k not in keep]

        # ── optional Inspire finger drive ────────────────────────────────────
        self.hand_ids: Optional[List[int]] = None
        self.hand_traj = None
        if drive_hands:
            self._setup_hands(z)

    # ──────────────────────────────────────────────────────────────────────────
    def _setup_hands(self, z) -> None:
        """Map the 6 per-hand close fractions onto the sim's Inspire finger joints.

        For each hand joint we resolve which finger fraction drives it (by name),
        then lerp from the joint's default (open) pose toward whichever limit is
        farther from default (closed) — so the sign of the joint axis does not
        matter, closing always curls the finger."""
        torch = self._torch
        robot = self.robot
        jnames = list(robot.data.joint_names)
        lo = robot.data.joint_pos_limits[0, :, 0]
        hi = robot.data.joint_pos_limits[0, :, 1]
        dft = robot.data.default_joint_pos[0]

        hl = z["hand_left"]  # (N,6) close fractions, left hand
        hr = z["hand_right"]  # (N,6) close fractions, right hand
        ids: List[int] = []
        frac_cols = []  # per joint: which (N,) fraction column drives it
        signed_span = []  # per joint: (closed_limit - default)
        base = []  # per joint: default
        for ji, nm in enumerate(jnames):
            if not (nm.startswith("L_") or nm.startswith("R_")):
                continue
            side_frac = hl if nm.startswith("L_") else hr
            # thumb lateral = the proximal *yaw* joint; thumb curl (open/close) =
            # the proximal-pitch / intermediate / distal thumb joints.
            if "thumb" in nm:
                fk = "thumb_lat" if "yaw" in nm else "thumb_oc"
            else:
                fk = next((k for k in ("index", "middle", "ring", "pinky") if k in nm), None)
            if fk is None:
                continue
            col = FINGER_KEYWORDS.index(fk)
            closed = lo[ji] if (abs(float(lo[ji] - dft[ji])) > abs(float(hi[ji] - dft[ji]))) else hi[ji]
            ids.append(ji)
            frac_cols.append(side_frac[:, col])
            signed_span.append(float(closed - dft[ji]))
            base.append(float(dft[ji]))

        if not ids:
            return
        fr = torch.tensor(np.stack(frac_cols, axis=1), device=self.device, dtype=torch.float32)  # (N,H)
        span = torch.tensor(signed_span, device=self.device, dtype=torch.float32)  # (H,)
        b = torch.tensor(base, device=self.device, dtype=torch.float32)  # (H,)
        self.hand_ids = ids
        self.hand_traj = b.unsqueeze(0) + fr * span.unsqueeze(0)  # (N,H)

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def n_frames(self) -> int:
        return self.n

    def frame_target(self, i: int):
        """The body-joint target row for dataset frame ``i`` (clamped)."""
        i = max(0, min(self.n - 1, i))
        return self.body_traj[i: i + 1]

    def warmup(self, alpha: float) -> None:
        """Blend from the robot's current body-joint pose to trajectory frame 0.
        ``alpha`` goes 0->1; call repeatedly while stepping to ease in the motion."""
        cur = self.robot.data.joint_pos[0, self.body_ids]
        tgt = self.body_traj[0]
        blend = (1.0 - alpha) * cur + alpha * tgt
        self.robot.set_joint_position_target(blend.unsqueeze(0), joint_ids=self.body_ids)

    def apply(self, i: int) -> None:
        """Set this robot's joint position targets to dataset frame ``i``."""
        self.robot.set_joint_position_target(self.frame_target(i), joint_ids=self.body_ids)
        if self.hand_ids is not None:
            i = max(0, min(self.n - 1, i))
            self.robot.set_joint_position_target(self.hand_traj[i: i + 1], joint_ids=self.hand_ids)
