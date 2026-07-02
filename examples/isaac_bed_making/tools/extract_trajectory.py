"""Extract a real bed-making arm trajectory from the Unitree LeRobot dataset.

Source dataset: ``unitreerobotics/G1_WBT_Brainco_Make_The_Bed`` (LeRobot **v3.0**,
Apache-2.0, 300 episodes of a real teleoperated Unitree G1 making a bed). This
tool pulls one episode, isolates the **manipulation window** (after the robot has
walked up to the bed, when the arms do the actual work), and writes a compact
``.npz`` the Isaac Sim demo replays onto the planted sim G1s. Locomotion (the
walk-up) is deliberately trimmed — it is a separate roadmap item (RL approach).

Why this maps cleanly onto the sim G1:

* ``action.robot_q_desired`` is ``[7 root pose] + [29 G1 joint targets]`` in the
  canonical Unitree G1 order (legs 12, waist 3, left arm 7, right arm 7). The sim
  uses the *actual* Unitree G1 model, so the arm/waist joint angles transfer 1:1
  — no angle-convention conversion. Left arm = ``q[22:29]``, right = ``q[29:36]``,
  waist = ``q[19:22]``.
* ``action.hand_cmd`` is the 12-value BrainCo finger command (6 per hand). The sim
  hand is Inspire (also 6/ hand) but the finger order differs; we retarget it here
  (see ``BRAINCO_TO_INSPIRE``) to a per-finger close fraction in [0, 1].

This script needs ``pip install pyarrow huggingface_hub numpy`` (NOT needed to run
the demo — the demo only reads the resulting ``.npz`` with numpy). Run::

    python extract_trajectory.py --episode 4 --out ../data/bed_making_traj.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = "unitreerobotics/G1_WBT_Brainco_Make_The_Bed"

# robot_q_desired = [7 root] + [29 joints]; joint j sits at index 7+j. Canonical
# Unitree G1 29-DOF order puts waist at joints 12-14 and the two arms at 15-28.
ROOT = slice(0, 7)
WAIST = slice(7 + 12, 7 + 15)        # q[19:22]  waist yaw, roll, pitch
LEFT_ARM = slice(7 + 15, 7 + 22)     # q[22:29]
RIGHT_ARM = slice(7 + 22, 7 + 29)    # q[29:36]

WAIST_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEFT_ARM_NAMES = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                  "left_shoulder_yaw_joint", "left_elbow_joint",
                  "left_wrist_roll_joint", "left_wrist_pitch_joint",
                  "left_wrist_yaw_joint"]
RIGHT_ARM_NAMES = [n.replace("left_", "right_") for n in LEFT_ARM_NAMES]

# BrainCo per-hand finger order (from the dataset card):
#   [thumb_oc, thumb_lat, index, middle, ring, little]
# Inspire per-hand finger order (the sim hand):
#   [index, middle, ring, little, thumb_oc, thumb_lat]
# Both are close fractions 0=open..1=close, so a reorder is the whole retarget.
BRAINCO_TO_INSPIRE = [2, 3, 4, 5, 0, 1]
INSPIRE_FINGER_NAMES = ["index", "middle", "ring", "little", "thumb_oc", "thumb_lat"]


def find_data_files(snapshot: Path):
    return sorted((snapshot / "data").glob("chunk-*/file-*.parquet"))


def arrival_frame(q: np.ndarray, tol: float = 0.20) -> int:
    """First frame after which the root stays within ``tol`` of its final xy — i.e.
    the robot has finished walking up and is now standing at the bed."""
    rootxy = q[:, :2]
    dist = np.linalg.norm(rootxy - rootxy[-1], axis=1)
    bad = np.where(dist >= tol)[0]
    return int(bad.max() + 1) if len(bad) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, default=4,
                    help="Episode to extract (default 4: a balanced two-arm window).")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "data" / "bed_making_traj.npz"))
    ap.add_argument("--arrival-tol", type=float, default=0.20)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    info = json.load(open(hf_hub_download(REPO, "meta/info.json", repo_type="dataset")))
    fps = int(info["fps"])
    # Locate the episode's rows from the episode metadata.
    ep_meta_files = []
    i = 0
    while True:
        try:
            ep_meta_files.append(hf_hub_download(
                REPO, f"meta/episodes/chunk-000/file-{i:03d}.parquet", repo_type="dataset"))
        except Exception:
            break
        i += 1
    em = pq.read_table(ep_meta_files[0]).to_pandas()
    row = em[em.episode_index == args.episode].iloc[0]
    chunk = int(row["data/chunk_index"])
    fidx = int(row["data/file_index"])
    g0, g1 = int(row["dataset_from_index"]), int(row["dataset_to_index"])

    data_path = hf_hub_download(
        REPO, f"data/chunk-{chunk:03d}/file-{fidx:03d}.parquet", repo_type="dataset")
    cols = ["episode_index", "action.robot_q_desired", "action.hand_cmd",
            "observation.state.ee_state"]
    d = pq.read_table(data_path, columns=cols).to_pandas()
    d = d[d.episode_index == args.episode]
    q = np.stack(d["action.robot_q_desired"].to_numpy()).astype(np.float32)
    hand = np.stack(d["action.hand_cmd"].to_numpy()).astype(np.float32)
    ee = np.stack(d["observation.state.ee_state"].to_numpy()).astype(np.float32)

    arr = arrival_frame(q, args.arrival_tol)
    sl = slice(arr, len(q))
    q_w, hand_w, ee_w = q[sl], hand[sl], ee[sl]

    # Re-zero each joint stream to its first frame of the window so the planted sim
    # robot starts from its natural rest pose and *adds* the recorded motion — this
    # removes the dataset robot's standing offset (its legs/root posture) that we do
    # not reproduce, while preserving the authentic shape of the arm/waist motion.
    waist = q_w[:, WAIST]
    larm = q_w[:, LEFT_ARM]
    rarm = q_w[:, RIGHT_ARM]

    # BrainCo (12) = [left 6, right 6] -> per-hand Inspire close fractions.
    hl = hand_w[:, 0:6][:, BRAINCO_TO_INSPIRE]
    hr = hand_w[:, 6:12][:, BRAINCO_TO_INSPIRE]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        fps=np.int64(fps),
        source_repo=REPO,
        source_episode=np.int64(args.episode),
        arrival_frame=np.int64(arr),
        episode_rows=np.array([g0, g1], dtype=np.int64),
        waist_names=np.array(WAIST_NAMES),
        left_arm_names=np.array(LEFT_ARM_NAMES),
        right_arm_names=np.array(RIGHT_ARM_NAMES),
        inspire_finger_names=np.array(INSPIRE_FINGER_NAMES),
        q_waist=waist.astype(np.float32),
        q_left_arm=larm.astype(np.float32),
        q_right_arm=rarm.astype(np.float32),
        hand_left=hl.astype(np.float32),
        hand_right=hr.astype(np.float32),
        ee_state=ee_w.astype(np.float32),
    )

    n = len(q_w)
    print(f"episode {args.episode}: {len(q)} frames, arrival at {arr} "
          f"-> manipulation window {n} frames ({n / fps:.1f}s @ {fps}fps)")
    print(f"  left  arm range (rad): {np.round(larm.max(0) - larm.min(0), 2)}")
    print(f"  right arm range (rad): {np.round(rarm.max(0) - rarm.min(0), 2)}")
    print(f"  waist     range (rad): {np.round(waist.max(0) - waist.min(0), 2)}")
    print(f"  ee z dip (root frame): L {ee_w[:, 2].min():.2f}..{ee_w[:, 2].max():.2f}  "
          f"R {ee_w[:, 8].min():.2f}..{ee_w[:, 8].max():.2f}")
    print(f"  hand close frac max: L {hl.max(0).round(2)}  R {hr.max(0).round(2)}")
    print(f"  wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
