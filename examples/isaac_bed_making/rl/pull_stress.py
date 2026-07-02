"""Sustained-draw pull-load stress test for the 23-DOF G1 EDU bed-reach policy (issue #3).

Re-litigating "can the EDU survive pulling a bedsheet?" The training disturbance (`randomize_ee_load`)
is a RANDOM-direction force that toggles ON/OFF every 1-2.5 s — it teaches "don't fall when the grip
slips," but it is NOT a sustained directional draw. A real sheet draw is a CONSTANT pull the policy
must hold against the whole time. This harness applies exactly that: a constant horizontal force on
the gripping (same-side) hand, a different magnitude per env, and renders a tiled montage so you can
SEE the topple ceiling by eye (the durable rule: judge behaviour by what the human sees).

The pull is world-frame +y — OUTWARD, the same side the hand is reaching — which drags the CoM toward
the stance edge. That is the WORST-CASE direction for balance (a headward/footward draw is gentler),
so a magnitude the robot survives here it survives in the real task with margin.

Each env holds the SAME fixed left-hand reach target; only the constant draw force differs. The random
ee_load and push perturbations are OFF, so the ONLY load is the sustained draw — clean attribution.

The mp4 lands in `examples/isaac_bed_making/rl/eval/` and its absolute path is printed.

Run:
  cd ~/workspaces/git/IsaacLab && export LD_PRELOAD=$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1 \
    && ./isaaclab.sh -p ~/workspaces/git/robots-issue3/examples/isaac_bed_making/rl/pull_stress.py \
       --headless --enable_cameras --forces 0,15,35,55,75
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sustained-draw pull-load stress test (23-DOF G1 EDU).")
parser.add_argument("--forces", type=str, default="0,15,35,55,75",
                    help="comma list of constant horizontal draw forces (N), one per env.")
parser.add_argument("--checkpoint", type=str, default=None, help="model_*.pt (default: latest g1edu).")
parser.add_argument("--target", type=str, default="0.45,0.30,-0.05",
                    help="fixed base-frame reach target x,y,z (a +y target => LEFT hand grips).")
parser.add_argument("--seconds", type=float, default=8.0, help="render length (50 Hz control).")
parser.add_argument("--tag", type=str, default="pull_stress", help="filename tag for the mp4.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob  # noqa: E402
import importlib.metadata as metadata  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.agents import BedReachPPORunnerCfg  # noqa: E402
from rl.bed_reach_env_cfg_g1edu import BedReachEduEnvCfg_PLAY  # noqa: E402
from rl.robot_cfg_g1edu import LEFT_EE_BODY, RIGHT_EE_BODY  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT = "bed_reach_g1edu"


def latest_checkpoint() -> str | None:
    runs = sorted(glob.glob(os.path.join(HERE, "logs", EXPERIMENT, "*")))
    for run in reversed(runs):
        ckpts = sorted(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(p.split("_")[-1].split(".")[0]))
        if ckpts:
            return ckpts[-1]
    return None


def encode(frames_dir: str, out_mp4: str, n: int) -> None:
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", "25", "-pattern_type", "glob",
                "-i", os.path.join(frames_dir, "frame_*.png"),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", out_mp4,
            ],
            capture_output=True,
        )
        print(f"\n[pull-stress] >>> STRESS MP4 READY: {out_mp4}  ({n} frames)\n")
    else:
        print(f"[pull-stress] {n} frames in {frames_dir} (install ffmpeg for mp4)")


def main():
    forces = [float(x) for x in args_cli.forces.split(",")]
    tx, ty, tz = (float(x) for x in args_cli.target.split(","))

    env_cfg = BedReachEduEnvCfg_PLAY()
    env_cfg.scene.num_envs = len(forces)
    env_cfg.scene.env_spacing = 3.0
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    # Pin every env to the SAME fixed reach target (a +y target => left hand is the same-side grip).
    r = env_cfg.commands.hand_target.ranges
    r.pos_x, r.pos_y, r.pos_z = (tx, tx), (ty, ty), (tz, tz)
    env_cfg.commands.hand_target.resampling_time_range = (1.0e9, 1.0e9)  # never resample
    # The ONLY load is our sustained draw: kill the random grip-slip force and the random pushes.
    env_cfg.events.ee_load = None
    env_cfg.events.push_robot = None
    # Smaller per-env cams so a 5-up montage encodes quickly.
    env_cfg.scene.eval_cam.height, env_cfg.scene.eval_cam.width = 288, 512

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device

    agent_cfg = BedReachPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = str(device)
    resume = args_cli.checkpoint or latest_checkpoint()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume)
    policy = runner.get_inference_policy(device=device)
    print(f"[pull-stress] loaded checkpoint: {resume}")
    print(f"[pull-stress] forces (N): {forces}   target(base): {(tx, ty, tz)}")

    # Resolve the two hand links; the +y target means the LEFT hand grips, so draw on the left link.
    robot = env.unwrapped.scene["robot"]
    ee_ids, _ = robot.find_bodies([RIGHT_EE_BODY, LEFT_EE_BODY], preserve_order=True)
    right_id, left_id = ee_ids[0], ee_ids[1]
    forces_t = torch.tensor(forces, device=device, dtype=torch.float32)
    # The same-side hand grips: a +y target -> LEFT hand (slot 1), a -y target -> RIGHT (slot 0).
    # The draw pulls OUTWARD along the reach side (away from the midline) -- the worst case for
    # balance, since it drags the CoM toward the stance edge. (Body subset order is [right, left].)
    use_left = ty >= 0.0
    hand_slot = 1 if use_left else 0
    wrench = torch.zeros(len(forces), 2, 3, device=device)
    wrench[:, hand_slot, 1] = forces_t if use_left else -forces_t

    out_dir = os.path.join(HERE, "eval")
    frames_dir = os.path.join(out_dir, f"frames_{args_cli.tag}")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)
    out_mp4 = os.path.join(out_dir, f"pull_stress_{args_cli.tag}.mp4")

    cam = env.unwrapped.scene["eval_cam"]
    origins = env.unwrapped.scene.env_origins
    eyes = origins + torch.tensor([2.4, 2.4, 1.7], device=origins.device)
    targets = origins + torch.tensor([0.25, 0.0, 0.7], device=origins.device)
    cam.set_world_poses_from_view(eyes, targets)

    obs = env.get_observations()
    steps = int(args_cli.seconds * 50)
    saved = 0
    with torch.inference_mode():
        for _ in range(steps):
            actions = policy(obs)
            # Hold the sustained draw on the gripping hand (world frame) for this control step.
            robot.set_external_force_and_torque(
                wrench, torch.zeros_like(wrench), body_ids=[right_id, left_id], is_global=True
            )
            obs, _, _, _ = env.step(actions)
            rgb = cam.data.output["rgb"]
            if rgb is None or rgb.shape[0] == 0:
                continue
            tiles = []
            for i in range(len(forces)):
                img = Image.fromarray(rgb[i, :, :, :3].detach().cpu().numpy().astype(np.uint8))
                d = ImageDraw.Draw(img)
                d.rectangle([0, 0, 150, 26], fill=(0, 0, 0))
                d.text((6, 6), f"draw = {forces[i]:.0f} N", fill=(255, 255, 0))
                tiles.append(np.asarray(img))
            montage = Image.fromarray(np.concatenate(tiles, axis=1))
            montage.save(os.path.join(frames_dir, f"frame_{saved:04d}.png"))
            saved += 1

    env.close()
    if saved > 0:
        encode(frames_dir, out_mp4, saved)
    else:
        print("[pull-stress] WARNING: no frames captured (camera produced no rgb).")


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)
