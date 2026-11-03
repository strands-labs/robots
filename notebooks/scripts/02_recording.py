# %% [markdown]
# # 02 · Recording
#
# Add `record=` to any run and it becomes one episode of a LeRobot v3 dataset, written by LeRobot's own writer.
# Run again on the same folder and it adds an episode. Anything that trains on LeRobot data can read it.
#
# ![one run, one episode](media/02/run_to_dataset.svg)
#
# Needs LeRobot with its dataset extra, on top of `.[mujoco]`:
#
# ```bash
# uv pip install -e ".[mujoco,lerobot]"
# uv pip install "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@53feaef37b67e8f4c23b6953e880ff535e39d6fe"
# ```

# %%
import json
import math
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from strands_robots import Policy, Robot, Simulation

# LeRobot's dataset stack is chatty on stderr. Quiet it; errors still print.
os.environ["SVT_LOG"] = "1"  # the AV1 encoder: ~10 info lines per episode
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"  # a progress bar per saved episode
warnings.filterwarnings("ignore", "Cannot enable progress bars")  # LeRobot turns them back on after each save
import av  # noqa: E402

av.logging.set_libav_level(av.logging.ERROR)  # ffmpeg: a colour-space notice per decoded frame on macOS

MEDIA = Path("media/02") if Path("media").is_dir() else Path("notebooks/media/02")
MEDIA.mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="sr-02-"))
target = work / "so101-waves"

# %% [markdown]
# ## Three episodes, one camera
#
# A camera attached to the robot becomes a video feature (`observation.images.front`). `instruction=` becomes the
# episode's task. A dataset has an integer fps, so the run needs an integer `hz`.

# %%
def so101():
    arm = Robot("so101", sim=Simulation())
    arm.sim.camera("front", position=[0.5, -0.6, 0.42], target=[-0.08, -0.05, 0.08], size=(240, 180), attach=arm)
    return arm


def wave(sign):
    return lambda obs: {"shoulder_pan": sign * 0.9 * math.sin(3 * obs.t), "elbow_flex": 0.6 + 0.4 * math.sin(2 * obs.t)}


arm = so101()
for sign, task in [(1, "wave right"), (-1, "wave left"), (1, "wave right")]:
    r = arm.run(wave(sign), hz=20, steps=40, record=target, instruction=task)
    print(f"{task:10s} steps={r.steps} stopped_by={r.stopped_by} recorded={Path(r.recorded).name}")
arm.sim.close()

info = json.loads((target / "meta" / "info.json").read_text())
print({k: info[k] for k in ("codebase_version", "robot_type", "total_episodes", "total_frames", "fps")})

# %%
print(*sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()), sep="\n")

# %% [markdown]
# ## A second run resumes
#
# A new world, a new robot, the same folder: the episode index continues. A run that would change the dataset's shape,
# here a different rate, is refused before the robot moves.

# %%
arm = so101()
arm.run(wave(-1), hz=20, steps=30, record=target, instruction="wave left")
info = json.loads((target / "meta" / "info.json").read_text())
print("after resume:", info["total_episodes"], "episodes,", info["total_frames"], "frames")
try:
    arm.run(wave(1), hz=10, steps=5, record=target)
except ValueError as e:
    print("ValueError:", str(e).replace(str(work), "<tmp>"))
print("time did not move for the refused run:", round(arm.sim.time, 2), "s")

# %% [markdown]
# ## Read it back with LeRobot
#
# `LeRobotDataset` is LeRobot's reader; nothing from strands-robots is needed to load the data.

# %%
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/so101-waves", root=target)
print("episodes", ds.num_episodes, "| frames", ds.num_frames, "| fps", ds.fps)
print("state names:", ds.meta.features["observation.state"]["names"])
first = ds[0]
print({k: tuple(v.shape) for k, v in first.items() if hasattr(v, "shape") and v.ndim > 0}, "| task:", first["task"])
print("frames per episode:", np.bincount([int(ds[i]["episode_index"]) for i in range(ds.num_frames)]).tolist())

# %% [markdown]
# The same read in a fresh Python process, so nothing cached in this one can make it look right.

# %%
code = (
    "import sys; from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; "
    "d = D('local/x', root=sys.argv[1]); print(d.num_episodes, d.num_frames, d[d.num_frames - 1]['task'])"
)
out = subprocess.run([sys.executable, "-c", code, str(target)], capture_output=True, text=True, timeout=300)
print("fresh process (episodes, frames, last task):", out.stdout.strip().splitlines()[-1])

# %% [markdown]
# ## The frames in the dataset
#
# Decoded from the dataset's own MP4 files, not from the simulator: two frames from each episode, in order.

# %%
cam = "observation.images.front"
starts = [0, *np.cumsum(np.bincount([int(ds[i]["episode_index"]) for i in range(ds.num_frames)]))[:-1]]
picks = [int(s) + off for s in starts for off in (5, 25)]
tiles = [(ds[i][cam].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8) for i in picks]
strip = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])  # 2 rows: episodes 0-1, then 2-3
Image.fromarray(strip).resize((640, 240)).save(MEDIA / "frame_strip.png")
print("frames", picks, "| strip", strip.shape, "| pixel std", round(float(strip.std()), 1))

# %% [markdown]
# ![frames read back from the dataset](media/02/frame_strip.png)
#
# ## `meta/units.json`: what the numbers mean
#
# Written next to LeRobot's own `meta/`. A function policy works in radians, so this dataset says `rad`.

# %%
print((target / "meta" / "units.json").read_text())

# %% [markdown]
# With `units="lerobot"` the policy sees and returns LeRobot's normalised units (SO-101: -100..100, gripper 0..100),
# and the dataset stores those, and says so. A radian run into that folder would be refused.
# A `{joint: value}` mapping works here too, in the same units; joints it leaves out hold.

# %%
lr_target = work / "so101-lerobot-units"
seen = []


def wave_lerobot(obs):
    seen.append(obs["shoulder_pan"])  # already in lerobot units here
    return {"shoulder_pan": 60 * math.sin(3 * obs.t), "elbow_flex": 30.0}


arm = so101()
arm.run(wave_lerobot, hz=20, steps=30, units="lerobot", record=lr_target, instruction="wave")
said = json.loads((lr_target / "meta" / "units.json").read_text())
print("units:", said["units"], "| robot:", said["robot"])
for k in ("shoulder_pan.pos", "gripper.pos"):
    print(f"  {k:17s} range {said['joints'][k]['range']} <-> rad {[round(x, 3) for x in said['joints'][k]['rad']]}")
print(f"the policy saw shoulder_pan from {min(seen):.1f} to {max(seen):.1f} (lerobot units)")
try:
    arm.run(wave(1), hz=20, steps=3, record=lr_target)
except ValueError as e:
    print("ValueError:", str(e).replace(str(work), "<tmp>"))
arm.sim.close()

# %% [markdown]
# ## Replay an episode
#
# `Policy("<dataset>?episode=N")` plays back that episode's `action` column at the dataset's fps, in the units
# `meta/units.json` names.

# %%
arm = so101()
replay = Policy(f"{target}?episode=1")
r = arm.run(replay, seconds=2, record=work / "replayed")
print(type(replay).__name__, "|", r.steps, "steps at", r.hz_target, "Hz, stopped_by", r.stopped_by)
again = LeRobotDataset("local/replayed", root=work / "replayed")
a = np.stack([again[i]["action"].numpy() for i in range(again.num_frames)])
b = np.stack([ds[int(starts[1]) + i]["action"].numpy() for i in range(again.num_frames)])
print(f"replayed actions vs episode 1's: {len(a)} frames, max |diff| {np.abs(a - b).max():.2e} rad")
arm.sim.close()

# %% [markdown]
# ## Hub targets
#
# Only the `hf://owner/name` spelling means the Hugging Face Hub; any other string is a folder. Nothing is uploaded
# unless the run also passes `push=True`, and `push=True` needs an `hf://` target.

# %%
from strands_robots.recording import is_repo_id

for t in ["hf://you/so101-waves", "you/so101-waves", "data/so101-waves"]:
    print(f"{t!r:24} -> Hub" if is_repo_id(t) else f"{t!r:24} -> folder")
arm = so101()
try:
    arm.run(wave(1), hz=20, steps=2, record=work / "plain", push=True)
except ValueError as e:
    print("ValueError:", str(e).replace(str(work), "<tmp>"))
arm.sim.close()

# %% [markdown]
# A run that raises leaves nothing behind; a run you stop (or e-stop) keeps what it recorded.
#
# ## Next
#
# [03](03_agent_builds_a_world.ipynb): an agent builds a scene from language.
