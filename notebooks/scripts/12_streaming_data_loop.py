# Paired script for notebooks/12_streaming_data_loop.ipynb.
# Run: .venv/bin/python notebooks/scripts/12_streaming_data_loop.py
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # nb: script-only

# %% [markdown]
# # 12 · The streaming data loop
#
# The companion to [Record, train, and deploy from one place with Strands Agents, LeRobot, and Hugging Face Storage
# Buckets](https://huggingface.co/blog/amazon/strands-lerobot-streaming-data-loop), on this release. Strands Robots
# records and deploys; LeRobot streams and trains; the `hf` CLI moves the data between them. Four steps: record a
# LeRobot v3 dataset, sync it into a Hugging Face Storage Bucket, stream it back and train on it with no local copy,
# and load the checkpoint as a `Policy` on the same robot.
#
# Everything runs in MuJoCo on a laptop CPU. Set `HF_BUCKET` (for example `my-org/robot-fave`, after
# `hf auth login`) to sync to a private bucket and stream from it; without it the notebook trains from the local
# folder instead. The training step needs `lerobot[training]`.

# %%
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import importlib.util
import math
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import _nb
from strands_robots import Policy, Robot, Simulation

_nb.quiet_lerobot()
BUCKET = os.environ.get("HF_BUCKET")  # <namespace>/<bucket>; created private if it does not exist
RUN_ID = f"nb12-{uuid.uuid4().hex[:6]}"  # one folder per run: a bucket keeps no revisions
TASK = "pick up the red cube"
WORK = Path(tempfile.mkdtemp(prefix="dataloop-"))
HF = shutil.which("hf") or str(Path(sys.executable).with_name("hf"))
print("bucket:", f"{BUCKET}/{RUN_ID}" if BUCKET else "none (local only)")

# %% [markdown]
# ## Step 1: record a demonstration
#
# `run(record=...)` writes LeRobot v3 through LeRobot's own writer, one episode per run, into one directory. `wave`
# is a placeholder motion so the loop runs before you have a trained policy: the episodes are structurally complete
# (joint states, actions, the camera stream) but they are not useful training data.

# %%
sim = Simulation()
arm = Robot("so101", sim=sim)
sim.add_object("red_cube", shape="box", size=[0.015] * 3, position=[0.0, -0.2, 0.015], color=[0.85, 0.1, 0.1, 1])
sim.camera("front", position=(0.5, -0.6, 0.42), target=(-0.08, -0.05, 0.08), size=(320, 240), attach=arm)


def wave(obs):
    return {"shoulder_pan": 0.6 * math.sin(3.0 * obs.t), "elbow_flex": 0.5 + 0.3 * math.sin(2.0 * obs.t)}


dataset = WORK / "cube_pick"
for episode in range(2):
    r = arm.run(wave, hz=30, seconds=4, record=dataset, instruction=TASK)
    print(f"episode {episode}: {r.steps} steps at {r.hz_target} Hz, stopped by {r.stopped_by}")
print("files:", *sorted(str(p.relative_to(dataset)) for p in dataset.rglob("*") if p.is_file()), sep="\n  ")

# %% [markdown]
# ## Step 2: sync it into a Storage Bucket
#
# A bucket is mutable, non-versioned, Xet-backed storage beside your Hub repositories. `hf sync` uploads only what
# changed, and Xet deduplicates at the byte level, so tomorrow's sync sends the new trailing shards rather than the
# whole dataset. The run goes under its own folder, `hf://buckets/<bucket>/<run_id>`.

# %%
if BUCKET:
    for cmd in ([HF, "buckets", "create", BUCKET, "--private", "--exist-ok"],
                [HF, "sync", str(dataset), f"hf://buckets/{BUCKET}/{RUN_ID}"]):
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("$", *cmd[1:], "\n ", out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "ok")
else:
    print("HF_BUCKET is not set: the dataset stays in", dataset)

# %% [markdown]
# ## Step 3: stream it back
#
# LeRobot's `StreamingLeRobotDataset` reads a bucket with `repo_type="bucket"`: camera frames decode on the fly from
# the remote MP4 shards and state and action come from the Parquet shards, so only `meta/` and a small seek index
# land on disk. The first loop reads one frame in capture order; the second wraps the reader in an ordinary
# `DataLoader` (raise `num_workers` in a script with an `if __name__ == "__main__":` guard).

# %%
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

if BUCKET:
    reader = StreamingLeRobotDataset(f"{BUCKET}/{RUN_ID}", repo_type="bucket",
                                     shuffle=False, max_num_shards=1, buffer_size=1)
    print("episodes", reader.num_episodes, "| frames", reader.num_frames, "| fps", reader.fps)
    frame = next(iter(reader))
    print("one frame:", {k: tuple(v.shape) for k, v in frame.items() if hasattr(v, "shape") and v.ndim})
    _nb.show(_nb.image((frame["observation.images.front"].permute(1, 2, 0).numpy() * 255).astype("uint8"), width=320))
    batches = torch.utils.data.DataLoader(StreamingLeRobotDataset(f"{BUCKET}/{RUN_ID}", repo_type="bucket"),
                                          batch_size=64, num_workers=0)
    frames = sum(b["action"].shape[0] for b in batches)
    print("streamed", frames, "frames in batches of 64")
else:
    ds = LeRobotDataset("local/cube_pick", root=dataset)
    print("local read: episodes", ds.num_episodes, "| frames", ds.num_frames, "| fps", ds.fps)

# %% [markdown]
# ## Step 4: train on the stream, then load the checkpoint as a Policy
#
# `lerobot-train` reads through the same engine. A bucket is streaming-only, so `--dataset.repo_type=bucket` goes
# with `--dataset.streaming=true`. Twenty CPU steps over two placeholder episodes check that the loop connects; they
# do not train a policy that grasps anything. For a real run, raise `--steps`, drop `--policy.device=cpu`, and record
# real demonstrations.

# %%
TRAINED = importlib.util.find_spec("accelerate") is not None  # lerobot[training]
out_dir = WORK / "train"
source = (["--dataset.repo_id", f"{BUCKET}/{RUN_ID}", "--dataset.repo_type=bucket", "--dataset.streaming=true"]
          if BUCKET else ["--dataset.repo_id", "local/cube_pick", "--dataset.root", str(dataset)])
cmd = [str(Path(sys.executable).with_name("lerobot-train")), "--policy.type=act", "--policy.device=cpu",
       "--policy.push_to_hub=false", *source, "--batch_size=4", "--steps=20", "--save_freq=20", "--log_freq=10",
       "--num_workers=0", f"--output_dir={out_dir}"]
if TRAINED:
    run = subprocess.run(cmd, capture_output=True, text=True)
    log = [line[line.index("step:"):] for line in (run.stdout + run.stderr).splitlines() if " step:" in line]
    print(f"lerobot-train exit {run.returncode}", *log, sep="\n  ")
else:
    print("lerobot[training] is not installed, so training is skipped: uv pip install 'lerobot[training]>=0.7,<0.8'")

# %% [markdown]
# The checkpoint loads the same way as any other policy, and runs on the same `Robot`. On hardware the constructor
# changes to `Robot("so101", port=..., id=..., cameras={"front": ...}, operator=...)`; notebooks 04 and 11 show it.

# %%
ckpt = out_dir / "checkpoints" / "last" / "pretrained_model"
if ckpt.is_dir():
    act = Policy(str(ckpt), device="cpu")
    print(act)
    before = arm.observe(images=False).joints.positions.copy()
    r = arm.run(act, hz=30, steps=30, instruction=TASK)
    moved = arm.observe(images=False).joints.positions - before
    print(f"{r.steps} steps, stopped by {r.stopped_by}; joint change (rad):", [round(float(x), 3) for x in moved])
    _nb.show(_nb.image(sim.render("front"), width=480))

# %% [markdown]
# ## Clean up
#
# The bucket run stays until you remove it: `hf buckets rm <bucket>/<run_id>/ --recursive`, or
# `hf buckets delete <bucket>` for the whole bucket.

# %%
sim.close()
print("closed; the local dataset and checkpoint are under", WORK)
if BUCKET:
    print(f"remove the bucket run with: hf buckets rm {BUCKET}/{RUN_ID}/ --recursive --yes")
