#!/usr/bin/env python3
"""Agent-driven data collection + native streaming read-back.

The physical-AI data loop in one screen: a Strands Agent records a
LeRobotDataset from one natural-language prompt, then we stream it straight
back with ``sim.stream_dataset(...)`` — no torchcodec/av plumbing in user code
(it's a declared dependency of the ``[lerobot]`` extra).

Run:
    python examples/06_agent_collect_and_stream.py

Dependencies:
    pip install "strands-robots[sim-mujoco,lerobot]" strands-agents
    AWS credentials for Bedrock (or any strands-agents model provider).
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")  # offscreen GL

from strands import Agent

from strands_robots import Robot

DATASET_ROOT = "/tmp/strands_agent_dataset"
REPO_ID = "local/agent_demo"

# Robot() is a Strands AgentTool — hand it straight to an Agent.
sim = Robot("so100", mesh=False)
agent = Agent(tools=[sim])

# 1. COLLECT — one prompt drives scene + cameras + policy + recording.
agent(
    f"Create a world with the so100 robot. Add a red cube at [0.2, 0.0, 0.05] "
    f"and a blue cube at [0.25, 0.05, 0.05]. Add a front camera looking at them. "
    f"Start recording a LeRobot dataset (repo_id='{REPO_ID}', root='{DATASET_ROOT}', "
    f"fps=30, overwrite=True, task='pick up the red cube', cameras=['front']). "
    f"Run the mock policy "
    f"for 60 steps with instruction 'pick up the red cube'. Stop recording."
)

# 2. STREAM — read the dataset back lazily (Phase 3). Native facade method,
#    same object that recorded it. Camera frames are decoded on the fly from
#    the MP4 shards (torchcodec, shipped by the [lerobot] extra); state/action
#    come from the parquet shards. Nothing is re-materialized to disk.
#    shuffle=False only fixes the seed; the reader re-shards and yields from a
#    random reservoir, so max_num_shards=1 + buffer_size=1 are what make this
#    read capture order. Drop all three to train (the shuffle is wanted there).
reader = sim.stream_dataset(
    REPO_ID,
    root=DATASET_ROOT,
    shuffle=False,
    max_num_shards=1,
    buffer_size=1,
)

print(f"\nStreaming {reader.num_episodes} episode(s), {reader.num_frames} frames @ {reader.fps} fps")
print(f"cameras: {reader.meta.video_keys}")
for i, frame in enumerate(reader):
    cams = {k: tuple(frame[k].shape) for k in frame if k.startswith("observation.images.")}
    print(
        f"  frame {i}: state{tuple(frame['observation.state'].shape)} action{tuple(frame['action'].shape)} cams={cams}"
    )
    if i >= 2:
        break

# 3. (Optional) Sync the finished dataset into a mutable HF Storage Bucket for
#    collection — Xet-deduped, overwrite-in-place (Phase 1/2). Use the
#    lifecycle-independent helper, which syncs an on-disk dataset without a live
#    recording session (needs `hf auth login` + huggingface_hub >= 1.0). Set
#    STRANDS_DEMO_BUCKET="your-org/robot-fave" to run it; it stays a no-op
#    otherwise so the default path needs no Hub credentials.
bucket = os.environ.get("STRANDS_DEMO_BUCKET")
if bucket:
    from strands_robots import sync_dataset_to_bucket

    result = sync_dataset_to_bucket(DATASET_ROOT, bucket)
    if result.get("status") == "success":
        print(f"\nSynced to bucket: {result['bucket_uri']}")
    else:
        # Surface the failure instead of reporting a sync that did not happen.
        raise RuntimeError(f"bucket sync failed: {result.get('message')}")
else:
    print("\nDone. Set STRANDS_DEMO_BUCKET=org/name to sync this dataset to an HF Storage Bucket.")
