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
    f"fps=30, overwrite=True, task='pick up the red cube'). Run the mock policy "
    f"for 60 steps with instruction 'pick up the red cube'. Stop recording."
)

# 2. STREAM — read the dataset back lazily (Phase 3). Native facade method,
#    same object that recorded it. Camera frames are decoded on the fly from
#    the MP4 shards (torchcodec, shipped by the [lerobot] extra); state/action
#    come from the parquet shards. Nothing is re-materialized to disk.
reader = sim.stream_dataset(
    REPO_ID,
    root=DATASET_ROOT,
    shuffle=False,
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

# 3. (Optional) Dump to a mutable HF Storage Bucket during collection —
#    Xet-deduped, overwrite-in-place (Phase 1/2). One kwarg on stop_recording:
#    sim.stop_recording(bucket="your-org/robot-fave")
print("\nDone. To dump to an HF Storage Bucket: stop_recording(bucket='org/name')")
