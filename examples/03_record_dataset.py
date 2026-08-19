#!/usr/bin/env python3
"""Record a demonstration as a LeRobotDataset - from sim, push to HF Hub.

Goal: Show the recording lifecycle (start -> run policy -> stop) produces
a training-ready LeRobotDataset with zero manual feature wrangling.

Dependencies: pip install "strands-robots[sim-mujoco,lerobot]"
Expected output: "Episode saved ... 100 frames, 1 episode(s)" under /tmp.
Runtime: ~3 seconds.
"""

import sys

from strands_robots import MockPolicy, Robot

# Robot("so100") already builds the world and adds the "so100" robot.
sim = Robot("so100", mesh=False)
sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0, 0.05])

# Start recording. Pass an explicit local root= directory: without it lerobot
# resolves repo_id into the read-only Hub snapshot cache, where DatasetWriter
# cannot be created (recorder init fails). Features are auto-inferred from the
# robot - no manual schema wrangling.
start = sim.start_recording(
    repo_id="local/my_demo",
    root="/tmp/strands_demo_dataset",
    fps=30,
    task="pick up the red cube",
    overwrite=True,
)
if start["status"] != "success":
    # Surface the real failure instead of pretending the recording worked.
    print(f"start_recording failed: {start['content'][0]['text']}", file=sys.stderr)
    raise SystemExit(1)

# Run the policy - each step is automatically captured.
# control_frequency must equal the recording's fps: the recorder writes one
# frame per control step with no decimation, so the 50 Hz default rollout
# against this 30 fps recording is refused rather than written at a distorted
# timestamp rate, and the episode would land with zero frames.
rollout = sim.run_policy(
    robot_name="so100",
    policy_object=MockPolicy(),
    instruction="pick up the red cube",
    n_steps=100,
    control_frequency=30.0,
)
if rollout["status"] != "success":
    # A refused rollout captures nothing, so stop_recording below would report
    # an empty dataset instead of the reason. Surface it at its source.
    print(f"run_policy failed: {rollout['content'][0]['text']}", file=sys.stderr)
    raise SystemExit(1)

# Finalize - writes parquet + video, ready for lerobot training scripts.
# The response text reports the actual frame/episode count and on-disk path,
# so you can confirm frames were captured rather than trusting a bare status.
result = sim.stop_recording()
print(result["content"][0]["text"])
if result["status"] != "success":
    raise SystemExit(1)

# To push to HF Hub instead: use repo_id="your_user/dataset_name"
# and call sim.stop_recording(push_to_hub=True) with HF_TOKEN set.
