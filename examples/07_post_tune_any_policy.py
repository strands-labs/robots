#!/usr/bin/env python3
"""Post-tune any policy natively, then load it back - the full data loop.

Goal: Show the ``Trainer`` abstraction (peer of ``Policy``) closing the
physical-AI loop in one screen: RECORD a LeRobotDataset in sim, TRAIN a policy
with ``create_trainer(provider)`` (here lerobot ACT), EXPORT a loadable
artifact, and DEPLOY it back via ``create_policy`` - all with the SAME provider
name on both the training and inference sides.

Swap ``PROVIDER`` to "groot" or "cosmos3" and only the provider string changes:
the lerobot draccus CLI, GR00T's tyro FinetuneConfig, and Cosmos's TOML+DCP
pipeline all hide behind one ``TrainSpec`` + ``Trainer`` lifecycle.

Dependencies: pip install "strands-robots[sim-mujoco,lerobot]"
Expected output: a trained ACT checkpoint under /tmp, loaded back as a Policy.
Runtime: ~30s on CPU (2 training steps - just enough to prove the loop).
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")  # offscreen GL

from strands_robots import MockPolicy, Robot, create_policy
from strands_robots.training import TrainSpec, create_trainer

PROVIDER = "lerobot_local"  # swap -> "groot" / "cosmos3" (only this changes)
DATASET_ROOT = "/tmp/strands_post_tune_ds"
OUTPUT_DIR = "/tmp/strands_post_tune_ft"

# 1. RECORD - drive the sim with a mock policy, capture a LeRobotDataset v3.
sim = Robot("so100", mesh=False)
sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0, 0.05])
sim.start_recording(
    repo_id="local/post_tune_demo",
    root=DATASET_ROOT,
    fps=30,
    task="pick up the red cube",
    overwrite=True,
)
sim.run_policy(
    robot_name="so100",
    policy_object=MockPolicy(),
    instruction="pick up the red cube",
    n_steps=60,
)
sim.stop_recording()
print(f"Recorded LeRobotDataset -> {DATASET_ROOT}")

# 2. TRAIN - the trainer is selected by the SAME name as the inference policy.
#    For lerobot this shells out to `python -m lerobot.scripts.lerobot_train`.
trainer = create_trainer(PROVIDER, device="cpu")
spec = TrainSpec(
    dataset_root=DATASET_ROOT,
    base_model="",  # ACT from scratch (smallest CPU path)
    output_dir=OUTPUT_DIR,
    steps=2,
    save_freq=2,
    global_batch_size=2,
    extra={"policy_type": "act", "num_workers": 0},
)

problems = trainer.validate(spec)  # pure preflight - launch nothing if bad
if problems:
    raise SystemExit("Spec invalid:\n  - " + "\n  - ".join(problems))

result = trainer.train(spec)
print(f"train status: {result.status} | checkpoint: {result.checkpoint_dir}")
if result.status != "success":
    raise SystemExit(result.message)

# 3. EXPORT - a path create_policy can consume (HF-native passthrough here;
#    Cosmos would convert DCP -> safetensors under the hood).
exported = trainer.export(spec, result.checkpoint_dir)
print(f"exported artifact: {exported}")

# 4. DEPLOY - load the freshly-trained checkpoint back as a Policy and run it.
os.environ.setdefault("STRANDS_TRUST_REMOTE_CODE", "1")
policy = create_policy(exported, device="cpu")
print(f"loaded trained policy: {type(policy).__name__} (provider={policy.provider_name})")
print("\nLoop closed: record -> train -> export -> load. Swap PROVIDER to 'groot'/'cosmos3' to retarget the same flow.")
