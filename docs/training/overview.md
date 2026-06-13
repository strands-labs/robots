---
description: Where training lives — upstream LeRobot, Isaac-GR00T, Cosmos. strands-robots ships data + inference.
---

# Training

`strands-robots` ships data collection and policy inference. Training runs upstream.

| Want to train | Use |
|---------------|-----|
| ACT, Pi0, SmolVLA, Diffusion Policy | `lerobot` — `python -m lerobot.scripts.train` |
| GR00T fine-tune | `Isaac-GR00T` (NVIDIA) |
| Cosmos | NVIDIA Cosmos Framework — see [Cosmos3Policy](../policies/cosmos3.md) |
| Custom architecture | Read LeRobot v3 dataset with `pyarrow` / `datasets` |

## Round-trip

```python
# 1. Record
from strands_robots import Robot

sim = Robot("so100")
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=30)
for episode in range(50):
    sim.reset()
    sim.randomize(randomize_colors=True)
    sim.run_policy(robot_name="so100", instruction="pick up the cube",
                   policy_provider="mock", duration=10.0)
sim.stop_recording()

# 2. Train upstream
# uv pip install lerobot
# python -m lerobot.scripts.train policy=act dataset.root=/tmp/my_dataset

# 3. Infer with checkpoint
from strands_robots.policies import create_policy

policy = create_policy("lerobot_local", pretrained_name_or_path="path/to/checkpoint")
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_object=policy, duration=15.0)

# 4. Deploy (HardwareRobot has no run_policy — use start_task)
real = Robot("so100", mode="real", cameras={...})
real.start_task(instruction="pick up the cube", policy_port=5555)
```

## See also

- [Recording](../recording.md) — produce the dataset.
- [LerobotLocalPolicy](../policies/lerobot-local.md) — inference with a trained checkpoint.
- [Cosmos3Policy](../policies/cosmos3.md) — NVIDIA Cosmos 3 VLA.
