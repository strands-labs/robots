---
description: GPU-parallel RL through the isaaclab trainer, with Isaac Lab in its own venv.
---

# Isaac Lab training

The `isaaclab` trainer runs `python -m isaaclab train` in a separate Isaac Lab venv and reads its log; strands-robots never imports it (pins conflict; Kit exits the closing process).

```bash
uv venv --python 3.12 ~/il && uv pip install --python ~/il/bin/python --prerelease=allow \
  --index https://pypi.nvidia.com --index-strategy unsafe-best-match "isaaclab[rsl-rl,isaacsim]==3.0.0rc1"
export ISAACLAB_PYTHON=~/il/bin/python OMNI_KIT_ACCEPT_EULA=YES   # the EULA is yours to accept
```

```python title="sketch"
train_policy(action="train", provider="isaaclab", steps=50, output_dir="runs",
             extra={"task": "Isaac-Cartpole", "num_envs": 4096, "physics": "newton_mjwarp", "timeout_s": 600})
```

It returns a `job_id`; `action="status"` reports rewards, `success_rate`, a failure's cause and `checkpoint_dir`; `action="stop"` ends it; `action="play"` records video; `action="record"` writes a LeRobotDataset (`extra["dataset_dir"]`); `action="export"` writes MLP (not CNN/recurrent) `rsl_rl` actors for `create_policy("rl", checkpoint_dir=...)`. `extra['rl_library']` is `rsl_rl`, `skrl` (AMP, multi-agent) or `rl_games` (Factory, Forge, AutoMate). Your own task package trains once the operator sets `STRANDS_ISAACLAB_TASK_PACKAGES=module:register_fn` (importable in that venv).


`extra['overrides']` (`env.*`/`agent.*`, checked against the task config), `agent`, `device`, `video`, `deterministic`, `base_model`, `resume` reach Isaac Lab; `learning_rate` pins `schedule=fixed`.

One L40S, 4096 envs: Cartpole 291k steps/s, G1 110k.

Caveats: Isaac Lab 3.0 is an RC; first RTX use compiles shaders (~4 min); PhysX and Newton differ, even in joint order, so export records a `deploy_contract` `create_policy("rl")` applies by name.
