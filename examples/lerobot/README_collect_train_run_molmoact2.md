# Collect, train, and run a MolmoAct2 policy

Runnable companion to the blog post *Collect, train, and run a MolmoAct2 policy
with Strands Agents and LeRobot*. It walks the full VLA data flywheel on one
machine: **collect** demonstrations in simulation, **train** (fine-tune) upstream
in LeRobot, and **run** the result back through the same `Robot()` abstraction.

> **Status: pending GPU validation.** The script is verified statically (every
> Strands API call checked against the SDK; inputs aligned with the
> `allenai/MolmoAct2-SO100_101` model card) but has not yet been executed on a
> GPU. The two items marked **[verify]** below are confirmed during the first
> run on hardware.

| File | What it is |
|------|------------|
| [`collect_train_run_molmoact2.py`](./collect_train_run_molmoact2.py) | The runnable CLI (`collect` / `train` / `run` subcommands). |
| `README_collect_train_run_molmoact2.md` (this file) | Setup, the three phases, hardware notes. |

---

## Why MolmoAct2

[MolmoAct2](https://huggingface.co/allenai/MolmoAct2-SO100_101) is Allen AI's open
vision-language-action model. The `SO100_101` checkpoint is fine-tuned for the
SO-100/101 arms and is explicitly intended for **both inference and further
fine-tuning**. We use the pretrained checkpoint as a *teacher* to bootstrap
demonstrations, then fine-tune it on those demonstrations into a sharpened,
task-specific policy.

---

## Requirements

MolmoAct2's policy class shipped in LeRobot **after** the 0.5.1 PyPI release, so
install LeRobot from source alongside the `molmoact2` extra (the documented path
in `pyproject.toml`):

```bash
uv pip install "strands-robots[sim-mujoco,molmoact2,mesh]" \
    "lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git"

export STRANDS_TRUST_REMOTE_CODE=1   # MolmoAct2 loads with trust_remote_code
export MUJOCO_GL=egl                 # headless rendering (SSH / CI / cloud GPU)
```

**Hardware:** a CUDA GPU is required for every phase (MolmoAct2 is a 16B-class
VLA). A **g6.4xlarge (NVIDIA L4, 24GB)** is sufficient. Per the model card:

- `bfloat16` runs under ~16GB **(recommended on the L4)**.
- `float32` needs ~24-26GB (tight on a 24GB card; use only if you have headroom).

---

## Phase 1: Collect

Drives the pretrained MolmoAct2 teacher in MuJoCo and records a LeRobotDataset of
pick-and-place demonstrations with per-episode domain randomization.

```bash
# 50 episodes into a local dataset
python collect_train_run_molmoact2.py collect --episodes 50

# push to the Hub instead of keeping it local
export HF_TOKEN=hf_...
python collect_train_run_molmoact2.py collect --episodes 50 --hf-user my_user
```

The dataset lands at `~/.cache/huggingface/lerobot/<repo_id>/` in LeRobot v3
format (parquet + per-camera MP4).

Notes:
- The scene uses **two cameras** (`image`, `wrist_image`), matching the
  checkpoint's two-RGB-view training. Camera order does not matter for this
  checkpoint.
- **[verify]** that driving the teacher via `run_policy(policy_object=...)` while
  recording produces usable demonstrations (vs. the lower-level
  `get_actions` -> `set_joint_positions` loop in `molmoact2_sim_pickplace.py`).

## Phase 2: Train (runs upstream in LeRobot)

`strands-robots` ships data collection and inference, **not** training. MolmoAct2
is integrated into LeRobot as a policy, so fine-tuning uses LeRobot's standard
trainer on the LeRobot v3 dataset you just collected. Print the command:

```bash
python collect_train_run_molmoact2.py train
```

Run the printed command on the GPU box. It is a long job, so run/monitor it
yourself rather than from inside this script.

- **[verify]** the exact `lerobot.scripts.train` flags for the MolmoAct2 policy.
  The authoritative reference is the LeRobot source tree at
  `docs/source/molmoact2.mdx` (the MolmoAct2 repo,
  [`allenai/molmoact2`](https://github.com/allenai/molmoact2), integrates LeRobot
  on its `molmoact2-policy` branch). Confirm with `... train --help`.

## Phase 3: Run

Load your fine-tuned checkpoint back through the same `Robot()` abstraction.

```bash
python collect_train_run_molmoact2.py run \
    --checkpoint outputs/molmoact2_ft/checkpoints/last
```

Swap `Robot("so101")` to `mode="real"` with `cameras={...}` to deploy the same
code to a physical SO-101.

---

## The round trip in one place

```
collect (pretrained MolmoAct2 -> demos -> LeRobotDataset -> Hub)
   │
   ▼
train   (LeRobot fine-tunes MolmoAct2 on your dataset, on the GPU box)
   │
   ▼
run     (your fine-tuned checkpoint -> Robot() in sim, or mode="real" on hardware)
```

## Safety

Per the MolmoAct2 model card: validate model outputs before deploying on
hardware, bound the action space to your controller limits (speed, workspace,
torque, contact force), keep an emergency stop available, and operate only under
human supervision. In Strands Robots, the mesh's actuating actions are gated
behind a human-in-the-loop approval by default.
