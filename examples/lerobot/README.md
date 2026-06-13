# LeRobot ↔ Strands example

Runnable companion to *From Hugging Face Hub to robot hardware with Strands Agents and LeRobot*. Demonstrates the full loop: build a Strands agent over the LeRobot AgentTools, record a `LeRobotDataset` in simulation, run a policy on the same robot, optionally deploy the same agent code to a physical SO-101, and broadcast across the Zenoh mesh.

| File | What it is |
|------|------------|
| [`hub_to_hardware.py`](./hub_to_hardware.py)   | CLI script with argparse flags. The runnable artefact. |
| [`hub_to_hardware.ipynb`](./hub_to_hardware.ipynb) | Notebook walkthrough with the same workflow, cell by cell. |
| `README.md` (this file) | Quick start, configuration, troubleshooting, production patterns. |

## Quick start

```bash
pip install "strands-robots[sim-mujoco,lerobot,mesh]" strands-agents

# Dev/lab mesh posture so Step 5 has a mesh to talk to
export STRANDS_MESH_LOCAL_DEV=1

python examples/lerobot/hub_to_hardware.py
```

This records one demonstration of `pick up the red cube` in simulation with the Mock policy, runs a 200-step rollout afterwards, and broadcasts a `go to home pose` instruction over the mesh. No GPU, no Docker, no Hugging Face account required. First run downloads MuJoCo Menagerie assets for the SO-100 (~30 seconds, one-time).

The resulting dataset is at:

```
~/.cache/huggingface/lerobot/local/strands-cube-pick/
├── data/chunk-000/episode_000000.parquet
├── meta/info.json
└── videos/chunk-000/observation.images.front/episode_000000.mp4
```

Open the MP4 to see the recording.

## Push to the Hugging Face Hub

```bash
export HF_TOKEN=hf_xxxxxxxxxx          # write scope
python examples/lerobot/hub_to_hardware.py --hf-user <your_hf_username>
```

The dataset lands at `https://huggingface.co/datasets/<your_hf_username>/strands-cube-pick`.

## Configuration

### LLM

Defaults to **Claude Opus 4.8 on Bedrock** (`global.anthropic.claude-opus-4-8`). The AWS region resolves from your environment (`AWS_REGION` / `AWS_DEFAULT_REGION`, then `~/.aws/config`). Opus 4.8 orchestrates the LeRobot tool surface in 8–13 tool calls per recording phase; lower-tier models work but issue more defensive state-querying calls.

Override per-run:

```bash
# Different model
python hub_to_hardware.py --model-id global.anthropic.claude-sonnet-4-6

# Different region
python hub_to_hardware.py --aws-region us-east-1

# Both via env vars
export STRANDS_BEDROCK_MODEL_ID=us.anthropic.claude-opus-4-8
export AWS_REGION=<your-region>     # e.g., us-east-1, us-west-2, eu-central-1
```

Verify the exact model ID in your AWS Bedrock console (Model catalog → Anthropic). Cross-region inference profile IDs are prefixed with `us.`, `eu.`, etc.

If `BedrockModel` init fails (model not enabled in your account, wrong region, stale ID), the script logs a warning and falls back to Strands' default model — the workflow still runs.

### Policy provider

| Flag | Requirements | When to use |
|------|--------------|-------------|
| `--policy mock` *(default)* | None | Workflow sanity-check. Random/placeholder actions — no real grasp behaviour. |
| `--policy groot --checkpoint <hf_repo>` | Docker + NVIDIA GPU | NVIDIA GR00T container; brings up a ZMQ inference service alongside the agent. |
| `--policy lerobot_local --checkpoint <hf_repo>` | GPU + `STRANDS_TRUST_REMOTE_CODE=1` | In-process LeRobot policy (ACT, Pi0, SmolVLA, Diffusion). |

Example with GR00T:

```bash
python hub_to_hardware.py \
    --policy groot \
    --checkpoint nvidia/GR00T-N1.7-LIBERO
```

Example with LerobotLocal:

```bash
export STRANDS_TRUST_REMOTE_CODE=1
python hub_to_hardware.py \
    --policy lerobot_local \
    --checkpoint lerobot/act_aloha_sim_transfer_cube_human
```

### Mesh posture

By default the Zenoh mesh refuses to start without TLS certificates *and* an ACL file. Two dev/lab shortcuts:

```bash
# One-var dev preset: permissive ACL + no auth, mesh runs without certs
export STRANDS_MESH_LOCAL_DEV=1
```

```bash
# Or skip the mesh entirely (Step 5 becomes a no-op)
export STRANDS_MESH=0
```

### Hardware

```bash
python hub_to_hardware.py \
    --mode real \
    --port /dev/ttyACM0 \
    --leader-port /dev/ttyACM1
```

Requires a calibrated SO-101 follower and leader. Calibrate once via the agent or directly:

```bash
python -c "
from strands import Agent
from strands_robots import lerobot_calibrate
Agent(tools=[lerobot_calibrate])(
    'Calibrate the so101_follower on /dev/ttyACM0'
)"
```

## CLI reference

```
--mode {sim,real}                  Execution mode (default: sim)
--policy {mock,groot,lerobot_local}  Policy provider (default: mock)
--checkpoint <hf_repo>             Required for groot / lerobot_local
--model-id <bedrock_id>            Override the LLM
--aws-region <region>              Override the Bedrock region
--port <device>                    SO-101 follower (--mode real)
--leader-port <device>             SO-101 leader (--mode real, recording)
--hf-user <username>               Push dataset to <username>/<dataset-name>
--dataset-name <name>              Dataset slug (default: strands-cube-pick)
--num-steps <int>                  Demonstration length (default: 1000)
--instruction <text>               Task instruction (default: "pick up the red cube")
--clean-cache                      Wipe local cache before recording
--skip-record / --skip-mesh        Skip individual steps
--verbose / -v                     Show prompts, tool calls, dataset state
```

## Where the dataset lives

LeRobotDatasets sit under the standard Hugging Face cache:

```
~/.cache/huggingface/lerobot/<repo_id>/
```

Where `<repo_id>` is `<hf_user>/<dataset_name>` (pushed) or `local/<dataset_name>` (local-only). Verify the contents with:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/strands-cube-pick")
print(ds.num_episodes, ds.num_frames, ds.fps, list(ds.features.keys()))
```

The same dataset is consumable by upstream LeRobot training scripts (`lerobot/scripts/train.py`) without conversion.

## Production patterns

This example records one longer episode per run. That keeps the agent-driven story honest — you tell the agent in English to record a demonstration once and the tool sequence comes out in one shot.

For production multi-episode collection, wrap the loop in Python and use direct tool dispatch for the per-iteration save:

```python
from strands_robots import Robot
from strands import Agent

robot = Robot("so100", data_config="so100_dualcam")
agent = Agent(tools=[robot])

# Setup phase via the agent (natural-language scene composition)
agent(
    "Add a red cube near the robot and a front camera looking at it. "
    "Then call start_recording with repo_id='my-dataset' at FPS 30."
)

# Deterministic per-episode loop (direct dispatch, no LLM variance)
for episode_idx in range(50):
    agent.tool.so100_sim(action="reset")
    agent.tool.so100_sim(
        action="run_policy",
        policy_provider="mock",
        instruction="pick up the red cube",
        n_steps=200,
    )
    agent.tool.so100_sim(action="save_episode")

# Finalize via the agent again
agent("Stop recording and push the dataset to the Hub.")
```

The split — agent for setup and finalization, Python loop for the per-episode boundary — pairs the agent's strength (free-form composition) with the determinism a multi-step loop needs.

## Troubleshooting

**`Failed to initialise mesh ... STRANDS_MESH_AUTH_MODE=mtls requires ...`**  
The mesh subsystem tries mTLS by default. Set `STRANDS_MESH_LOCAL_DEV=1` to use the dev/lab posture, or `STRANDS_MESH=0` to disable the mesh entirely for sim-only runs.

**`BedrockModel(...) init failed`**  
Common causes: the model isn't enabled in your AWS account, the region doesn't have the model, or the model ID is stale. Check the AWS Bedrock console (Model catalog → Anthropic). The script falls back to Strands' default model and continues, but you'll see degraded tool-orchestration quality.

**`resume() requires an explicit 'root' directory`**  
A prior run's dataset cache is on disk. Pass `--clean-cache` to wipe it, or pass a fresh `--dataset-name`.

**SVT-AV1 encoder output spam**  
The `Svt[info]:` lines come from the video codec inside LeRobot's `dataset_recorder` and aren't a Python logger we can silence cleanly. They're harmless — one block per camera per encoder init.

**Agent's narration claims things that don't match the tool calls**  
LLMs sometimes confabulate in narration. The dataset on disk is the ground truth — load it through `LeRobotDataset(...)` to check episode and frame counts. Pass `--verbose` to see the actual tool calls the agent made.

## What's next

- **Train a policy** on the recorded dataset using upstream LeRobot.
- **Swap the Mock policy** for a real one — `groot` for the NVIDIA container, `lerobot_local` for ACT/Pi0/SmolVLA/Diffusion checkpoints.
- **Run on physical hardware** by flipping `--mode real`.
- **Read the blog post** for the design background and the full architecture diagram: *From Hugging Face Hub to robot hardware with Strands Agents and LeRobot*.

## Repository

Strands Robots: https://github.com/strands-labs/robots  
Heavy simulation backends (Isaac Sim, Newton): https://github.com/strands-labs/robots-sim  
Upstream LeRobot: https://github.com/huggingface/lerobot
