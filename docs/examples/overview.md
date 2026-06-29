---
description: Runnable example scripts - links to the repo's examples/ directory.
---

# Examples

## Getting-started notebooks

New to Strands Robots? The [`examples/notebooks/`](https://github.com/strands-labs/robots/tree/main/examples/notebooks)
folder is a click-and-run series that runs end-to-end in simulation - no
hardware, no GPU, no Hugging Face credentials.

| Notebook | What it shows |
|----------|---------------|
| [`01_getting_started.ipynb`](https://github.com/strands-labs/robots/blob/main/examples/notebooks/01_getting_started.ipynb) | `Robot("so100")`, run a policy, read joint state, `create_policy()`. |
| [`02_record_and_stream.ipynb`](https://github.com/strands-labs/robots/blob/main/examples/notebooks/02_record_and_stream.ipynb) | Record a LeRobotDataset, then stream it back with `stream_dataset()`. |
| [`03_record_train_deploy.ipynb`](https://github.com/strands-labs/robots/blob/main/examples/notebooks/03_record_train_deploy.ipynb) | The full loop: record, train an ACT policy on CPU, export, and load it back. |

```bash
uv pip install "strands-robots[sim-mujoco,lerobot]" jupyterlab
jupyter lab   # open examples/notebooks/
```

## Scripts

Browse [`examples/`](https://github.com/strands-labs/robots/tree/main/examples):

| File | What it does |
|------|--------------|
| `cosmos3_sim_rollout.py` | Cosmos 3 sim rollout: spawn SO-100, connect to Cosmos 3 server, run episodes, save LeRobot v3 recording. |
| `molmoact2_so101_pickplace.py` | SO-101 pick-and-place via `LerobotLocalPolicy` with `norm_tag` / `image_keys` / `inference_action_mode`. Requires hardware + GPU. |
| `mesh_acl_example.json5` | Mesh ACL config: per-peer allow/deny rules for Zenoh mesh. |

```bash
git clone https://github.com/strands-labs/robots
cd robots
uv pip install -e ".[all]"

python examples/cosmos3_sim_rollout.py           # needs cosmos3-service + server on :8000
python examples/molmoact2_so101_pickplace.py     # requires hardware
                                                 # requires GPU
```

## See also

- [Quickstart](../getting-started/quickstart.md) - minimal starter.
- [Cosmos3Policy](../policies/cosmos3.md) - Cosmos 3 details.
- [LerobotLocalPolicy](../policies/lerobot-local.md) - MolmoAct2 and other local models.
