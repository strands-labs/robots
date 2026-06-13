---
description: Runnable example scripts — links to the repo's examples/ directory.
---

# Examples

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

- [Quickstart](../getting-started/quickstart.md) — minimal starter.
- [Cosmos3Policy](../policies/cosmos3.md) — Cosmos 3 details.
- [LerobotLocalPolicy](../policies/lerobot-local.md) — MolmoAct2 and other local models.
