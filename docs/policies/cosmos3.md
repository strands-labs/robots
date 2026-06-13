---
description: NVIDIA Cosmos 3 omnimodal VLA — WebSocket service, droid/umi/av/bridge embodiments, MuJoCo rollout.
---

# Cosmos 3

```bash
uv pip install "strands-robots[cosmos3-service]"   # adds msgpack + websockets; no openpi-client needed
```

```python
from strands_robots.policies import create_policy

policy = create_policy("cosmos3", embodiment="droid", port=8000)
# or: create_policy("cosmos3://localhost:8000")
```

## Start the server

```bash
python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000
# embodiment is selected client-side via create_policy(..., embodiment="droid")
```

## Parameters

```python
Cosmos3Policy(
    embodiment="droid",          # droid | umi | av | bridge
    host="localhost",
    port=8000,
    action_space=None,
    observation_mapping=None,
    action_mapping=None,
    robot=None,                  # "franka" or "panda" for built-in DROID→sim mapping
    prompt="",
    api_key=None,
    client=None,
    transport="raw",
)
```

## Embodiments

| Embodiment | Robot hardware | Strands sim asset |
|------------|----------------|-------------------|
| `droid` | Franka / DROID dataset | `"panda"` or `"franka"` |
| `umi` | UMI gripper | — |
| `av` | Autonomous vehicle cameras | — |
| `bridge` | Bridge dataset robots | — |

## Rollout

```python
from strands_robots import Robot

sim = Robot("panda")
sim.run_policy(
    robot_name="panda",
    instruction="pick up the red block",
    policy_provider="cosmos3",
    policy_config={"embodiment": "droid", "robot": "panda", "port": 8000},
    duration=15.0,
    control_frequency=50.0,
)
# see examples/cosmos3_sim_rollout.py
```

`robot="panda"` activates the built-in DROID-layout mapping (`joint_0..6/gripper` → `joint1..7/finger_joint1`). `requires_images=True`.

## See also

- [Policy overview](overview.md)
- [GR00T](groot.md)
- [LeRobot Local](lerobot-local.md)
- [Custom policies](custom-policies.md)
- [Policy providers](../policies/overview.md)
