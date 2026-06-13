---
description: The Policy ABC and the four providers that ship — MockPolicy, Gr00tPolicy, LerobotLocalPolicy, Cosmos3Policy.
---

# Policy providers

```python
from strands_robots.policies import create_policy, list_providers

print(list_providers())   # ['cosmos3', 'groot', 'lerobot_local', 'mock', ...]

policy = create_policy("mock")                                                     # always works, no model
policy = create_policy("groot", port=5555, data_config="so100_dualcam")
policy = create_policy("lerobot_local", pretrained_name_or_path="lerobot/pi0_so100")
policy = create_policy("cosmos3", embodiment="droid", port=8000)
```

## Providers

| Provider | Class | Install extra | When to use |
|----------|-------|---------------|-------------|
| `mock` | `MockPolicy` | _(core)_ | Tests, smoke checks; sinusoidal joints, no GPU |
| `groot` | `Gr00tPolicy` | `groot-service` | NVIDIA GR00T N1.5/N1.6/N1.7 over ZMQ |
| `lerobot_local` | `LerobotLocalPolicy` | `lerobot` | HF LeRobot in-process (ACT, Pi0, SmolVLA, …) |
| `cosmos3` | `Cosmos3Policy` | `cosmos3-service` | NVIDIA Cosmos 3 VLA over WebSocket |

## Policy ABC

```python
from strands_robots.policies import Policy   # strands_robots/policies/base.py

class MyPolicy(Policy):
    # three abstract methods — must implement all:
    async def get_actions(self, observation_dict: dict, instruction: str, **kw) -> list[dict]: ...
    def set_robot_state_keys(self, keys: list[str]) -> None: ...
    @property
    def provider_name(self) -> str: ...

    # optional overrides:
    @property
    def requires_images(self) -> bool: return True   # False for state-only policies
    def reset(self, seed=None): pass                  # clear episode state; default no-op
    # sync helper provided by base: get_actions_sync(obs, instruction, **kw) -> list[dict]
```

## Factory

```python
from strands_robots.policies import register_policy

register_policy("my_prov", lambda: MyPolicyClass, aliases=["mp"])
policy = create_policy("my_prov")
```

Smart URI strings also resolve: `"zmq://localhost:5555"` → groot; `"cosmos3://host:8000"` → cosmos3.

## In simulation

```python
# Provider name + kwargs in policy_config={}
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_provider="groot",
               policy_config={"port": 5555, "data_config": "so100_dualcam"},
               duration=10.0)

# Pre-built instance via policy_object=
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_object=create_policy("groot", port=5555, data_config="so100_dualcam"),
               duration=10.0)
```

`LerobotLocalPolicy` requires `export STRANDS_TRUST_REMOTE_CODE=1` (raises `UntrustedRemoteCodeError` otherwise).

## See also

- [GR00T](groot.md) — ZMQ server, 27 embodiments, container lifecycle.
- [LeRobot Local](lerobot-local.md) — in-process HF models, RTC.
- [Cosmos 3](cosmos3.md) — NVIDIA Cosmos 3 omnimodal VLA.
- [Custom policies](custom-policies.md) — implement the ABC.
