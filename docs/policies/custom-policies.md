---
description: Implement the Policy ABC, register the provider, plug it into Robot.run_policy. Full walkthrough.
---

# Custom policies

```python
# my_policy.py
from strands_robots.policies import Policy, register_policy

class MyPolicy(Policy):
    async def get_actions(self, observation_dict, instruction, **kwargs):
        return [{"motor.0": 0.5, "motor.1": -0.2}]   # list of action dicts

    def set_robot_state_keys(self, keys: list[str]) -> None:
        self._keys = keys

    @property
    def provider_name(self) -> str:
        return "my_provider"

    @property
    def requires_images(self) -> bool:
        return False   # True (default) = cameras required; False = state-only

register_policy("my_provider", lambda: MyPolicy, aliases=["mine"])
```

```python
# usage
import my_policy                              # side-effect: runs register_policy
from strands_robots.policies import create_policy
from strands_robots import Robot

policy = create_policy("my_provider")         # or "mine"
sim = Robot("so100")
sim.run_policy(robot_name="so100", instruction="do something",
               policy_object=policy, duration=5.0)
```

## Permanent registration (JSON)

Add to `strands_robots/registry/policies.json`:

```json
{
  "my_provider": {
    "module": "my_pkg.my_policy",
    "class": "MyPolicy",
    "shorthands": ["mine"],
    "description": "My custom policy."
  }
}
```

The factory imports lazily on first use.

Aliases and shorthands are validated on load: each must be unique across providers and must not collide with a *different* provider's canonical name (that would silently shadow it, since lookups resolve through the alias/shorthand map before the canonical name). Listing a provider's own name in its `shorthands` is allowed and idiomatic -- it is how the bare name resolves. A colliding entry raises `ValueError` at registry load.

## ABC contract

| Method / property | Abstract | Default |
|---|---|---|
| `async get_actions(obs, instruction, **kw) -> list[dict]` | yes | - |
| `set_robot_state_keys(keys)` | yes | - |
| `provider_name` (property) | yes | - |
| `requires_images` (property) | no | `True` |
| `reset(seed=None)` | no | no-op |
| `get_actions_sync(...)` | no | sync wrapper |

## Action value convention

`get_actions` returns a `list[dict]` -- one dict per control tick, each mapping a
robot state key (joint/actuator name) to its **target value** for that tick. The
value MUST be **JSON / python-native**:

- a python `float` for a single-DOF actuator, or
- a `list[float]` for a multi-DOF actuator group.

Do **not** return raw `np.ndarray` objects. If your policy computes actions with
numpy / torch, coerce before returning (`float(v)` for scalars, `v.tolist()` for
arrays). This lets downstream consumers treat every provider's output uniformly
(`float(v)` on a scalar, `len(v)` on a group) regardless of the policy's internal
compute backend. The list length is the action-chunk horizon; consumers execute
it at a fixed control rate (e.g. 50Hz). See `strands_robots/policies/mock.py` for
the canonical reference.

## See also

- [Policy overview](overview.md) - factory, providers.
- [cuRobo](curobo.md) - reference non-VLA goal-kwargs planner.
- [Architecture](../architecture.md)
- `strands_robots/policies/mock.py` - minimal reference implementation.
