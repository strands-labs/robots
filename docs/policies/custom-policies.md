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
| `required_bodies` (property) | no | `()` |
| `reset(seed=None)` | no | no-op |
| `get_actions_sync(...)` | no | sync wrapper |

## Declaring body poses your policy needs

`get_actions` receives per-joint state plus, for a floating-base robot, the base
pose (`base_pos`, `base_quat`, `base_lin_vel`, `base_ang_vel`). Do not assume the
per-joint half is non-empty: an aerial robot is actuated by forces applied at
sites on its airframe rather than by joints, so it declares no joint at all
besides its floating base and its entire observation *is* the base pose (its
`robot_action_keys` are rotor names such as `thrust1`, not joint names). A policy
that indexes a joint key unconditionally cannot fly one. A whole-body
**motion-mimic tracker** needs more than that: its network is conditioned on the
world orientation of one *anchor* link -- `torso_link` on a Unitree G1 -- and
`base_quat` is the **pelvis**, separated from the torso by the three waist
joints. Reading `base_quat` as if it were the anchor feeds the network the wrong
frame whenever the waist is not neutral (measured on a G1 sweeping its waist:
the two diverge by up to 42 degrees).

Declare the links you need and the runtime supplies them:

```python
class MyTracker(Policy):
    @property
    def required_bodies(self) -> tuple[str, ...]:
        return ("torso_link",)

    async def get_actions(self, obs, instruction="", **kw):
        anchor_quat = obs["body.torso_link.quat"]   # world w, x, y, z
        ...
```

For each declared body the observation gains four keys:

| Key | Contents |
|---|---|
| `body.<name>.pos` | world position x, y, z (m) |
| `body.<name>.quat` | world orientation w, x, y, z |
| `body.<name>.lin_vel` | world linear velocity x, y, z (m/s) |
| `body.<name>.ang_vel` | world angular velocity x, y, z (rad/s) |

Notes:

- Names are resolved once, before the rollout. A body the scene does not contain
  raises there -- listing what is available -- rather than going missing from
  every observation and being read as a zero pose.
- Declaring nothing (the default) leaves the observation exactly as the backend
  produced it, so policies that do not need a link pay no extra read.
- In a multi-robot scene MuJoCo body names carry the robot's namespace prefix
  (`alice/Lower_Arm`), the same spelling `get_body_state` resolves.
- Requires a backend that implements `get_body_state` (MuJoCo, Isaac).

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
