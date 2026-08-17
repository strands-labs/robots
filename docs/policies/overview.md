---
description: The Policy ABC and every provider that ships - mock, groot, lerobot_local, lerobot_async, cosmos3, vera, remote, curobo, moveit2, wbc, wbc_gait, motionbricks, kimodo, protomotions.
---

# Policy providers

`strands_robots` ships several policy providers. The registry is the ground
truth - list the providers with:

```bash
python -c 'from strands_robots.policies import list_providers; print(list_providers())'
# ['cosmos3', 'curobo', 'groot', 'kimodo', 'lerobot_async', 'lerobot_local', 'mock', 'motionbricks', 'moveit2', 'protomotions', 'remote', 'vera', 'wbc', 'wbc_gait']
```

`create_policy` also accepts each provider's declared aliases and shorthands,
which `list_providers()` does not repeat. `list_aliases()` reports those, so
the two together are every spelling `create_policy` accepts:

```bash
python -c 'from strands_robots.policies import list_aliases; print(list_aliases())'
# {'lerobot': 'lerobot_local', 'sonic': 'wbc', 'moveit': 'moveit2', 'cumotion': 'curobo', ...}
```

```python
from strands_robots.policies import create_policy, list_aliases, list_policy_types, list_providers

print(list_providers())     # sorted provider names (registry ground truth)
print(list_aliases())       # alias -> canonical, e.g. {'sonic': 'wbc', 'lerobot': 'lerobot_local'}
print(list_policy_types())  # lerobot_local policy_type strings: ['act', 'diffusion', 'smolvla', ...]

policy = create_policy("mock")                                                     # always works, no model
policy = create_policy("groot", port=5555, data_config="so100_dualcam")
policy = create_policy("lerobot_local", pretrained_name_or_path="lerobot/pi0_so100")
policy = create_policy("cosmos3", embodiment="droid", port=8000)
policy = create_policy("remote", endpoint="ws://gpu-box:8765")
```

## Providers

Every row below is a registered provider (`create_policy("<name>")`). The table
is kept in sync with `list_providers()` by a regression test
(`tests/test_docs_policy_coverage.py`), so it can never silently drift.

| Provider | Class | Install extra | When to use |
|----------|-------|---------------|-------------|
| [`mock`](custom-policies.md) | `MockPolicy` | _(core)_ | Tests, smoke checks; sinusoidal joints, no GPU. Reference minimal `Policy` (documented inline + custom-policies) |
| [`groot`](groot.md) | `Gr00tPolicy` | `groot-service` | NVIDIA GR00T N1.5/N1.6/N1.7 over ZMQ |
| [`lerobot_local`](lerobot-local.md) | `LerobotLocalPolicy` | `lerobot` | HF LeRobot in-process (ACT, Pi0, SmolVLA, MolmoAct2, ...) |
| [`lerobot_async`](lerobot-async.md) | `LerobotAsyncPolicy` | `lerobot-async` | Offload a LeRobot policy to a GPU box over lerobot's native async-inference gRPC transport; the robot host stays light. Edge-device inference |
| [`cosmos3`](cosmos3.md) | `Cosmos3Policy` | `cosmos3-service` | NVIDIA Cosmos 3 omnimodal VLA over WebSocket |
| [`vera`](vera.md) | `VeraPolicy` | `vera` | MIT VERA video-to-action (DFoT/WAN planner + Jacobian IDM) over a containerized GPU server |
| [`remote`](remote.md) | `RemotePolicy` | `inference` | Offload a large policy to a GPU box: forward observations to a remote `PolicyServer` over WebSocket, get back action chunks. Edge-device inference |
| [`curobo`](curobo.md) | `CuroboPolicy` | `curobo` | NVIDIA cuRobo collision-aware motion planning, in-process CUDA (non-VLA) |
| [`moveit2`](moveit2.md) | `MoveIt2Policy` | `moveit2` | MoveIt2 motion planning over a ROS 2 sidecar (ZMQ), no in-venv ROS 2 deps (non-VLA) |
| [`wbc`](wbc.md) | `WBCPolicy` | `wbc` | NVIDIA GR00T Whole-Body-Control (SONIC) Unitree G1 humanoid locomotion, in-process ONNX, no GPU (non-VLA) |
| [`wbc_gait`](wbc_gait.md) | `WBCGaitPolicy` | `wbc` | WBC gait-clock variant: single ONNX policy, 95-dim obs + bipedal phase clock (non-VLA) |
| [`motionbricks`](motionbricks.md) | `MotionBricksPolicy` | `motionbricks` | Generative kinematic Unitree G1 motion (style-driven: walk/stealth_walk/...), in-process torch (non-VLA) |
| [`kimodo`](kimodo.md) | `KimodoPolicy` | `kimodo` | Text-to-motion diffusion for the Unitree G1 (free-form prompt -> kinematic qpos), in-process torch (non-VLA) |
| [`protomotions`](protomotions.md) | `ProtoMotionsPolicy` | `protomotions` | ProtoMotions Generalist Tracking Policy: tracks a reference motion clip on the Unitree G1, in-process ONNX (non-VLA) |

## Policy ABC

```python
from strands_robots.policies import Policy   # strands_robots/policies/base.py

class MyPolicy(Policy):
    # three abstract methods - must implement all:
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

`policy_config` must be a dict (it is forwarded to `create_policy` with `**`); the same holds for the
per-call `policy_kwargs`. Any other shape - a `"port=5555"` string, a list of pairs, an unparsed JSON
blob - is rejected by `run_policy` / `start_policy` / `eval_policy` / `evaluate_benchmark` with a
structured error naming the parameter, before any policy is created.

`LerobotLocalPolicy` requires `export STRANDS_TRUST_REMOTE_CODE=1` (raises `UntrustedRemoteCodeError` otherwise).

## See also

- [GR00T](groot.md) - ZMQ server, 27 embodiments, container lifecycle.
- [LeRobot Local](lerobot-local.md) - in-process HF models, RTC.
- [LeRobot Async](lerobot-async.md) - offload a LeRobot policy to a gRPC `PolicyServer` (edge offload).
- [MolmoAct2 (SO-100/101)](molmoact2.md) - action/observation contract for the SO-arm checkpoints.
- [Persistent worker](persistent-worker.md) - load once, reuse across rollouts; cache controls + telemetry.
- [Cosmos 3](cosmos3.md) - NVIDIA Cosmos 3 omnimodal VLA.
- [VERA](vera.md) - MIT video-to-action planner + Jacobian IDM over a GPU server.
- [Remote](remote.md) - forward observations to a remote `PolicyServer` over WebSocket (edge offload).
- [cuRobo](curobo.md) - in-process collision-aware motion planning (non-VLA, GPU).
- [MoveIt2](moveit2.md) - ROS 2 sidecar collision-aware planning (non-VLA, no in-venv ROS 2).
- [WBC](wbc.md) - GR00T Whole-Body-Control (SONIC) G1 locomotion (non-VLA, in-process ONNX).
- [WBC gait-clock variant](wbc_gait.md) - single-ONNX gait-clock G1 controller (non-VLA).
- [MotionBricks](motionbricks.md) - generative kinematic G1 motion (non-VLA, in-process torch).
- [Kimodo](kimodo.md) - text-to-motion diffusion for the G1 (non-VLA, in-process torch).
- [ProtoMotions](protomotions.md) - GTP reference-motion tracker for the G1 (non-VLA, in-process ONNX).
- [Custom policies](custom-policies.md) - implement the ABC.
