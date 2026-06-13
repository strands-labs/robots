---
description: Every public symbol grouped by module — Robot, registry, simulation, policies, tools, dataset_recorder, mesh.
---

# API reference

## `strands_robots`

```python
import strands_robots
```

| Symbol | What | More |
|--------|------|------|
| `Robot(name, mode='sim', ...)` | Factory → `Simulation` or `HardwareRobot` | [Robot factory](getting-started/robot-factory.md) |
| `list_robots(category='all')` | Catalog query | [Robot catalog](robots/index.md) |
| `Policy` | Policy ABC | [Policies](policies/overview.md) |
| `MockPolicy` | Sinusoidal mock | [Policies](policies/overview.md) |
| `create_policy(provider, **kw)` | Policy factory | [Policies](policies/overview.md) |
| `register_policy(name, loader, aliases=None)` | Runtime registration | [Custom policies](policies/custom-policies.md) |
| `list_providers()` | Known policy providers | [Policies](policies/overview.md) |
| `Simulation` (lazy) | MuJoCo-backed AgentTool | [Simulation overview](simulation/overview.md) |
| `Gr00tPolicy` (lazy) | NVIDIA GR00T client | [GR00T](policies/groot.md) |

## `strands_robots.registry`

```python
from strands_robots.registry import (
    list_robots, resolve_name, get_robot, has_sim, has_hardware, get_hardware_type,
    list_robots_by_category, list_aliases, format_robot_table,
    register_robot, unregister_robot, list_user_robots,
    list_policy_providers, resolve_policy, import_policy_class, build_policy_kwargs,
)
```

| Symbol | What |
|--------|------|
| `list_robots(category)` | All robots in a category, or `"all"`. |
| `resolve_name(name)` | Alias → canonical name. |
| `get_robot(name)` | Full registry entry dict. |
| `has_sim(name)` / `has_hardware(name)` | Sim / real support flags. |
| `get_hardware_type(name)` | LeRobot type string for `mode="real"`. |
| `list_robots_by_category()` | `{category: [names]}`. |
| `list_aliases()` | All 106 aliases. |
| `format_robot_table()` | Pretty-printed robot table. |
| `register_robot(name, entry)` | Add user-defined robot at runtime. |
| `unregister_robot(name)` | Remove a runtime-registered robot. |
| `list_user_robots()` | Names from `register_robot`. |
| `list_policy_providers()` | Providers from `policies.json`. |
| `resolve_policy(uri)` | URI → provider name. |
| `import_policy_class(provider)` | Lazy import of provider class. |
| `build_policy_kwargs(provider, **kw)` | Normalise + validate kwargs. |

## `strands_robots.simulation`

```python
from strands_robots.simulation import (
    Simulation, SimWorld, SimRobot, SimObject, SimCamera,
    create_simulation, list_backends, register_backend,
)
from strands_robots.simulation.base import SimEngine
```

| Symbol | What |
|--------|------|
| `Simulation` | MuJoCo backend — 60+ agent actions. |
| `SimWorld`, `SimRobot`, `SimObject`, `SimCamera` | Shared dataclasses. |
| `create_simulation(backend='mujoco')` | Factory for non-`Robot()` construction. |
| `list_backends()` / `register_backend(name, cls)` | Backend registry. |
| `SimEngine` | ABC custom backends implement. |

Selected actions:

| Action | What |
|--------|------|
| `run_policy(robot_name, ...)` | Blocking policy rollout. |
| `start_policy(robot_name, ...)` | Async rollout (background thread). |
| `stop_policy(robot_name)` | Stop running policy. |
| `run_multi_policy(policies, ...)` | Synchronized multi-robot rollout, one merged frame per step. |
| `eval_policy(robot_name, n_episodes, ...)` | Multi-episode evaluation. |
| `evaluate_benchmark(benchmark_name, ...)` | Run registered benchmark. |
| `list_benchmarks()` / `register_benchmark_from_file(name, spec_path)` | Benchmark registry. |
| `replay_episode(repo_id, robot_name, ...)` | Replay a recorded episode. |

## `strands_robots.hardware_robot`

```python
from strands_robots.hardware_robot import Robot, TaskStatus, RobotTaskState
```

| Symbol | What |
|--------|------|
| `Robot` | Real-hardware AgentTool. |
| `TaskStatus` | Enum: `IDLE` / `CONNECTING` / `RUNNING` / `COMPLETED` / `STOPPED` / `ERROR`. |
| `RobotTaskState` | Dataclass: status, step count, error. |

| Method | What |
|--------|------|
| `start_task(instruction, policy_port, ...)` | Async task start. |
| `stop_task()` | Halt running policy. |
| `get_task_status()` | Return `RobotTaskState`. |
| `cleanup()` | Stop tasks, close cameras, stop mesh. |

## `strands_robots.policies`

```python
from strands_robots.policies import (
    Policy, MockPolicy, create_policy, register_policy, list_providers, UntrustedRemoteCodeError,
)
from strands_robots.policies.groot import Gr00tPolicy
from strands_robots.policies.lerobot_local import LerobotLocalPolicy
from strands_robots.policies.cosmos3 import Cosmos3Policy
```

| Symbol | What |
|--------|------|
| `Policy` | ABC: `get_actions`, `set_robot_state_keys`, `requires_images`, `provider_name`. |
| `MockPolicy` | Sinusoidal mock. `requires_images=False`. |
| `create_policy(provider, **kw)` | Resolve + construct. Accepts `zmq://`, `cosmos3://`, HF `org/model`. |
| `register_policy(name, loader, aliases)` | Runtime registration. |
| `list_providers()` | `['cosmos3', 'groot', 'lerobot_local', 'mock', + aliases]`. |
| `UntrustedRemoteCodeError` | Raised when `STRANDS_TRUST_REMOTE_CODE` is required but unset. |
| `Gr00tPolicy` | GR00T N1.5/N1.6/N1.7 via ZMQ (service) or in-process. |
| `LerobotLocalPolicy` | HF LeRobot inference (ACT, Pi0, Pi0.5, SmolVLA, …). Needs `STRANDS_TRUST_REMOTE_CODE=1`. |
| `Cosmos3Policy` | NVIDIA Cosmos 3 VLA over WebSocket. |

## `strands_robots.tools`

```python
from strands_robots.tools import (
    download_assets, gr00t_inference, lerobot_calibrate, lerobot_camera,
    lerobot_teleoperate, pose_tool, serial_tool, robot_mesh,
)
# All return {"status": "...", "content": [{"text": "..."}]}
```

See [Hardware tools](hardware/tools.md).

## `strands_robots.dataset_recorder`

```python
from strands_robots.dataset_recorder import DatasetRecorder, has_lerobot_dataset
```

| Symbol | What |
|--------|------|
| `DatasetRecorder.create(repo_id, fps, ...)` | New dataset. |
| `DatasetRecorder.resume(repo_id, root, task, ...)` | Append to existing (`lerobot>=0.5.2`). |
| `recorder.add_frame(observation, action, task=...)` | Append one frame. |
| `recorder.save_episode()` | Finalise episode. |
| `recorder.clear_episode_buffer()` | Discard buffer. |
| `recorder.finalize()` | Flush and close. |
| `recorder.push_to_hub(tags=None, private=False)` | Upload to HuggingFace. |
| `has_lerobot_dataset()` | Cached import check. |

See [Recording](recording.md).

## `strands_robots.mesh`

```python
from strands_robots.mesh import init_mesh, Mesh, InputPublisher, InputReceiver
```

| Symbol | What |
|--------|------|
| `init_mesh(robot, peer_id=None, ...)` | Attach mesh to a robot instance. |
| `Mesh` | `peer_id`, `peers`, `alive`, `send`, `broadcast`, `tell`, `emergency_stop`. |
| `InputPublisher` | Stream teleoperator actions over mesh. |
| `InputReceiver` | Receive + apply remote teleoperator actions. |

See [Multi-robot mesh](mesh.md).

## `strands_robots.benchmarks.libero`

```python
from strands_robots.benchmarks.libero import LiberoSuite
```

LIBERO task suites, BDDL parser. Install: `uv pip install "strands-robots[benchmark-libero]"`.

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `STRANDS_ASSETS_DIR` | Robot model asset cache | `~/.strands_robots/assets/` |
| `STRANDS_ROBOT_MODE` | Force `Robot()` mode | (kwarg honoured) |
| `STRANDS_TRUST_REMOTE_CODE` | Allow HF `trust_remote_code=True` | unset → blocked |
| `STRANDS_MESH` | Disable mesh globally when `false` | `true` |
| `STRANDS_MESH_AUDIT_DIR` | Safety event audit log | `~/.strands_robots/` |
| `MUJOCO_GL` | GL backend for MuJoCo | auto |
| `GROOT_API_TOKEN` | GR00T cloud inference token | (unset) |
| `STRANDS_GROOT_WIRE_LOG` | Log raw ZMQ frames when `1` | (unset) |

## See also

- [Architecture](architecture.md) — module map + ABC contracts.
- [Robot factory](getting-started/robot-factory.md) — full factory signature.
- [Quickstart](getting-started/quickstart.md) — concept walkthroughs.
