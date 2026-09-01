---
description: One diagram, one source of truth. Module boundaries, ABC contracts, and the rule every layer obeys.
---

# Architecture

<figure class="brand-figure" markdown="span">
  ![Four-layer stack - Agent, Policies, Backends, Robots - with green action signals flowing down and cyan observation signals flowing back up](assets/architecture_flow.svg){ .brand-svg }
</figure>

```mermaid
graph TB
    subgraph user[Your code]
        AGENT["Strands Agent"]
        FACTORY["Robot('so100')"]
    end

    subgraph factory_layer[Robot factory  -  strands_robots/robot.py]
        ROBOT["Robot()"]
        REGISTRY["registry/robots.json<br/>73 robots, 8 categories"]
        ROBOT --> REGISTRY
    end

    subgraph backends[Backends]
        SIM["Simulation<br/>simulation/mujoco/simulation.py"]
        HW["HardwareRobot<br/>hardware_robot.py"]
        SIM_ABC["SimEngine ABC<br/>simulation/base.py"]
        SIM -.implements.-> SIM_ABC
    end

    subgraph policies[Policy layer  -  strands_robots/policies]
        POLICY_ABC["Policy ABC<br/>policies/base.py"]
        MOCK["MockPolicy"]
        GROOT["Gr00tPolicy"]
        LEROBOT["LerobotLocalPolicy"]
        COSMOS3["Cosmos3Policy"]
        FACTORY_FN["create_policy()"]
        MOCK -.implements.-> POLICY_ABC
        GROOT -.implements.-> POLICY_ABC
        LEROBOT -.implements.-> POLICY_ABC
        COSMOS3 -.implements.-> POLICY_ABC
        FACTORY_FN --> POLICY_ABC
    end

    subgraph extras[Cross-cutting]
        TOOLS["Tools<br/>tools/*.py"]
        RECORDER["DatasetRecorder<br/>dataset_recorder.py"]
    end

    AGENT --> FACTORY
    FACTORY --> ROBOT
    ROBOT -->|mode='sim' default| SIM
    ROBOT -->|mode='real'| HW

    SIM --> POLICY_ABC
    HW --> POLICY_ABC
    SIM --> RECORDER
    HW --> RECORDER

    AGENT --> TOOLS

    classDef user fill:#2ea44f,stroke:#1b7735,color:#fff
    classDef factory fill:#0969da,stroke:#044289,color:#fff
    classDef backend fill:#bf8700,stroke:#875e00,color:#fff
    classDef policy fill:#8250df,stroke:#5a32a3,color:#fff
    classDef cross fill:#cf222e,stroke:#86181d,color:#fff

    class AGENT,FACTORY user
    class ROBOT,REGISTRY factory
    class SIM,HW,SIM_ABC backend
    class POLICY_ABC,MOCK,GROOT,LEROBOT,COSMOS3,FACTORY_FN policy
    class TOOLS,RECORDER cross
```

## Modules

| Module | What it owns | Key types |
|--------|--------------|-----------|
| `strands_robots/robot.py` | Factory `Robot(name, mode, backend, **kwargs)`. Name resolution, sim/real dispatch, mesh attach. | `Robot()` function |
| `strands_robots/registry/` | 73 robots, 121 aliases, 8 categories. `robots.json` is source of truth. | `list_robots()`, `resolve_name()`, `get_robot()` |
| `strands_robots/simulation/` | MuJoCo `AgentTool` - 77 actions. | `Simulation`, `SimWorld`, `SimRobot`, `SimObject`, `SimCamera` |
| `strands_robots/simulation/base.py` | Backend ABC for future Isaac/Newton backends. | `SimEngine` |
| `strands_robots/hardware_robot.py` | Real-servo path. Async task execution + status. | `Robot` (class), `TaskStatus`, `RobotTaskState` |
| `strands_robots/policies/` | ABC + 15 providers + factory + JSON registry. | `Policy`, `create_policy()` |
| `strands_robots/dataset_recorder.py` | LeRobot v3 writer. | `DatasetRecorder` |
| `strands_robots/tools/` | 26 `@tool`-decorated helpers. | `lerobot_calibrate`, `serial_tool`, etc. |

## ABCs

**`Policy`** - three abstract members every implementation supplies: `get_actions(observation_dict, instruction) -> list[dict]` (async), `set_robot_state_keys(keys)`, and the `provider_name` property. The rest of the contract is *declared* rather than implemented - the runtime reads the declaration and supplies what it names: `requires_images` (default `True`), `required_bodies` (default `()`; the only way to receive a body pose beyond the floating base - a tracker that skips it gets `base_quat`, the pelvis) and `children` (default `()`; what a wrapper returns so a capability probe reaches the policy inside it). `reset(seed)` (default no-op) is called per episode. 17 implementations ship; [Policies](policies/overview.md) is the provider table.

**`SimEngine`** - `create_world()`, `step()`, 30+ abstract actions. Today: MuJoCo CPU. Roadmap: Isaac Sim, Newton.

**Strands `AgentTool`** - `Simulation` and `HardwareRobot` are both `AgentTool` subclasses. `Agent(tools=[robot])` calls actions through the tool dispatcher.

## The one rule

**Lazy imports everywhere.** `strands_robots/__init__.py` exports `Policy`, `MockPolicy`, `create_policy` eagerly. Everything else (`Robot`, `Simulation`, `Gr00tPolicy`, the tools) is behind `__getattr__`. Enforced by `tests/test_init.py`.

## Extras

| Extra | Pulls in | When |
|-------|----------|------|
| `[sim-mujoco]` | `mujoco`, `numpy`, `imageio`, `imageio-ffmpeg`, `mink`, `qpsolvers[daqp]` | `Robot(mode="sim")`; `mink`/`qpsolvers` solve IK for the `move_to` primitive (the `[daqp]` backend extra is what makes the solve runnable - `qpsolvers` alone ships no solver) |
| `[lerobot]` | `lerobot>=0.6.1,<0.7.0`, `torch` | Real hardware OR `LerobotLocalPolicy` |
| `[groot-service]` | `pyzmq`, `msgpack` | `Gr00tPolicy` ZMQ |
| `[cosmos3-service]` | `msgpack`, `websockets` | `Cosmos3Policy` WebSocket |
| `[mesh]` | `eclipse-zenoh`, `json5` | Multi-robot mesh |
| `[mesh-iot]` | above + `awsiotsdk`, `awscrt`, `boto3` | AWS IoT Core transport |
| `[all]` | 20 of the 32 extras - not a union; see [installation](getting-started/installation.md) for the 11 it leaves opt-in | CI / exploration |

## See also

- [Robot factory](getting-started/robot-factory.md) - every `Robot(...)` kwarg.
- [Custom policies](policies/custom-policies.md) - implement and register.
- [Simulation overview](simulation/overview.md) - the 60+ action vocabulary.
- [Contributing](contributing.md) - module conventions.
