# Policy Providers Overview

!!! info "Coming Soon"
    This page is under active development.

Strands Robots uses a plugin-based policy registry — any VLA model can be integrated.

## Architecture

```mermaid
classDiagram
    class Policy {
        <<abstract>>
        +get_actions(observation, instruction)
        +provider_name
    }
    class Gr00tPolicy {
        +data_config
        +get_actions()
    }
    class MockPolicy {
        +get_actions()
    }
    class CustomPolicy {
        +get_actions()
    }
    Policy <|-- Gr00tPolicy
    Policy <|-- MockPolicy
    Policy <|-- CustomPolicy
```

## Supported Providers

| Provider | Description |
|----------|-------------|
| `groot` | NVIDIA GR00T N1.5/N1.6 |
| `lerobot_local` | LeRobot local policies (ACT, Pi0, SmolVLA) |
| `mock` | Random actions for testing |
