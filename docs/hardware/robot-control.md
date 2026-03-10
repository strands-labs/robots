# Robot Control

!!! info "Coming Soon"
    This page is under active development.

Control real robot hardware through the Robot tool.

## Control Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent as Strands Agent
    participant Robot as Robot Tool
    participant Policy as Policy
    participant HW as Hardware

    User->>Agent: "Pick up the red block"
    Agent->>Robot: execute(instruction)
    loop Control Loop @ 50Hz
        Robot->>HW: get_observation()
        Robot->>Policy: get_actions(obs)
        Robot->>HW: send_action(action)
    end
    Robot-->>Agent: Task completed
```

## Actions

| Action | Description |
|--------|-------------|
| `execute` | Blocking execution |
| `start` | Non-blocking async start |
| `status` | Get current task status |
| `stop` | Interrupt running task |
