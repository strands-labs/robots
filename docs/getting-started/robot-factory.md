---
description: Robot(name, mode, backend, urdf_path, cameras, position, data_config, mesh, peer_id, **kwargs) — the full signature with every kwarg explained.
---

# Robot factory

`Robot(...)` returns a `Simulation` or `HardwareRobot` based on `mode`.

```python
from strands_robots import Robot

robot = Robot("so100")               # Simulation (default, safe)
robot = Robot("so100", mode="real")  # HardwareRobot
robot = Robot("so100", mode="auto")  # probes USB, falls back to sim
```

## Parameters

| Param | Type | Default | What |
|-------|------|---------|------|
| `name` | str | required | Catalog name or alias. Resolved via `registry/robots.json`. |
| `mode` | str | `"sim"` | `"sim"` / `"real"` / `"auto"`. Overridden by `STRANDS_ROBOT_MODE`. |
| `backend` | str | `"mujoco"` | Sim backend. Ignored when `mode="real"`. |
| `urdf_path` | str | `None` | Explicit MJCF/URDF path — bypasses registry. |
| `cameras` | dict | `None` | Real-hardware camera config. **Rejected in `mode="sim"`** — raises `ValueError`. |
| `position` | list | `None` | Robot position `[x, y, z]` in sim world. |
| `data_config` | str | `None` | GR00T data_config name. |
| `mesh` | bool | `True` | Auto-join Zenoh mesh. |
| `peer_id` | str | `None` | Stable mesh peer id. Auto-generated if omitted. |
| `**kwargs` | | | Forwarded to backend constructor. Unknown kwargs raise `ValueError`. |

## Name resolution

```python
from strands_robots.registry import resolve_name

resolve_name("SO-100")    # 'so100'
resolve_name("franka")    # 'panda'
resolve_name("g1")        # 'unitree_g1'
```

Case-insensitive, hyphens/underscores interchangeable. Full alias map in `registry/robots.json`.

## Real hardware

```python
robot = Robot(
    "so100",
    mode="real",
    cameras={"wrist": {"type": "opencv", "index_or_path": "/dev/video0"}},
    port="/dev/tty.usbserial-A50285BI",
    control_frequency=50.0,
)
```

Forwardable kwargs: `port`, `robot_ip`, `kp`, `kd`, `default_positions`, `control_dt`,
`is_simulation`, `gravity_compensation`, `controller`, `calibration_dir`, `mock`,
`use_degrees`, `max_relative_target`, `disable_torque_on_disconnect`.

## Mesh

```python
sim = Robot("so100")
sim.mesh.peer_id   # 'so100_sim-a1b2c3d4'
sim.mesh.alive     # True

Robot("so100", mesh=False)   # per-robot off
# STRANDS_MESH=false         # process-wide off
```

Mesh failure is non-fatal; `.mesh = None` if Zenoh unavailable.

## See also

- [Robot catalog](../robots/index.md) — 68 catalog names.
- [Architecture](../architecture.md) — factory in the module map.
- [Multi-robot mesh](../mesh.md) — mesh peer discovery.
