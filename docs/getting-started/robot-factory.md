---
description: Robot(name, mode, backend, urdf_path, cameras, position, data_config, mesh, peer_id, orientation, keyframe, **kwargs) - the full signature with every kwarg explained.
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
| `urdf_path` | str | `None` | Explicit MJCF/URDF path - bypasses registry. Ignored when `mode="real"` (reported at debug level). |
| `cameras` | dict | `None` | Real-hardware camera config. **Rejected in `mode="sim"`** - raises `ValueError`. |
| `position` | list | `None` | Robot position `[x, y, z]` in sim world. Ignored when `mode="real"` (reported at debug level). |
| `data_config` | str | `None` | GR00T data_config name. Honoured in both modes: `mode="sim"` defaults it to the canonical robot name, `mode="real"` forwards it to the hardware driver, which carries it into the `policy_config` a policy is built with. |
| `mesh` | bool \| None | `None` | Join the Zenoh fleet mesh. `None` consults `STRANDS_MESH`, which leaves it **off** unless set to `true`/`1`/`yes` - pass `mesh=True` to opt in per robot. |
| `peer_id` | str | `None` | Stable mesh peer id. Auto-generated if omitted. |
| `orientation` | list | `None` | Robot base orientation `[w, x, y, z]` in sim world. Ignored when `mode="real"` (reported at debug level). |
| `keyframe` | str \| int | `None` | Spawn in a model `<keyframe>` pose (name or index) instead of the zero configuration. Ignored when `mode="real"` (reported at debug level). |
| `**kwargs` | | | Forwarded to the backend or driver constructor as given. A name it does not recognize is ignored, not refused, so check the spelling against the forwardable list below. |

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

`control_frequency` (Hz) sets the control loop's per-action period,
`1 / control_frequency` - the only throttle between two servo commands. It must be a
positive finite number: `0`, a negative rate, `nan` or `inf` raises `ValueError` at
construction, before the serial port is opened, rather than leaving the loop free-running
against the arm. This is the same domain the simulation applies to `run_policy`'s
`control_frequency`, so a rollout rehearsed in sim is honored identically on hardware.

Forwardable kwargs: `port`, `robot_ip`, `kp`, `kd`, `default_positions`, `control_dt`,
`is_simulation`, `gravity_compensation`, `controller`, `calibration_dir`, `mock`,
`use_degrees`, `max_relative_target`, `disable_torque_on_disconnect`.

Forwardable values are passed to the driver as given, because their accepted domains are
robot-specific. `max_relative_target` is the exception: it caps how far each commanded goal
position may move from the joint's present position, so it must be a positive finite number
(or a mapping of motor name to one). `0`, a negative limit, `nan`, `inf`, a bool or a
non-numeric value raises `ValueError` when the config is built, before the serial port is
opened - a non-finite limit would otherwise disable the clamp with no signal, and a negative
one inverts it into a fixed-magnitude step that ignores the policy. An `int` limit is
normalized to `float` so it reaches the motors. Omit the parameter (or pass `None`) to leave
the clamp disabled.

## Mesh

Mesh is opt-in, so a bare `Robot(...)` never starts Zenoh, ACL or e-stop machinery:

```python
sim = Robot("so100")
sim.mesh                     # None - never joined

sim = Robot("so100", mesh=True)   # per-robot on
sim.mesh.peer_id             # 'so100_sim-a1b2c3d4'
sim.mesh.alive               # True

# STRANDS_MESH=true          # process-wide on, for a bare Robot(...)
```

`STRANDS_MESH=false` is a kill switch: it keeps mesh off even where a caller passed
`mesh=True`. The environment never forces mesh on for a robot constructed with
`mesh=False`.

Mesh failure is non-fatal; `.mesh = None` if Zenoh unavailable.

## See also

- [Robot catalog](../robots/index.md) - 68 catalog names.
- [Architecture](../architecture.md) - factory in the module map.
- [Multi-robot mesh](../mesh.md) - mesh peer discovery.
