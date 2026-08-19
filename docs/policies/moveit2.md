---
description: MoveIt2 motion planning via a ROS 2 sidecar - ZMQ + msgpack client, goal via target_pose / target_joints kwargs, no ROS 2 deps in the Python venv.
---

# MoveIt2

[`MoveIt2Policy`](https://github.com/strands-labs/robots/blob/main/strands_robots/policies/moveit2/policy.py)
is a thin ZMQ + msgpack client for a sidecar ROS 2 node running
[`moveit_py`](https://github.com/moveit/moveit2). Like
[cuRobo](curobo.md) it is a **non-VLA, collision-aware motion planner**: it
reads its goal from `**kwargs` (`target_pose` / `target_joints`), ignores
camera frames (`requires_images = False`), and never parses the instruction
string for control.

Unlike cuRobo's in-process CUDA library, MoveIt2 runs **out-of-process**: the
ROS 2 stack and `moveit_py` live entirely in a sidecar, so the Python venv
running `strands_robots` stays free of ROS 2 deps. The only client-side
requirements are `pyzmq` + `msgpack` (the `[moveit2]` extra).

## Install

```bash
pip install 'strands-robots[moveit2]'   # client side: pyzmq + msgpack only
```

The ROS 2 / `moveit_py` deps stay out of `pyproject.toml` - they live in the
sidecar. Bring the sidecar up via the docker-compose recipe or natively:

```bash
# Option 1 - docker compose (pinned, reproducible)
cd strands_robots/policies/moveit2/server
docker compose up

# Option 2 - native ROS 2 (dev loop)
source /opt/ros/jazzy/setup.bash         # or your distro
pip install pyzmq msgpack                # the only non-ROS deps
python -m strands_robots.policies.moveit2.server.zmq_node \
    --port 5556 --planning-group arm
```

See [`policies/moveit2/server/README.md`](https://github.com/strands-labs/robots/blob/main/strands_robots/policies/moveit2/server/README.md)
for the sidecar deployment and forking guidance.

## Quickstart

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "moveit2",                     # alias: "moveit"
    host="127.0.0.1",
    port=5556,
    planning_group="arm",
)

actions = policy.get_actions_sync(
    observation_dict={"observation.state": [0.0] * 6},
    instruction="reach for the red block",   # ignored by planners
    target_pose=[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],   # [x,y,z, qw,qx,qy,qz]
)
```

The smart-string resolver also works:
`create_policy("zmq://127.0.0.1:5556", planning_group="arm")`.

## Parameters

```python
MoveIt2Policy(
    host="127.0.0.1",      # sidecar hostname (loopback only by default)
    port=5556,             # sidecar port
    planning_group="arm",  # default MoveIt2 planning group; per-call override allowed
    timeout_ms=15000,      # ZMQ send + recv timeout in milliseconds
    api_token=None,        # included in every request; falls back to MOVEIT2_API_TOKEN
)
```

`timeout_ms` is applied as `RCVTIMEO` and `SNDTIMEO` on the REQ socket, so only
a positive whole number of milliseconds up to `2**31 - 1` names a budget. An
integral `float` or NumPy integer is accepted and stored as an `int`. `0` is
ZMQ's "return immediately" spelling and `-1` its "block forever" one; both are
refused at construction, because each makes `ping()` unable to report a
reachable sidecar as reachable.

`api_token` falls back to the `MOVEIT2_API_TOKEN` environment variable when not
passed. The client emits a plaintext-over-TCP warning when `host` is a non-
loopback address (the token travels unencrypted; terminate TLS at a proxy or
keep the sidecar on loopback).

## Goal kwargs

`MoveIt2Policy` shares the non-VLA goal vocabulary with the rest of the planner
family (see [cuRobo](curobo.md)), so a goal can flow across providers without
coupling to a backend:

| Key | Type | Meaning |
|-----|------|---------|
| `target_pose` | `list[float]` | Cartesian goal `[x, y, z, qw, qx, qy, qz]` in the base frame |
| `target_joints` | `dict[str, float]` | Joint-space goal keyed by joint name (rad / m) |
| `world_update` | `dict \| None` | Per-call collision-scene refresh, forwarded to the sidecar |
| `planning_group` | `str` | Override the policy's default planning group for this call |

Provide at least one of `target_pose` / `target_joints`. If **both** are set,
`target_joints` wins (mirrors MoveIt2's `setJointValueTarget`). When neither is
given, `get_actions(...)` raises `ValueError` rather than returning a
zero-action.

### Input validation

Goals are validated up-front before they reach the wire (defence-in-depth for
LLM-agent inputs):

- `target_pose` must be exactly 7 finite floats (NaN / inf rejected).
- `target_joints` keys must match `^[A-Za-z][A-Za-z0-9_-]*$` - the same
  allowlist `mesh.security.validate_command` applies, so a value the mesh
  accepts flows end-to-end without a second mismatch. Values must be finite.
- `planning_group` must be a plain identifier (no shell metacharacters or path
  traversal).

## Wire protocol

```
request  = {"endpoint": "plan",
            "data": {"joint_state": list[float] | None,
                     "planning_group": str,
                     "target_pose": [x, y, z, qw, qx, qy, qz] | None,
                     "target_joints": dict[str, float] | None,
                     "world_update": dict | None}}
response = {"trajectory": list[list[float]],
            "success": bool,
            "status": str}
```

Trajectory rows are `[time_from_start_seconds, q0, q1, ..., qN]`. The client
drops the leading time column when packing per-step action dicts and maps the
remaining columns onto the keys from `set_robot_state_keys(...)`. A
`success=False` response raises `RuntimeError`. The sidecar also exposes
`ping` (health check) and `reset` (per-episode hook).

### Failure reporting

REQ/REP is lockstep, so the sidecar answers every request it receives: a
request it cannot serve comes back as a failure response, never by dropping the
reply and exiting. A planning failure is reported in the `plan` response as
`success=False` plus a `status` naming the stage that failed:

| `status` prefix | Meaning |
| --- | --- |
| `unknown_planning_group:` | The requested group does not resolve. |
| `start_state_error:` | The current robot state is not readable (no `/joint_states` yet, state monitor not warmed up). |
| `missing_goal:` | Neither `target_pose` nor `target_joints` was supplied. |
| `invalid_goal:` | The goal was rejected - a joint the group does not have, an unresolvable pose link, or a `target_pose` that is not 7 values. |
| `planner_exception:` / `planner_returned_empty` | Planning ran and failed. |
| `trajectory_error:` | The planned trajectory did not serialise. |

Client-side validation checks `target_joints` key *syntax*, not whether the
group actually has those joints, so a joint-name typo is reported by the
sidecar as `invalid_goal:` rather than rejected before the call. Anything
outside the `plan` contract - an unknown endpoint, a request that does not
decode or that decodes to something other than a map, an unexpected error in a
forked handler - comes back as `{"error": ...}`, which the client raises as
`RuntimeError`. Both malformed-payload cases share the `malformed_request:`
prefix, so a client has one class to match whether the bytes were undecodable
or decoded to an integer.

## In simulation

```python
from strands_robots import Robot

sim = Robot("panda")              # sim-by-default
sim.run_policy(
    robot_name="panda",
    instruction="",               # ignored by the planner
    policy_provider="moveit2",
    policy_config={"host": "127.0.0.1", "port": 5556, "planning_group": "arm"},
    policy_kwargs={"target_pose": [0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]},
    duration=10.0,
    control_frequency=50.0,
)
```

`policy_config` and `policy_kwargs` are two different sinks. `policy_config`
is expanded into the policy **constructor**; the per-call goal belongs in
`policy_kwargs`, which the runner hands to every `get_actions()` call. Passing
`target_pose=` directly to `run_policy` raises `TypeError` - it has no such
parameter and no `**kwargs`.

The mesh path forwards the same goal vocabulary:
`mesh.tell(peer, "...", policy_provider="moveit2", target_pose=[...])`.
`Robot.start_task` forwards the same keywords through `**policy_kwargs` to
`create_policy`, so the hardware path accepts the goal vocabulary the mesh
dispatch collects from the wire command.

## See also

- [Policy overview](overview.md)
- [cuRobo](curobo.md) - in-process collision-aware planning (non-VLA, GPU).
- [GR00T](groot.md) - ZMQ service VLA.
- [Cosmos 3](cosmos3.md) - WebSocket VLA.
- [Custom policies](custom-policies.md) - implement the non-VLA goal-kwargs contract.
- [MoveIt2 project](https://moveit.ai/)
