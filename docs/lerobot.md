---
description: Run a LeRobot checkpoint in this process, or on a GPU machine with lerobot-policy-server over Zenoh.
---

# Running a LeRobot policy

You can run any LeRobot checkpoint on a robot. LeRobot runs the model. strands-robots matches joint names, cameras
and units. It stops with a clear error if they do not fit. Install LeRobot first, see
[Installation](install.md#lerobot).

<!-- test: skip, downloads a LeRobot checkpoint and needs LeRobot installed -->
```python
from strands_robots import Policy, Robot

arm = Robot("so101")
arm.sim.camera("front", size=(640, 480), attach=arm)
arm.run(Policy("lerobot/smolvla_base"), instruction="pick up the cube", seconds=10)
```

## On a policy server

Run the model on a GPU machine:

```bash
lerobot-policy-server --config_path=smolvla.yaml
```

Then point the robot at it:

```python
arm.run(Policy("zenoh://gpu-box:7447/smolvla"), instruction="pick up the cube", seconds=10)
```

`server_config(robot.description, "lerobot/smolvla_base")` writes the server config for your robot.
For a server on another network, name the link: `zenoh://tls/host:7447/smolvla`.

The session's frames and action chunks carry no identity of their own. Through a `tls`, `quic` or `ws` link they pass
a zenoh router, and any client of that router can read them and answer for the server; a plain link to an internet
host crosses the network with no server authentication at all, which is weaker still. A real robot therefore refuses
any link off its own LAN with a `PermissionError` until you vouch for it: `Policy("zenoh://tls/host:7447/smolvla",
trusted=True)` or `STRANDS_ZENOH_TRUSTED=1`. For a router link, vouching means client certificates and access control
on the router; never point a real robot at a shared `zenohd`. A loopback or LAN link (`zenoh://gpu-box:7447/...`)
reaches a server on a network you control and is taken as is; a simulated robot is never gated.

If the server stops answering, the robot holds its last target for 1 s. Then the run stops with a fault.

## Options

| Option | Does |
|---|---|
| `tasks=[...]` | Refuse instructions the policy was not trained on |
| `trusted=True` | Let a real robot take a server off its own LAN: a router link or a plain internet link (see above) |
| `start_pose={...}` | Move here before the first step, over `start_seconds` (default 3) |
| `norm={...}` | Set the units a checkpoint was trained in |

## Units

The library checks a checkpoint's units against its saved statistics. If they do not match, the error names the
joint and the fix, for example `Policy(repo, norm={"*": "degrees", "gripper": "range_0_100"})`.

## Supported families

ACT, SmolVLA, pi0, pi0.5, GR00T, XVLA, EVO1, LaWAM, VLA-JEPA, FineART, MolmoAct2 and DM05 all run. LeRobot refuses
other families with its own error.
