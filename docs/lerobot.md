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

Then point the robot at it. On the same machine that is all:

```python
arm.run(Policy("zenoh://127.0.0.1:7447/smolvla"), instruction="pick up the cube", seconds=10)
```

`server_config(robot.description, "lerobot/smolvla_base")` writes the server config for your robot.

### From another machine: mutual TLS

The session's frames and action chunks carry no identity of their own, so whoever shares the link can read the
cameras and answer for the server: every client of a zenoh router, or anyone who answers at the server's address.
A real robot therefore takes actions only over an **authenticated channel**: a `tls` (or `quic`) link whose zenoh
config makes both ends present a certificate from your CA. Give both ends a zenoh config file with your certificates:

```json5
{ transport: { link: { tls: {
  root_ca_certificate: "/etc/strands/ca.pem",
  enable_mtls: true,
  listen_certificate: "/etc/strands/cert.pem", listen_private_key: "/etc/strands/key.pem",    // server
  connect_certificate: "/etc/strands/cert.pem", connect_private_key: "/etc/strands/key.pem",  // robot
} } } }
```

<!-- test: skip, needs a running lerobot-policy-server with certificates -->
```python
cfg = server_config(arm.description, "lerobot/smolvla_base", endpoint="tls/0.0.0.0:7447", zenoh_config="gpu.json5")
arm.run(Policy("zenoh://tls/gpu-box:7447/smolvla", zenoh_config="robot.json5"), instruction="pick up the cube")
```

`STRANDS_ZENOH_CONFIG=robot.json5` sets the robot's file for every session. Anything else is refused on a real robot
with a `PermissionError`: a plain link to another machine (even on your LAN, where a shared `zenohd` or a host
taking the address would answer), and a `tls`, `quic` or `ws` link without mutual TLS. A server on this machine
(`127.0.0.1`, `[::1]`, `localhost`) is connected to at its literal loopback address. A simulated robot is never gated.

To keep a link you control without certificates, vouch for it: `Policy("zenoh://gpu-box:7447/smolvla",
trusted=True)` or `STRANDS_ZENOH_TRUSTED=1`.

Mutual TLS admits every holder of a certificate from your CA, and LeRobot's chunks are not signed: another holder
connected to the server (or to the router) can still read the frames and answer for it. Give each robot and server
its own certificate, and let the server's zenoh config (or the router's) admit only the robot's certificate name:

```json5
access_control: { enabled: true, default_permission: "deny",
  rules: [{ id: "lerobot", flows: ["ingress", "egress"], permission: "allow", key_exprs: ["lerobot/**"],
            messages: ["put", "delete", "declare_subscriber", "query", "declare_queryable", "reply",
                       "liveliness_token", "declare_liveliness_subscriber", "liveliness_query"] }],
  subjects: [{ id: "arm", cert_common_names: ["robot"] }],
  policies: [{ rules: ["lerobot"], subjects: ["arm"] }] }
```

If the server stops answering, the robot holds its last target for 1 s. Then the run stops with a fault.

## Options

| Option | Does |
|---|---|
| `tasks=[...]` | Refuse instructions the policy was not trained on |
| `zenoh_config="robot.json5"` | The zenoh config file the session opens with (certificates for mutual TLS) |
| `trusted=True` | Let a real robot take actions over a link without mutual TLS (see above) |
| `start_pose={...}` | Move here before the first step, over `start_seconds` (default 3) |
| `norm={...}` | Set the units a checkpoint was trained in |

## Units

The library checks a checkpoint's units against its saved statistics. If they do not match, the error names the
joint and the fix, for example `Policy(repo, norm={"*": "degrees", "gripper": "range_0_100"})`.

## Supported families

ACT, SmolVLA, pi0, pi0.5, GR00T, XVLA, EVO1, LaWAM, VLA-JEPA, FineART, MolmoAct2 and DM05 all run. LeRobot refuses
other families with its own error.
