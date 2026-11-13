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

### From another machine: a router with LeRobot's roles

The session's frames and action chunks carry no identity of their own, so whoever shares the link can read the
cameras and answer for the server: every client of a zenoh router, or anyone who answers at the server's address.
A real robot therefore takes actions only over an **authenticated channel**, LeRobot's role-based router setup:
mutual TLS on every link, discovery off, server and robot both router clients, and a router that denies by default
and lets each certificate name do only its part. Only `policy-server` may answer and publish actions; only
`robot-client` may publish frames and read actions. Another certificate from the same CA, or a second robot's,
cannot answer for the server. One command writes the certificates and the three zenoh configs for a deployment:

```bash
python -m strands_robots.policy.lerobot zenoh-config --deployment smolvla \
  --out ~/.strands/lerobot/smolvla --host gpu-box
zenohd -c ~/.strands/lerobot/smolvla/router.json5
```

`zenohd`, the router, is not on PyPI: install a 1.x release (`brew install zenoh`, the
[eclipse-zenoh releases](https://github.com/eclipse-zenoh/zenoh/releases) or the `eclipse/zenoh` container image).

<!-- test: skip, needs a running zenohd with the generated certificates -->
```python
from strands_robots.policy.lerobot import server_config

d = "~/.strands/lerobot/smolvla"
cfg = server_config(arm.description, "lerobot/smolvla_base", deployment="smolvla", endpoint="tls/gpu-box:7447",
                    zenoh_config=f"{d}/server.json5")  # the policy-server role joins the router as a client
arm.run(Policy("zenoh://tls/gpu-box:7447/smolvla", zenoh_config=f"{d}/robot.json5"), instruction="pick up the cube")
```

`--host` is the name (or address, repeatable) the router is dialled by; it goes into the router's certificate.
The configs hold absolute paths to `certs/`: a robot needs `ca.pem` and `robot-client*.pem` at the same paths,
and `ca-key.pem` stays where you made the certificates. Re-running keeps existing certificates and rewrites the
configs, so `--deployment` can change. The ACL covers only `lerobot/inference/v1/deployments/<deployment>/**`.

`STRANDS_LEROBOT_ZENOH_CONFIG=robot.json5` sets the robot's file for every session. Anything else is refused on a real robot
with a `PermissionError`: a plain link to another machine (even on your LAN, where a shared `zenohd` or a host
taking the address would answer), a `tls`, `quic` or `ws` link without mutual TLS, a config whose certificate is not
named `robot-client`, and `zenoh_mode="peer"`. A server on this machine (`127.0.0.1`, `[::1]`, `localhost`) is
connected to at its literal loopback address. A simulated robot is never gated.

To keep a link you control without certificates, vouch for it: `Policy("zenoh://gpu-box:7447/smolvla",
trusted=True)` or `STRANDS_ZENOH_TRUSTED=1`.

Every holder of a role's certificate has that role: give each trust group its own CA (`--out` per group). The
router cannot see traffic inside one process, and cannot defend a compromised robot or server process.

If the server stops answering, the robot holds its last target for 1 s. Then the run stops with a fault.

A server off the LAN gets 30 s timeouts and JPEG frames instead of LeRobot's 5 s and raw frames. An address that
looks local (`127.0.0.1`, a LAN address) is measured first: a tunnel such as `ssh -L` to a far machine is `127.0.0.1`
too, so a round trip above 20 ms counts as off the LAN. `policy.network` and `policy.rtt_s` show what was decided.
This never changes which links a real robot may use.

If the server goes away (its presence on the link is lost), the robot holds its last target and the session is
reopened in the background, retrying with backoff for `reconnect_s` (default 10 s). The new session must be the same
checkpoint on the same channel, with the same features, semantics and rate; anything else ends the run with a fault
that names what changed. Once it is back, the policy starts from the current frame; no old plan is replayed.
`policy.reconnects` lists each recovered drop. `reconnect_s=0` ends the run with a fault at once.

## Options

| Option | Does |
|---|---|
| `tasks=[...]` | Refuse instructions the policy was not trained on |
| `zenoh_config="robot.json5"` | The session's zenoh config (the `robot-client` role from `zenoh-config`) |
| `trusted=True` | Let a real robot take actions over a link without the robot role (see above) |
| `network="wan"` | Use the internet's settings (30 s timeouts, JPEG frames); `"lan"` keeps LeRobot's; `"auto"` (default) measures a local-looking address |
| `reconnect_s=10` | Hold and reopen the session for this long when the server goes away (0: fault at once) |
| `start_pose={...}` | Move here before the first step, over `start_seconds` (default 3) |
| `norm={...}` | Set the units a checkpoint was trained in |

## Units

The library checks a checkpoint's units against its saved statistics. If they do not match, the error names the
joint and the fix, for example `Policy(repo, norm={"*": "degrees", "gripper": "range_0_100"})`.

## Supported families

ACT, SmolVLA, pi0, pi0.5, GR00T, XVLA, EVO1, LaWAM, VLA-JEPA, FineART, MolmoAct2 and DM05 all run. LeRobot refuses
other families with its own error.
