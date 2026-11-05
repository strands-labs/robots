---
description: "The robot mesh over Zenoh: discovery, remote commands, state streams, a shared e-stop and signed resume."
---

# Mesh networking

A mesh connects robots across processes and machines. It runs on Zenoh. On one LAN it needs no setup.

{{drawing:d07_mesh}}

```python
from strands_robots import Robot
from strands_robots.mesh import Mesh

lab = Mesh("zenoh://", tenant="lab-a")
arm = Robot("so101", mesh=lab)
```

From another process or machine:

```python
me = Mesh("zenoh://", tenant="lab-a")
me.discover("device(type:robot_arm)")
me.invoke("so101", "act", action={"shoulder_pan": 0.3}, seconds=1.0)
me.subscribe("so101", "state", print)
remote = me.connect("so101")  # a Robot you can run policies on
me.broadcast("estop")         # every robot in the tenant stops
```

<img class="sr-media" src="../assets/media/mesh_two_processes.gif" alt="two robots in two processes, driven from a third">

## Safety

Each robot runs its own safety chain on mesh commands. A remote caller cannot approve a move for the operator. If
the caller disappears, the robot does what `on_disconnect` says.

A real robot mounted with `watchdog_ms=None` only answers `observe` and `describe`. Set `watchdog_ms=` to move it
over the mesh.

The watchdog arms only on a move the robot carried out. A move it refused, for bad arguments or by its safety
chain, changes nothing, so a request that moved nothing can never make it apply `on_disconnect`. Every
`on_disconnect` the watchdog applies leaves a `strands_robots.audit` warning. If one request gets two different
answers (a second peer answering for the robot), `invoke` raises `RemoteError` instead of picking one.

## Securing the mesh

A plain `zenoh://` mesh lets any machine that reaches it join. That machine can read state and cameras, answer
for a robot, and send it commands that its safety chain then checks. Mounting a real robot on such a mesh logs an
audit warning. To say who may join, give zenoh a config file with `?config=<path>` or `STRANDS_ZENOH_CONFIG`. That
file can set TLS with client certificates on `tls/` endpoints, user and password authentication, and access
control. The file is the session's config. The mesh adds only its lease, any `?listen=` or `?connect=`, and its
scouting group and peer mode if the file sets neither:

```json5
{
  mode: "peer",
  listen: { endpoints: ["tls/0.0.0.0:7447"] },
  transport: { link: { tls: {
    listen_certificate: "robot.pem", listen_private_key: "robot.key",
    root_ca_certificate: "lab-ca.pem", enable_mtls: true,
  } } },
}
```

LeRobot remote inference (`Policy("zenoh://host:port/<deployment>")`) uses LeRobot's own zenoh client. Run it over a
`tls/` endpoint or a network you control, such as a VLAN or WireGuard.

## Resume after a mesh e-stop

By default, a resume over Zenoh is refused, because any peer can claim a name. Give each operator a key:

```python
from strands_robots.mesh import Mesh, authority

private, public = authority.keygen()  # the robot gets `public`
lab = Mesh("zenoh://", tenant="lab-a", resume_authority=authority.Authority(public_keys=(public,)))
me = Mesh("zenoh://", tenant="lab-a", resume_key=private)
me.invoke("so101", "resume", approved_by=me.operator)
```

On a network you trust, `authority.Authority(trust_names=True)` accepts names.

## Across the internet

`Mesh("iot://us-west-2")` runs the same mesh over [AWS IoT Core](iot.md). `lab.tool` gives an agent every robot
on the mesh, see [Agent tools](tools.md).
