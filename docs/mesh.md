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
