---
description: "The robot mesh over Zenoh: discovery, remote commands, state streams, a broadcast e-stop and signed resume."
---

# Mesh networking

{{drawing:d07_mesh}}

`Mesh("zenoh://")` joins a Zenoh peer-to-peer network. On one LAN no configuration is needed: peers discover
each other by multicast.

```python
from strands_robots import Robot
from strands_robots.mesh import Mesh

lab = Mesh("zenoh://", tenant="lab-a")
arm = Robot("so101", mesh=lab)       # it's on the mesh now
```

From another process or machine on the same network:

```python
me = Mesh("zenoh://", tenant="lab-a")
me.discover("device(type:robot_arm)")            # [Device('so101', ...)]
me.invoke("so101", "act", action={"shoulder_pan": 0.3}, seconds=1.0)  # reach it, not one step
me.subscribe("so101", "state", print)            # its live feed
remote = me.connect("so101")                     # a Robot you can .run() like any other
me.broadcast("estop")                            # every robot in the tenant stops
```

<img class="sr-media" src="../assets/media/mesh_two_processes.gif" alt="two sim robots in two processes, driven from a third">

## Safety over the mesh

The robot stays in charge of itself: every command from the mesh goes through its own safety chain, and a remote
caller can never approve a move on the operator's behalf. If whoever was driving disappears, the robot does what its
`on_disconnect` says.

A robot stopped over the mesh stays stopped until someone it trusts resumes it. Over `zenoh://` a claimed name is
not a credential (any peer that reaches the listener, even a loopback one, can type it), so by default a resume is
refused with `RequestRefused`. Give each operator a key, and the robot accepts only a resume they signed for this
stop:

```python
from strands_robots.mesh import Mesh, authority

private, public = authority.keygen()        # once, on the operator's machine; the robot gets `public`
lab = Mesh("zenoh://", tenant="lab-a", resume_authority=authority.Authority(public_keys=(public,)))  # robot side
me = Mesh("zenoh://", tenant="lab-a", resume_key=private)                                          # operator side
me.invoke("so101", "resume", approved_by=me.operator)                                             # signed, accepted
```

On a network you trust, `authority.Authority(trust_names=True)` accepts the operator who pressed the stop by name
instead. Names are trusted without asking only when every transport is in-process (`mem://`).

## Remote act

A remote `act` takes the same arguments as `Robot.act`: on its own it moves one control step toward the targets,
`steps=` advances the world that many, and `seconds=` (up to 10) reaches the pose before it answers. The watchdog
counts a reach in progress as its caller waiting, not as silence; the agent tool `use_mesh` offers `seconds` too.

Mount a real robot with `watchdog_ms=None` and it is served observe-only: `observe` and `describe` still
answer, but `act` and `hold` are refused by name, because with no watchdog a mover that vanished would leave
the arm commanded with nothing to recall it. Mount it with a `watchdog_ms=` to move it over the mesh. A
simulated body has no bus to run away and is unaffected.

## Performance and AWS IoT

On one machine, robots are discovered in 9-53 ms, a command round trip is about 1 ms, and state streams at 500 Hz
without dropping a beat. Across the internet, `Mesh("iot://us-west-2")` runs the same mesh over AWS IoT Core ([AWS IoT Core](iot.md))
(about 200 ms a command, and an e-stop to an offline robot comes straight back 404 instead of waiting in a queue).
`lab.tool` gives an agent every robot on the mesh; see [Agent tools](tools.md).
