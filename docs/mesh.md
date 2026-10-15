---
description: Robots that find each other on their own. Call any of them from anywhere, and stop all of them at once.
---

# Mesh

{{drawing:d07_mesh}}

Put two laptops on the same Wi-Fi. That's the setup.

```python
from strands_robots import Robot
from strands_robots.mesh import Mesh

lab = Mesh("zenoh://", tenant="lab-a")
arm = Robot("so101", mesh=lab)       # it's on the mesh now
```

And from the other laptop:

```python
me = Mesh("zenoh://", tenant="lab-a")
me.discover("device(type:robot_arm)")            # [Device('so101', ...)]
me.invoke("so101", "act", action={"shoulder_pan": 0.3})
me.subscribe("so101", "state", print)            # its live feed
remote = me.connect("so101")                     # a Robot you can .run() like any other
me.broadcast("estop")                            # everything in the room stops
```

<img class="sr-media" src="../assets/media/mesh_two_processes.gif" alt="two sim robots in two processes, driven from a third">

The robot stays in charge of itself: every command from the mesh goes through its own safety chain, and a remote
caller can never approve a move on the operator's behalf. If whoever was driving disappears, the robot does what its
`on_disconnect` says. Resuming after an e-stop takes the operator who pressed it.

On one machine it's fast: robots show up in 9-53 ms, a command round trip is about 1 ms, and state streams at 500 Hz
without dropping a beat. Need it across the internet? `Mesh("iot://us-west-2")` runs the same thing over AWS IoT Core
(about 200 ms a command, and an e-stop to an offline robot comes straight back 404 instead of waiting in a queue).
`lab.tool` hands the whole room to an agent.
