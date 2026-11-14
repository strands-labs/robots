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

A real robot mounted with `watchdog_ms=None` only answers `observe` and `describe`. `watchdog_ms` is an argument of
`Mesh.mount` (500 ms by default; `Robot(..., mesh=lab)` mounts with the default), not of `Robot`:
`lab.mount(Robot("so101", port="/dev/ttyACM0"), watchdog_ms=250)`. Like a command from an agent or the dashboard, a
move over the mesh to a real robot also needs the operator's yes from `operator=` on that `Robot` (its `Move` says `source="mesh"`) (see [Safety](safety.md)).
A real robot on a mesh any machine may join is observe-only as well (see "Securing the mesh"): there
the peer id a request carries is only a claim, so any machine could move the robot, or keep its watchdog from
noticing that the real mover is gone.

The watchdog arms only on a move the robot carried out. A move it refused, for bad arguments or by its safety
chain, changes nothing, so a request that moved nothing can never make it apply `on_disconnect`. Every
`on_disconnect` applied (by the mesh watchdog, a local `Watchdog` or a run that faulted) leaves one
`strands_robots.audit` warning with `rpc="on_disconnect"`, the mover's `peer`, the `reason`, what it `applied`
(`hold`, `targets`, or nothing while the e-stop is latched) and the `outcome` (`ok`, or the error's type). If one
request gets two different answers (a second peer answering for the robot), `invoke` raises `RemoteError` instead
of picking one.

## Skills: a policy that runs on the device

`me.connect(id).run(policy)` runs the policy in the caller and sends one observe and one act rpc per control step,
so the policy is only as fast as the network. Give the device a catalogue instead, and a caller starts a skill by
name; the policy runs on the device at its own rate:

<!-- test: skip, needs a second process for the device -->
```python
lab.mount(arm, skills={"walk": "microduck?skill=velstand&vx=0.3", "wave": my_policy})
me.invoke("so101", "skill", name="walk", seconds=4.0)  # answers at once
me.invoke("so101", "skill_status")                     # {"name": "walk", "state": "running", ...}
```

| Fact | |
|---|---|
| What a caller sends | A skill name from the device's own list and `seconds` (at most 60), optionally `instruction` |
| Where it runs | On the device, as an agent's run: every step passes its safety chain |
| A real robot | Asks its own `Robot(operator=)` with the `RunPlan`, on the device; the caller's reply never waits on a person |
| `stop`, `estop` | End a running skill |
| The caller goes quiet | The skill carries on; the caller may poll `skill_status` slowly |
| The caller's mesh vanishes | The device ends the skill and applies `on_disconnect` |

`mesh.tool` lists each device's skills, and its `run` starts one of them by name. A real robot on another machine
that offers no skills is refused there, instead of being driven one round trip per control step.

Measured over AWS IoT Core with the G1 walk (`holosoma?vx=0.5`, 4 s): 9.2 steps per second when the caller drove it,
the full 200 steps in 2 s as a skill.

## Device rpcs: a device's own commands

A device can answer more than the built-in rpcs. Give `mount` a handler per name; the device advertises each one
(its argument schema from the signature, its first doc line, and whether it writes) in its manifest, and a caller
resolves the name from there, so a process that never imported the handler can call it:

<!-- test: skip, needs a second process for the device -->
```python
from strands_robots.mesh import device_rpc

lab.mount(arm, rpcs={"gripper": set_gripper, "battery": device_rpc(read_battery, write=False)})
me.invoke("so101", "battery")               # {"percent": 87}
me.invoke("so101", "gripper", width=0.5)
```

| Fact | |
|---|---|
| Default kind | A write that arms the watchdog, the same as `act` |
| A write | Refused on an observe-only device, refused while e-stopped, needs the caller's peer id, refused on an id two peers hold |
| `device_rpc(fn, write=False)` | A read: answered everywhere, never arms the watchdog |
| `device_rpc(fn, arms=False)` | A write the watchdog need not watch (a light, a beep) |
| Arguments | Keyword only, JSON values; the device refuses an unknown, missing or wrongly typed one before the handler runs |
| `by` | Who asked; reaches the handler only if it names a `by` argument |
| Names | A mesh id that is not a built-in rpc (`estop`, `stop`, `hold`, ...), `run` or `calibrate`: refused at mount |
| A name the device does not advertise | `invoke` raises `LookupError` listing the device's rpcs |

The handler is the device owner's code and runs on the device: the safety chain checks what goes through
`act`, so a handler that moves the body should call `robot.confirm_act` itself rather than drive the bus.

## Securing the mesh

Give every machine a certificate from one lab CA. A machine with one joins the mesh over mutual TLS only, and signs
everything it sends with the same key:

```bash
python -m strands_robots.mesh ca                                                 # once, on an admin machine
python -m strands_robots.mesh cert lab-pc --serves 'so101,arm-*' --out ./lab-pc  # for the robot's machine
python -m strands_robots.mesh cert alice --out ./alice                           # for an operator
```

Run every `cert` on the admin machine: it signs with the CA key (`--ca`, default `~/.strands/mesh/ca`), which stays
there. Copy each `--out` folder to `~/.strands/mesh/identity` on its machine. `cert` replaces any identity already
in its `--out` folder, so give each machine its own folder and list every robot it serves in one `--serves`.

`cert` writes `~/.strands/mesh/identity` unless told otherwise, and a plain `zenoh://` mesh on that machine uses it.
`zenoh://?identity=<dir>` or `STRANDS_MESH_IDENTITY=<dir>` names another one, and `none` opts a process out (a
simulation). On such a mesh:

- A machine without a certificate from the lab CA cannot open a link: it cannot read, answer, publish or query.
  Machines with certificates still find each other on the LAN with no endpoints.
- A peer's id is `<name>.<random>`, where `<name>` is the name in its certificate. A robot answers only signed
  requests whose peer id belongs to the signer, and refuses a request it has already answered or one more than 30 s
  off its clock. No machine can send heartbeats for another, so a robot's watchdog always notices its mover leave.
- A request is refused unless it names the tenant, device and rpc of the key it arrives on, so a captured request for one device cannot be replayed onto another.
- A reply, a robot's announcement and its state count only when they are signed by a certificate whose `--serves`
  covers that robot's id. Another machine can neither answer for a robot nor announce one, and its answers never
  make a request fail. A machine refuses to mount a robot id its own certificate does not serve.
- An e-stop is never refused for its signature.

A real robot moves over the mesh only on a mesh with identities, or on a network you vouch for with
`zenoh://?trusted=1` or `STRANDS_ZENOH_TRUSTED=1` (a VLAN or WireGuard network only the lab reaches; peer ids there
are claims). Every machine on the robot's mesh needs this version. A robot with an identity refuses unsigned
requests from older peers.

Where peer ids are claims, the robot's reply to the `act` that armed its watchdog carries a beat key, and only
requests that carry it back count as its mover's. A machine that names the mover's peer id cannot keep its watchdog
from noticing the mover went quiet. The key travels in the clear, so a machine that can read the lab's traffic could
copy it. Only identities prove who sent a request. A mover on an older version sends no key, so its robot holds
`watchdog_ms` after its last `act`.

A zenoh config file (`?config=<path>` or `STRANDS_ZENOH_CONFIG`) is the session's config instead of the identity.
The mesh adds only its lease, any `?listen=` or `?connect=`, peer mode if the file sets no mode, and its scouting
group if the file sets neither `scouting.multicast.enabled` nor its `address`. A file counts as an identity when it
turns on `transport.link.tls.enable_mtls`, allows only the `tls` (or `quic`) link protocol, and names PEM certificate,
key and CA files that this CA issued. A file without them leaves the mesh open. To write one, for a `zenohd` router
or another zenoh client:

```bash
python -m strands_robots.mesh config --mode router --listen tls/0.0.0.0:7447 -o zenohd.json5
```

What signatures don't cover: a machine with a lab certificate can still drop traffic it routes, send stops, and hold
a copy of another peer's liveliness token. The copy only takes away the watchdog's early alarm when the mover
disconnects. Its quiet timer still fires after `watchdog_ms`. To revoke a certificate, make a new CA and reissue.

LeRobot remote inference (`Policy("zenoh://host:port/<deployment>")`) uses LeRobot's own zenoh client. Give it a
config file written by `config --mode client`, as [Running a LeRobot policy](lerobot.md#on-a-policy-server) describes.

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
