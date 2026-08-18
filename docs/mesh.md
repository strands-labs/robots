---
description: Two Robot() instances coordinating over the Zenoh mesh - peer discovery, RPC, emergency stop, teleop.
---

# Multi-robot mesh

<figure class="brand-figure" markdown="span">
  ![Robot peers discovering and coordinating over the Zenoh mesh](assets/mesh_network.svg){ .brand-svg }
</figure>

`Robot(name, mesh=True)` joins a Zenoh mesh - joining is opt-in, so a bare `Robot()` leaves `robot.mesh` as `None` unless `STRANDS_MESH` is `true`/`1`/`yes`. Peers then discover each other on the LAN and can query, command, and e-stop one another.

!!! info "Device Connect is the recommended networking layer"
    What's described here is the built-in **Zenoh mesh** — the automatic fallback. When the [`device-connect`](device-connect.md) extra is installed, `Robot().run()` and `robot_mesh()` use [**Device Connect**](device-connect.md) (structured RPC, presence, registry, safety) and fall back to this mesh only when it's unavailable. Both ride on Zenoh.

```python
# process A
from strands_robots import Robot
sim_a = Robot("so100", mesh=True)
print(sim_a.mesh.peers)          # discovers sim_b within ~1 s

# process B
sim_b = Robot("aloha", mesh=True)
sim_a.mesh.tell(sim_b.mesh.peer_id, "pick up the cube",
                policy_provider="mock", duration=10.0)
```

```bash
uv pip install "strands-robots[mesh]"   # eclipse-zenoh; already in the default install
```

`[mesh]` requires `eclipse-zenoh>=1.6.1`. The safety handlers authenticate an
e-stop / resume publisher at the wire level, below the JSON body, using
`zenoh.SourceInfo` on the publisher and `Sample.source_info` on the receiver.
Both names first ship in 1.6.1; on an older zenoh neither exists, so envelopes
travel unattributed and a receiver refuses one published by a peer that *is*
attributing. Upgrade every peer in a fleet together.

## Key mesh calls

```python
# Point-to-point status query
result = sim_a.mesh.send(target_peer_id, {"action": "status"}, timeout=5.0)

# Fan-out → list of responses collected within timeout
results = sim_a.mesh.broadcast({"action": "status"}, timeout=2.0)

# Safety primitive - writes a tamper-evident audit log
sim_a.mesh.emergency_stop()   # STRANDS_MESH_AUDIT_DIR overrides log location
```

## Recovering from an emergency stop

`emergency_stop()` latches a **lockout** on every peer that receives it. While a
peer is locked out it refuses every command except `status` and `resume`, and
nothing clears it on a timer - an e-stop that expired by itself would not be an
e-stop. Recovery is always an explicit `resume`:

```python
sim_a.mesh.send(peer_id, {"action": "resume", "override_code": OPERATOR_CODE})
```

Two prerequisites have to be in place *before* you e-stop a fleet, because both
are only observable once you are already locked out.

**1. Every peer needs the same override code.** `resume` is accepted only when
`STRANDS_MESH_OVERRIDE_CODE` is set, and receivers re-verify the operator's proof
against their own copy. With no code configured there is no remote resume at all
and each robot must be restarted with one set - so the mesh logs a WARNING at
startup when it is unset. Set it to the same value on every peer.

**2. Fleet clocks have to agree.** A resume envelope is stamped with the
operator's wall clock, and a receiver refuses one that is stale or future-dated:
older than `STRANDS_MESH_RESUME_FRESHNESS_S` (default 60s) or more than
`STRANDS_MESH_RESUME_FORWARD_SKEW_S` (default 5s) ahead. Each bound catches one
direction of skew - a receiver *ahead of* the operator trips the freshness
window, a receiver *behind* it trips the forward bound - so widening the other
one does not help. The forward bound is the tight one, which is the trap: a robot
whose clock is only **6 seconds behind** the operator sees a correct,
correctly-signed resume as future-dated and refuses it, logging

```
[safety] robot-1: refusing remote resume -- ``t``=... in future (forward_skew_s=5.0, now=...)
```

and every retry fails the same way, so the robot stays locked out until its clock
is corrected or the bound is widened. Keep fleet clocks in NTP sync - the same
"upgrade every peer together" discipline the zenoh floor needs above - or raise
both knobs on every peer.

**Correcting those clocks does not cost you the fleet.** The bounds above are the
only place a *stamp* crosses a machine boundary. Everything the mesh decides from
a **duration** on its own - how old a peer's last heartbeat is (`age`), whether
that peer has timed out (10s), which peer the registry evicts when it hits
`STRANDS_MESH_MAX_PEERS`, and the sensor publish intervals - is measured on
`time.monotonic()`, which no NTP correction, `date -s` or resume from suspend can
move. So bringing a robot's clock into sync to satisfy the forward bound above
will not make the next heartbeat tick drop every peer it can still hear.

Repeated wrong codes arm a brute-force cooldown
(`STRANDS_MESH_RESUME_MAX_FAILS`, `STRANDS_MESH_RESUME_BACKOFF_S`): during the
cooldown even the correct code is refused, so wait it out rather than retrying in
a loop. Every attempt, granted or refused, is written to the safety audit log.

## Published topics

| Topic | Rate | Content |
|-------|------|---------|
| `strands/{peer_id}/presence` | 2 Hz | heartbeat / peer discovery |
| `strands/{peer_id}/state` | 10 Hz | joints, sim time, task status |
| `strands/{peer_id}/cmd` | on demand | incoming RPC commands |
| `strands/{peer_id}/response/{id}` | on demand | RPC replies (turn_id correlated) |
| `strands/{peer_id}/stream` | on demand | VLA execution steps |
| `strands/{peer_id}/pose` | on demand | SE(3) from SLAM/odom/VIO |
| `strands/{peer_id}/imu` | on demand | orientation, gyro, accel |
| `strands/{peer_id}/health` | on demand | battery, CPU, memory |
| `strands/broadcast` | on demand | fan-out RPC |

Sensor topics only publish when the robot exposes the attribute. Zero cost when unused.

### Pose orientation

A robot that exposes a 4x4 SE(3) matrix as its pose provider has it decomposed
into `x` / `y` / `z`, a planar `theta`, and a `quat`. The quaternion is
scalar-first `[w, x, y, z]`, unit length, and sign-canonicalized to `w >= 0`
(`q` and `-q` are the same rotation, so an unchanged pose reads back
identically). `theta` and `quat` are decomposed from the same matrix, so they
always agree: both describe the full rotation for every orientation, including
the half of SO(3) past 120 degrees that a robot turning back the way it came
lands in.

## Agent-driven mesh

```python
from strands import Agent
from strands_robots.tools import robot_mesh

agent = Agent(tools=[sim_a, robot_mesh])
agent("Find every robot on the mesh and ask each one to report its status")
agent("E-STOP all peers")
```

## Mesh teleop

```python
# Machine A - leader publishes at 50 Hz  # requires hardware
leader = Robot("so100", mode="real", mesh=True)
leader.start_teleop_publish(teleoperator=leader.teleoperator,
                            device_name="leader", method="arm", hz=50)

# Machine B - follower applies incoming actions  # requires hardware
follower = Robot("so100", mode="real", mesh=True)
follower.start_teleop_receive(source_peer_id=leader.mesh.peer_id,
                              device_name="leader", apply_fn=None)

leader.stop_teleop("leader")
follower.stop_teleop("leader")
```

`get_teleop_status()` on either side inspects current teleop state.

Each published frame carries the operator's control signals from the
teleoperator's `get_teleop_events()` - `terminate_episode`, `success`,
`rerecord_episode`, `is_intervention` - alongside the joint action. Reading them
is best-effort: a teleoperator whose event surface stops answering (a keyboard
listener thread that died, a gamepad unplugged mid-session) never stops the joint
stream the follower is tracking. Because that field is also `null` for a leader
arm with no event surface at all, a failed read is reported on the publisher
rather than only on the wire - it increments `event_read_errors` in
`get_teleop_status()` and logs a warning naming the device and the cause, so an
operator whose signals are being dropped can see it.

`source_peer_id` and `device_name` are single segments of the mesh key
expression `strands/{peer_id}/input/{device_name}`, so both must be plain
identifiers (`[A-Za-z0-9_.-]+`, at most 128 chars). A Zenoh wildcard (`*`,
`**`) or an embedded `/` is refused with a `ValidationError` rather than
silently widening the stream: `source_peer_id="**"` would subscribe to
`strands/**/input/leader` and apply joint commands from every publishing peer,
not just the configured leader.

## Attach a mesh to a Simulation

`Robot(name, mode="sim", mesh=True)` is the normal path: it resolves the
`STRANDS_MESH` kill switch, starts a client, and stores it on the engine. To
attach one to a `Simulation` you built yourself, start the client and assign it:

```python
from strands_robots.mesh import init_mesh
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco")
sim.mesh = init_mesh(sim, peer_id="bench-sim")   # None when mesh is disabled
```

The `Simulation(mesh=...)` constructor argument takes that same started client -
it is not a boolean opt-in switch, and a truthy value with no `.stop()` (notably
`mesh=True`) is rejected at construction. `cleanup()` stops the client before it
tears down MuJoCo; a stop that fails is logged and stepped over, so the world,
renderers and executor are always released.

## Enable and disable

| Method | Scope |
|--------|-------|
| `Robot("so100", mesh=True)` | per-robot opt-in |
| `STRANDS_MESH=true` (or `1`/`yes`) | process-wide opt-in for a bare `Robot()` |
| `sim.mesh = init_mesh(sim, ...)` | a `Simulation` built directly (see above) |
| `STRANDS_MESH=false` | process-wide kill switch, overrides `mesh=True` |
| `Robot("so100", mesh=False)` | per-robot opt-out |

Unset `STRANDS_MESH` with no `mesh=` argument is the default, and it leaves the
mesh off.

Mesh failures are non-fatal - `robot.mesh` becomes `None`; the sim/hardware instance still works.

## See also

- [Device Connect](device-connect.md) - the recommended networking layer this mesh backs.
- [AI agents](agents.md) - drive the mesh with natural language.
- [Architecture](architecture.md) - where the mesh sits in the module map.
- [Mesh source](https://github.com/strands-labs/robots/tree/main/strands_robots/mesh) - `core.py`, `session.py`, `audit.py`, `sensors.py`, `input.py`.
