# 🎓 Sample 09: Multi-Robot Fleet — Zenoh Mesh & Fleet Orchestration

**Level:** 3 (Advanced / High School) · **Time:** 25 min · **Hardware:** CPU (real hardware optional)

---

## What You'll Learn

1. How the Zenoh peer-to-peer mesh works (zero configuration)
2. Create multiple robots that auto-discover each other
3. Send commands between robots (`send`, `broadcast`, `tell`)
4. Watch live VLA execution streams from remote robots
5. Use ARM's `device_connect` as a drop-in for `robot_mesh`
6. Orchestrate a fleet: assign tasks, monitor status, emergency stop

## Prerequisites

- Samples 01–08 completed
- `pip install strands-robots[zenoh]` (installs `eclipse-zenoh`)
- Optional: Multiple physical robots on the same network
- Optional: ARM `device_connect`

---

## Architecture

Every `Robot()` is a mesh peer by default — no `join()`, no `connect()`, no broker.

```
┌──────────────────────────────────────────────────────────────────┐
│                     Zenoh Mesh (UDP Multicast)                   │
│                       224.0.0.224:7446                           │
│                                                                  │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│   │ Robot("so100")│   │ Robot("panda")│   │ Robot("g1")  │       │
│   │  peer: left   │◄─►│  peer: right  │◄─►│  peer: base  │       │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘       │
│          │                   │                   │               │
│          └───────────────────┴───────────────────┘               │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    │   Reachy Mini      │                        │
│                    │ (native Zenoh)     │                        │
│                    └───────────────────┘                        │
└──────────────────────────────────────────────────────────────────┘
```

### Zenoh Namespace Design

Every robot publishes and subscribes to structured topics:

| Topic | Frequency | Content |
|-------|-----------|---------|
| `strands/{peer_id}/presence` | 2 Hz | Identity, capabilities, task status |
| `strands/{peer_id}/state` | 10 Hz | Joint positions, sim time |
| `strands/{peer_id}/stream` | 50 Hz* | VLA execution: observation + action per step |
| `strands/{peer_id}/cmd` | On demand | Incoming commands |
| `strands/{peer_id}/response/*` | On demand | Responses to commands |
| `strands/broadcast` | On demand | Messages to ALL peers |

*\*Only during policy execution*

---

## Files in This Sample

| File | What It Does |
|------|-------------|
| `zenoh_mesh_demo.py` | Create 3 simulated robots, auto-discovery, send commands |
| `fleet_orchestration.py` | Coordinate a collaborative pick-and-place across robots |
| `live_stream.py` | Subscribe to VLA execution streams, plot joint trajectories |
| `device_connect_bridge.py` | ARM `device_connect` drop-in replacement demo |
| `emergency_stop.py` | Fleet-wide emergency stop, graceful shutdown |
| `configs/fleet_3_arms.yaml` | 3-arm fleet configuration |
| `configs/fleet_heterogeneous.yaml` | Mixed fleet: arm + humanoid + quadruped |

---

## `device_connect` vs `robot_mesh`

Both tools provide the same Zenoh-based coordination API:

| Feature | `robot_mesh` | ARM `device_connect` |
|---------|-------------|---------------------|
| Discovery | Zenoh multicast | Zenoh multicast |
| Protocol | `strands/**` topics | `device_connect/**` topics |
| Auto-detect | Always available | Used if installed |
| Reachy Mini | Via `subscribe()` | Native support |
| ToolSpec | Same | Same (drop-in) |

If `device_connect` is installed, `strands-robots` auto-routes to ARM's implementation.

---

## Real-World Fleet Topology

```
┌──────────────────────────────────────────────────────┐
│                    Lab Network (LAN)                  │
│                                                      │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  Thor GPU    │  │ G1 EDU+    │  │ Reachy Mini  │  │
│  │ (sm_110)     │  │ (29 DOF)   │  │ (Zenoh)      │  │
│  │ Newton+GR00T │  │ Real HW    │  │ Stewart Head │  │
│  └──────┬───────┘  └─────┬──────┘  └──────┬───────┘  │
│         └────────────────┼─────────────────┘          │
│                          │                            │
│  ┌─────────────┐  ┌──────┴──────┐  ┌──────────────┐  │
│  │ Open Duck   │  │ Orchestrator│  │ SO-100 ×3    │  │
│  │ Mini (walk) │  │ Agent (PC)  │  │ (bimanual)   │  │
│  └─────────────┘  └─────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## Exercises

1. **Multi-Policy Fleet** — Create 5 SO-100 arms. Assign a different policy provider
   to each (`mock`, `lerobot_local`, `groot`, etc.) and run them simultaneously.

2. **Follow the Leader** — Two SO-100 arms: `arm1` moves, `arm2` mirrors by
   subscribing to `arm1`'s state stream and replaying joint positions.

3. **Fleet Monitor Dashboard** — Write a script that subscribes to all `strands/*/presence`
   topics and prints a live table of peer status (name, type, task, last seen).

---

## What You Learned

- ✅ Every `Robot()` is automatically on the Zenoh mesh
- ✅ Peers discover each other via UDP multicast (zero config)
- ✅ `send()` / `broadcast()` / `tell()` for cross-robot commands
- ✅ `on_stream()` to watch live VLA execution from any peer
- ✅ `emergency_stop()` broadcasts stop to ALL robots instantly
- ✅ `subscribe()` bridges to Reachy Mini, sensors, anything Zenoh
- ✅ ARM `device_connect` is a drop-in replacement

## What's Next

→ [Sample 10: Autonomous Repository Case Study](../10_autonomous_repo/README.md)
