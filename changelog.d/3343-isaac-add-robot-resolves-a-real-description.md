### Fixed: the Isaac backend's `add_robot` loads a real robot instead of reporting one

`add_robot(name)` with no asset path took a "procedural" branch whose comment read
"Build procedurally via USD API" and which made no USD call at all. It read joint
names off a hardcoded dataclass, registered a prim path for a prim it never
created, and returned success:

```python
sim.create_world()
sim.add_robot("so100")
# before: {"status": "success", ... "Robot 'so100' added (procedural: so100,
#          6 joints: ['shoulder_pan', 'shoulder_lift', 'elbow_flex', ...])"}
#         - 0 prims under /World/Robots/so100, articulation None before AND after
#           reset(), get_observation() == {} for the whole lifecycle
# after:  {"status": "success", ... "Robot 'so100' added (MJCF: .../trs_so_arm100/
#          scene.xml -> USD: ..., 6 joints)"}
#         - a live articulation, and get_observation() returns 6 keys
```

Measured on `nvcr.io/nvidia/isaac-sim:6.0.1` (A10G), `_RobotState.articulation`
was `None` both before and after `world.reset()`, no prim existed at the
registered path, and `get_observation()` was empty at every point in the
lifecycle. That was the documented headline example
(`sim.add_robot("so100")  # procedural; no asset files needed`).

**It was also wrong as metadata**, which is the half a caller could have checked
without a GPU. For the same robot name the MuJoCo backend reports a different
vocabulary, and for `panda` a different count:

| robot | the deleted table | MuJoCo, real asset |
|---|---|---|
| `so100` | 6: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` | 6: `Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw` |
| `panda` | **7**: `panda_joint1..panda_joint7` | **9**: `joint1..joint7, finger_joint1, finger_joint2` |

So the parity these docs promise - "the joint-name and observation contract
matches the MuJoCo backend, [so] policies and observation mappings transfer
unchanged between backends" - was false before any physics was involved.

Both halves are answered by loading the description MuJoCo loads. `add_robot` now
resolves the robot name (or `data_config=`) through
`strands_robots.simulation.model_registry.resolve_model`, the same resolver the
MuJoCo backend's `add_robot` uses, so one name means one file on both backends and
the vocabularies cannot drift. Measured on the same runtime, this reproduces
MuJoCo's names exactly: 9 of 9 for `panda`, 6 of 6 for `so100`, with a wired
articulation and a non-empty observation.
