### Fixed: the Isaac backend refuses to step a scene PhysX's tensor view no longer covers

PhysX builds its tensor simulation view at `world.reset()`, and adding or
deleting a physics-body prim afterwards invalidates it. Stepping that stale view
advanced the clock and simulated nothing, while every envelope reported success:

```python
sim.create_world()
sim.add_robot("arm", urdf_path="arm.urdf")
sim.add_object(name="cube", shape="cuboid", position=[0.4, 0.0, 0.6], size=[0.05] * 3)
sim.step(90)
# before: {"status": "success", ... "Stepped 90x ... 33 steps/sec"}
#         - the cube never left 0.600 m, and the arm's get_observation() went 2 keys -> 0
# after:  {"status": "error", ... "the scene changed since the last reset() ... Call reset() first"}
```

Measured on `nvcr.io/nvidia/isaac-sim:6.0.1` (A10G), the invalidation had three
consequences and every one of them was silent. The body never moved, though
`step` reported a rate for the steps it claimed to have taken: a cube spawned at
`z=0.600` was still at `z=0.600` afterwards, and fell to `z=0.025` once the same
`step` ran after a `reset()`. An already-working robot's `get_observation()` went
empty - 2 keys to 0 on a 2-joint URDF arm, and back to 2 after the reset. And
after a `remove_object`, the observation stopped degrading and started *raising*
- `SingleArticulation.get_joint_positions` throws a bare `Exception` ("Failed to
get DOF positions from backend") straight out of a method the `SimEngine` ABC
documents as returning a dict, which the narrow handler downstream cannot catch
without widening to `except Exception`.

The observation half was measured on a URDF robot on purpose. The three
procedural builders (`so100`, `panda`, `unitree_g1`) leave their
`_RobotState.articulation` as `None` and report 0 observation keys at every point
in the lifecycle, reset or no reset, so neither the drop nor the raise is
observable on one. That is a separate defect and is not addressed here.

`reset()` repairs all three, so the scene was always one call away from correct
and nothing said so. `add_object` and `remove_object` now mark the scene, `step`
refuses while the mark stands and names the remedy, and `get_observation` answers
empty - its documented degraded mode - with a WARNING rather than raising.

Which mutations invalidate the view was measured rather than assumed:
`add_camera`, `remove_camera`, `move_object`, `add_robot` and `remove_robot` each
left that arm reporting both its keys, so none of them marks the scene. A gate
that fired on those would refuse `step` over a view that is perfectly live.

`load_scene` is the one path that repairs the view *without* a reset - it rebuilds
through `SimulationManager.initialize_physics()` and `world.play()`, deliberately
not `world.reset()`, because a full reset re-applies every registered prim's
default state and was measured to explode an already-posed articulation into
non-finite PhysX bounds (#1802). It clears the mark where that rebuild lands, so
a scene load still steps. A reload that removes prior objects and realizes none
skips the rebuild and stays marked, which is the honest verdict: those removals
invalidated the view with nothing behind them.

The refusal is Isaac-only and changes no cross-backend contract. MuJoCo needs no
equivalent - its `step` reads the compiled model directly - and the shared
step-count and lock-hold domains are unchanged, since the gate is inert until a
body mutation marks the scene and sits behind the existing "No world created" and
"World not initialized" refusals.

**A failed `add_object` marks the scene stale too.** Only the success path set the
flag, and the failure path is stale by the same mechanism - which that handler's own
comment already states: `_construct_shape_prim` stops the timeline, clearing the
physics sim view, *before* constructing a dynamic prim. By the time the handler runs
the view is already invalid whether the construction went on to succeed or to raise.

So a failed `add_object` left `step()` willing to advance, and the clock moved over a
scene PhysX was no longer simulating while every robot's `get_observation` came back
empty - the exact degradation the flag was added to refuse. To a caller it read as a
transient add failure followed by a sim that had quietly stopped simulating: no
exception, a plausible error envelope, and then empty observations attributed to
whatever ran next.

Pinned across all six exception types the handler names, and by a drift guard that
derives from the source: any `return` below the `_construct_shape_prim` call which
does not have the flag set on the way is a new instance of this bug. There were two
such returns when only one was covered.


---

**Correction, from hardware.** The first version of this entry marked the scene stale
for *any* `add_object` / `remove_object`, and had `step` refuse. Both parts were wrong,
and the evidence for them was one measurement generalised a step too far - the original
"measured, not inferred" note was taken on a **DynamicCuboid** and applied to every body.

Measured on an A10G under Isaac Sim 6.0.1, reading a Franka's joint keys either side of
each operation:

| operation | tensor view |
|---|---|
| `add_object(is_static=True)` | **intact** - 9 joint keys -> 9 |
| `add_object(is_static=False)` | invalidated - 9 -> 0 |
| remove a **static** prim | **intact** - 9 -> 9 |
| remove a **dynamic** prim | invalidated, and the next joint read **hung** until a 2-minute timeout |

The mechanism was already gated this way in `add_object`: `_construct_shape_prim` stops
the timeline - which is what clears the sim view - only for a dynamic prim. The mark sat
outside a gate the code itself drew.

Marking unconditionally was a live regression, not a theoretical one. It disabled **every**
`step` in `examples/isaac_gs`: all three of its `add_object` calls are `is_static=True`,
it never calls `reset()` anywhere (its agent prompt instructs the model *"never rebuild,
reset, or destroy it"*), and all six of its step sites discard the returned envelope. So
the physics settle, the camera warmup - whose own docstring says an unwarmed product
returns malformed frames - and the wave demo all silently stopped happening, with no
diagnostic, in an example whose entire output is images. The same shape hit
`tests_integ/simulation/test_isaac_body_state_gpu` and `docs/simulation/isaac.md`'s own
usage example.

**Refusing is kept for the case that genuinely is stale**, and an in-place repair was
attempted and abandoned. `SimulationManager.initialize_physics()` + `world.play()`
restored the view in one state (0 -> 9 joint keys on a post-reset engine whose timeline
the dynamic add had stopped) but **not** in the state a real caller is in: after
`examples/isaac_gs`'s `build_default_scene` the timeline is already playing, so `play()`
is a no-op and the rebuild left joint keys at 0 while reporting success. Stopping the
timeline first and then rebuilding also left them at 0. Only `reset()` restored (0 -> 18).

A repair that silently claims success is worse than a refusal, so there is none: `step`
refuses, and the message now names `reset()` while stating that it returns robots to
their default pose, so the cost is visible rather than discovered.

That leaves two consumers still refused on their dynamic adds -
`examples/so101_curobo` and `docs/simulation/isaac.md`'s usage example - and both
discard the step envelope, so the refusal is silent there. Recorded as a known
limitation rather than claimed fixed.

Each half of the mark is pinned by a test that fails when that half alone is reverted -
verified as a mutation matrix (3 failures for an unconditional add mark, 2 for an
unconditional remove mark).
