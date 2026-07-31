### Added: `IsaacSimulation.get_body_state` + an EEF state source for LIBERO-on-Isaac

`examples/libero/run_isaac.py --policy groot` failed every inference call with
`Server error: State key 'state.x' must be in observation`: the
`libero_panda` GR00T data-config requires `state.x/y/z/roll/pitch/yaw/gripper`,
and both of `LiberoAdapter`'s EEF pose sources were MuJoCo-shaped - the direct
site/body read found no compiled model on Isaac, and the `get_body_state`
fallback found no such method on `IsaacSimulation` (#1802).

`IsaacSimulation.get_body_state(body_name)` now implements the MuJoCo
envelope contract (`{"json": {position, quaternion (wxyz), rotation_matrix,
...}}`, namespace-aware `<robot>/<link>` resolution, structured errors for
unknown bodies, main-thread pump respected). This single surface also
unblocks the predicate DSL on Isaac (`body_above_z`, `body_on`,
`distance_less_than`, ... all read through `get_body_state`).

On the adapter side, `LiberoAdapter` gained `eef_pos_offset` /
`eef_quat_offset` (site-equivalent corrections for the body-only fallback,
applied in the body's local frame), an observation-dict gripper source
(`state_gripper_signs` restores RoboSuite's opposite-sign finger convention),
and a loud, actionable ERROR - replacing a DEBUG skip - when EEF injection is
enabled but produces no state keys. `load_libero_suite(adapter_kwargs=...)`
forwards backend-specific state-source config to every registered task, and
`run_isaac.py` plumbs the measured Isaac Franka calibration (`panda_hand`
+ `[0, 0, 0.097]` grip-site offset; the Isaac `panda_hand` frame was measured
to coincide with RoboSuite's `robot0_right_hand` within 0.03 degrees, so no
quaternion correction applies) via a new `--eef-body-name` flag.

Getting the injected state to be non-empty also required fixing
`IsaacSimulation.load_scene`'s physics-view lifecycle: realizing/removing
scene prims on an already-stepped stage left the PhysX tensor view stale, so
robot joint reads returned nothing (no `state.gripper`) and episode-2
`DynamicCuboid` constructions crashed with "Failed to get rigid body
velocities from backend". The fix invalidates the view after removals and
rebuilds it (`SimulationManager.initialize_physics()` + per-robot
articulation re-init) after adds - deliberately NOT `world.reset()`, which
re-applies registered default states and was measured to blow the Franka
articulation into a PhysX "Illegal BroadPhaseUpdateData - non-finite bounds"
explosion within seconds. `step()`'s camera warm-up now also refreshes
secondary render products so a post-scene-load `wrist_image` camera
accumulates frames.
