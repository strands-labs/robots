### Fixed: Isaac `load_scene` no longer leaves the episode on frozen physics; LIBERO init states now apply on Isaac as object/robot poses

Two composing defects kept every Isaac LIBERO eval motionless-but-green
(#1820). First, `IsaacSimulation.load_scene` realized dynamic objects through
constructors that stop the timeline (#159's deferred-physics guard) and
nothing restarted it, so the whole episode ran on frozen physics:
`SimulationContext.step` only integrates when `is_playing()`, joint reads
went stale, `send_action` targeted a view that never integrates - and every
envelope still reported success. `load_scene` now restarts the timeline after
the physics-view rebuild via `world.play()` (a bare `timeline.play()` only
queues the state change on 6.0.x and never lands on the headless
`step(render=False)` path - measured live: `is_playing()` still `False`
after `play()` + 5 steps), reads `is_playing()` back, and returns an error
envelope rather than handing out a silently frozen episode.

Second - previously *masked* by the frozen timeline - the realized objects
sit at MJCF **placeholder** poses (LIBERO encodes real per-episode poses in
BDDL init states, which the MuJoCo backend applies as qpos vectors and Isaac
silently skipped): coincident dynamic bodies inside the robot base, so live
physics starts from deep interpenetration and storms PhysX "Illegal
BroadPhaseUpdateData - non-finite bounds" into NaN joint state.
`LiberoAdapter._apply_canonical_state` now routes model-less backends to a
new `_apply_object_pose_state` branch: the init state is decoded through a
local CPU MuJoCo compile of the scene MJCF (same row-selection and
width-validation semantics as the MuJoCo branch) and applied per named prim
via `move_object` - composing each free body's `xpos`/`xquat` with the
prim's body-frame collision-AABB offset, now recorded on
`SceneObject.offset`. The robot base is aligned with the scene's
`robot0_base` body through a new `IsaacSimulation.set_robot_pose` (on
MuJoCo the robot is part of the scene MJCF at `(-0.66, 0, 0.912)`; on Isaac
the USD articulation spawned at the origin, inside the scene's
origin-anchored static fixtures). Failed teleports raise rather than
warn-and-continue, and a settle step runs only *after* the poses are legal.

A GPU integration test (`tests_integ/simulation/test_isaac_scene_physics_gpu.py`)
pins the gate frozen physics cannot fake: after `load_scene` (twice - the
episode-2 reload included) the timeline is playing and a joint-target
`send_action` measurably moves the articulation, with all joints finite.
Verified end-to-end on Isaac Sim 6.0 / L4: `run_isaac.py --policy mock
--n-episodes 2` completes both episodes with zero PhysX broadphase errors
(previously a storm of them).
