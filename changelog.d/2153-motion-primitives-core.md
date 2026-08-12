### Changed: backend-agnostic core extracted from the MuJoCo motion primitives

The MuJoCo-only `MotionPrimitivesMixin` inlined everything the
`move_to` / `set_gripper` / `rotate_wrist` primitives share with any future
backend: the parameter domains (pose-vector coercion, step-budget cap,
gripper-state and wrist-angle validation), the workspace sanity check, the
registry gripper-metadata contract (`robots.json` -> `<robot>.gripper`), and
the structured success/timeout result envelopes. That half now lives in
`strands_robots.simulation.motion_primitives_base.MotionPrimitivesCore`, which
imports without MuJoCo, and the mixin inherits from it - zero behavior change
on MuJoCo, pinned by the existing primitive suites running untouched. Step 1
of the Isaac motion-primitives parity work (#2123): later children add an
Isaac adapter on top of the extracted core.
