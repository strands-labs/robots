### Quality: pin every "cannot resolve what to drive" refusal in the MuJoCo motion primitives

``move_to`` / ``set_gripper`` / ``rotate_wrist`` each resolve the thing they are
about to command - an end-effector frame, the arm's joint-transmission
actuators, the gripper actuators, the wrist joint - before they touch physics,
and each resolution can come up empty. Six such refusals exist and none was
driven by any test in the tree: a regression turning one into a silent success,
a bare raise, or a message naming the wrong thing was invisible. The sibling
Isaac backend already pinned its wrist-resolution refusal, so MuJoCo - the
reference backend - was the unpinned one.

Adds ``tests/simulation/mujoco/test_primitive_resolution_refusals.py``: four
inline MJCF fixtures (no asset download, no ``mink``, no GL) each removing
exactly one thing a resolution needs, the six refusals with their message
contracts, the shared properties (a structured error naming the action and the
robot; the scene left byte-identical because the refusal precedes any tick), and
an over-reach control asserting the conventional arm still drives all three.
``strands_robots/simulation/mujoco/motion_primitives.py`` 93.7% -> 96.4% over
the primitive suites.

No library code changes.
