### Quality: pin the MuJoCo namespace fallback that keeps a floating base observable after a scene replace

`add_robot` records `robot.namespace = "<name>/"` and every joint lookup in
`simulation/mujoco/rendering.py` prefixes with it, but `replace_scene_mjcf`
recompiles from caller-supplied MJCF that need not reproduce the prefix and does
not rewrite the registry. Three byte-identical `if jnt_id < 0 and pfx:` retries --
in `_robot_base_free_joint`, `_robot_free_base_joint_id` and
`_get_sim_observation`'s read loop -- are what resolve the joints by their bare
names across that gap, and all three bodies were unexecuted by the suite. So the
behaviour they protect was pinned nowhere: with all three removed a floating-base
robot observes `{}` after such a replace, losing every `base_*` key and every
joint scalar, and `start_recording` derives a base-blind dataset schema from it.

4 test functions (6 cases) in
`tests/simulation/mujoco/test_namespace_fallback_after_scene_replace.py` drive
both floating-base shapes -- a humanoid's named `floating_base_joint` and a
mobile base's unnamed `<freejoint>` -- across a namespace-dropping
`replace_scene_mjcf`, and pin the base pose/twist and joint scalars in
`get_observation`, the `base` block in `get_robot_state`, and the joint id
`start_recording` reads to decide the dataset's base columns. The replacement
scene's namespace loss is asserted rather than assumed, and a robot whose joints
land under a third namespace must observe nothing, so a fallback that resolved
the wrong joint would fail rather than pass.

Each retry was removed from a `main` tree in turn: the tree walk's is decisive
for the mobile base (3 cases fail), the read loop's for the joint scalars (2
cases), and the named scan's is executed but masked by its own fallback -- which
is why deduplicating the three into one shared helper is noted as the way to make
that third one enforceable rather than attempted here. Tests only; no library
behaviour changes.
