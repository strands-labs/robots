### Added: the init-state arm qpos now lands on Isaac - robosuite joint names map onto the USD articulation

#1827 (fixing #1820) applies LIBERO init states on the Isaac backend as
per-object poses plus robot *base* alignment, decoded through a CPU MuJoCo
compile of the scene MJCF. One slice of the init state was still unapplied:
the robot **arm qpos** (LIBERO's Panda ready pose `[0, -0.161, 0, -2.444,
0, 2.227, pi/4]` + gripper). On MuJoCo the full qpos vector lands in one
write, so the arm starts every episode at the canonical pose the policy's
training data assumes; on Isaac the USD Franka articulation started at its
USD default (all-zero, upright), so the first observation was OOD relative
to the LIBERO training distribution (#1828).

The blocker was naming: the decode model uses robosuite prefixes
(`robot0_joint1..7` / `gripper0_finger_joint1..2`) while the USD
articulation names the DOFs `panda_joint1..7` / `panda_finger_joint1..2`,
and plain suffix matching is ambiguous (`joint1` is a suffix of both
`panda_joint1` and `panda_finger_joint1`). `LiberoAdapter` now strips its
`_scene_robot_prefix` / `_scene_gripper_prefix` and maps the bare names
onto articulation DOFs in the DOF -> scene direction with
longest-suffix-wins semantics (`panda_finger_joint1` claims
`finger_joint1` over `joint1`, so `joint1` lands uniquely on
`panda_joint1`), then writes the mapped values through
`IsaacSimulation.set_joint_positions` before the settle steps. Ambiguous
or unmappable scene joints raise BEFORE anything is written - no silent
partial writes - as do a failed engine write, several registered robots,
and a robot reporting no DOF names. Robot-less probe scenes and sims
without the joint-write seam (`set_joint_positions` / `robot_joint_names`
/ `list_robots`) keep the branch's graceful debug-log skip, and the
MuJoCo model-present path is untouched.

CPU unit tests pin the mapping (suffix-trap disambiguation, whole-token
boundary, dual-arm ambiguity) and the adapter routing with the
recording-stub pattern of `test_isaac_object_pose_state.py`; a GPU
integration test (`tests_integ/simulation/test_isaac_init_arm_qpos_gpu.py`)
asserts the one thing a stub cannot fake - after `_apply_canonical_state`
on a real Franka articulation the observed joint positions match the
decoded init-state arm qpos within tolerance, including holding the pose
through the settle steps (`set_joint_positions` writes both state and PD
targets).
