### Added: IK-backed `move_to` on the Isaac backend via the shared MinkIKBridge

The Isaac backend gains the third motion primitive, `move_to` (GH #2155, child
of the parity epic #2123): blocking Cartesian end-effector transport with the
same signature, defaults, structured result/abort envelopes and never-raises
contract as the MuJoCo reference. The solve reuses the shared
damped-least-squares bridge (`strands_robots.simulation.ik.MinkIKBridge`) on
the MuJoCo model the robot's `data_config` resolves to - Isaac registry robots
carry MJCF sources - seeded from the articulation's live joint positions, with
the same deterministic restart schedule, position-only solve when no
orientation is given (5-DOF arms), workspace sanity rejection, and unreachable
targets answered with the IK residual in a structured error.

The MJCF-side solution is reconciled with the Isaac articulation through an
explicit name-keyed joint map (MJCF joint name -> articulation DOF index): a
solved joint with no articulation counterpart is a structured refusal, never a
positional/flat-index write, so an articulation whose DOF order differs from
the MJCF converges identically. The world-frame target is mapped through the
articulation's live base pose, convergence is measured by FK of the live joint
readback through the same bridge frame the solver optimized, gripper DOFs are
held at their live position for the whole descent (grasp preservation), and a
policy running on the robot (`policy_running`) refuses the primitive up front
and aborts it mid-run.
