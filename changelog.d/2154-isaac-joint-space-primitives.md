### Added: joint-space motion primitives on the Isaac backend (`set_gripper` / `rotate_wrist`)

`IsaacSimulation` now carries the two joint-space motion primitives on a new
`IsaacMotionPrimitivesMixin`
(`strands_robots/simulation/isaac/motion_primitives.py`), built on the shared
`MotionPrimitivesCore` extracted in #2153: same parameter domains, same
registry-gripper-metadata-first resolution (`closed`/`open` -> limit-range
end, stale/malformed metadata a loud error rather than a silent heuristic
fallback), same `_GRIPPER_HINTS`/`_WRIST_HINTS` name fallbacks, and the same
structured success/timeout/abort envelopes as the MuJoCo reference
implementation. Previously the only Isaac motion path was the raw kinematic
`set_joint_positions` write - no blocking move, no timeout/abort contract, no
gripper semantics. Resolution happens against the articulation's demangled
URDF joint vocabulary (#1900), stripped of any robot-namespace path prefix;
the drive loops run PD position targets on the Kit-owning thread (through
`run_on_main` when the pump is engaged) and abort with a structured error if
the world is destroyed or the robot removed mid-run. `move_to` (IK-backed,
Cartesian) is a follow-up child of #2123.
