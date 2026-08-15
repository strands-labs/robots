### Tests: Isaac motion-primitive parity suite closed out, plus real-Isaac integration coverage

The MuJoCo motion-primitive suite is now mirrored case-for-case on the Isaac
backend where the behavior is backend-neutral (#2156, parent #2123):
`tests/simulation/isaac/test_motion_primitives.py` gains the hint-collider
and alias registry-metadata cases, the stale/malformed-metadata refusals for
`rotate_wrist`, the fallback-shift wrist-candidate pin, the
policy-stopped transition guard, and the recording-interplay pin
(primitive motion never feeds the dataset recorder);
`tests/simulation/isaac/test_move_to_ik.py` gains the restart-path grasp
preservation and the move_to halves of the metadata cases. MuJoCo-only
mechanisms (the AgentTool dispatch router, `data.ctrl` interplay,
`_SUBSTEPS_PER_TICK`) are noted as deliberate omissions rather than ported
emptily. Writing the suite exposed one real divergence - Isaac `set_gripper`
/ `rotate_wrist` did not refuse while a policy ran on the robot (`move_to`
did) - filed as #2231 rather than patched in a test-only change; #2235 closed
it and pins the refusal on both primitives, so this suite carries no marker
for it and keeps only the policy-stopped transition case.

`tests_integ/simulation/test_isaac_motion_primitives_gpu.py` drives the same
entry points against a real SimulationApp/Kit runtime (verified on GPU
hardware): `move_to` converges on a known-reachable pose within `tol`,
answers an unreachable target with the structured IK-residual envelope
without stepping physics, and `set_gripper` round-trips the jaw between the
mapped range ends of the articulation's own reported limits. One Kit session
is shared with no `reset()` between cases per the #1895 articulation-handle
caution.
