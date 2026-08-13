### Quality: Isaac `run_multi_policy` is now exercised on a real Kit runtime

`tests_integ/simulation/test_isaac_run_multi_policy_gpu.py` drives the
synchronized multi-robot loop (#2158) and its merged-frame recording parity
(#2159) against a real `SimulationApp` with two Franka USD articulations in
one stage, closing the GPU-coverage acceptance of the #2122 parity work
(#2160): a recorder-free 2-robot rollout reports per-robot step counts and
measurably moves both robots' joints under real physics; a recording rollout
round-trips ONE merged frame per timestep from disk with both robots'
namespaced `<robot>__<joint>` columns non-zero in every frame (the
real-runtime mirror of the unit-level B4 pin); the rollout driven off the Kit
main thread under `run_pump_forever` completes within a watchdog-bounded
window instead of wedging on the #1896 deadlock shape; and
`reset_between=True` returns the structured #1895 refusal on the real backend
without advancing physics. One module-scoped simulation is shared across the
tests because Kit startup dominates the wall time, and the Franka USD is
resolved through the same 6.0-then-legacy asset-path probe the LIBERO Isaac
example ships.
