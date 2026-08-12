### Added: Isaac synchronized multi-robot `run_multi_policy` control loop

`IsaacSimulation.run_multi_policy` now drives multiple robots, each with its
own policy and instruction, in one lockstep control loop (parity with the
MuJoCo implementation, #2122): per timestep every robot is observed once, a
policy is re-queried only when its buffered action chunk drains
(`resolve_chunk_length`, exactly as the single-policy runner sizes it), every
robot's joint targets are applied, and physics steps exactly once. All
Kit-touching work is marshalled through `run_on_main` in two batched hops per
timestep while policy inference stays off the main thread (#1896); a
worker-thread call with no pump running is refused in the tool envelope. The
Isaac override adds a keyword-only `reset_between: bool = False`
(forward-compat with `run_policy`'s multi-episode semantics); requesting
`reset_between=True` returns a structured error citing #1895 rather than
silently skipping the reset. Dataset recording inside this loop is not
implemented yet - a call during an active recording session is refused; the
merged-frame recording path is a follow-up to #2158.
