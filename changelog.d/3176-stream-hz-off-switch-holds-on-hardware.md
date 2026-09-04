### Fixed: `STRANDS_MESH_STREAM_HZ=0` publishes no step telemetry from the hardware control loop

The variable's documented contract is that a non-positive or unusable value
switches step publishing off rather than changing the rate, and
`stream_min_period_from_env` spells that off as an infinite period on the
reasoning that no elapsed time reaches one. `Robot._execute_task_async` starts
its throttle base at `-inf`, so that a rollout's first step is due wherever the
platform's monotonic epoch sits rather than depending on it being far from zero
- and `monotonic() - (-inf) >= inf` is `inf >= inf`, which is true. One
`publish_step`, carrying a whole observation, action and instruction, therefore
escaped onto the mesh per rollout despite the operator's opt-out; the publish
then wrote a finite base, which is why it was one and not many.

The period is now tested for finiteness once, when the constructor resolves it,
and the publish is gated on the result, so an infinite period is honored rather
than inferred from the subtraction. A configured rate is unaffected, and the
`-inf` base still owes an enabled stream its first publish immediately.
