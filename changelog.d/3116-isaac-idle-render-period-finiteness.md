### Fixed: an infinite `SO101_IDLE_RENDER_PERIOD` no longer freezes the Isaac live preview

The Isaac backend resolved its idle preview cadence with `float()` behind a
`v > 0` bound, and `inf > 0` is `True`, so `inf`, `Infinity` and an overflowing
`1e999` all became the period. The gate that reads it is
`now_mono - last_render_mono >= period`, which no elapsed span satisfies against
an infinite period: driving the real loop for 6 s of virtual time, the preview
refreshed once at `t=0` and never again while `pump()` kept draining the app - a
viewport frozen for the life of the process, indistinguishable from a stalled
simulation. Finiteness is now decided by the shared numeric domain
(`utils.finite_number_error`), and every fallback is logged so a misconfigured
knob is not mistaken for a stall.

The rule already existed and was already swept - but the sweep was rooted at
`strands_robots/mesh`, so a resolver anywhere else read as absent rather than as
unguarded. That sweep now walks the whole package, which is the scope of the rule
it enforces.
