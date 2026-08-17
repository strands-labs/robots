### Docs: the two prerequisites e-stop recovery depends on

A latched mesh e-stop is cleared only by an explicit `resume`, so every
precondition `resume` depends on is a precondition for recovering the fleet.
Two of them were only discoverable after a fleet was already locked out.
`STRANDS_MESH_RESUME_FRESHNESS_S` and `STRANDS_MESH_RESUME_FORWARD_SKEW_S`
bound whether a receiver accepts a resume envelope at all and had no entry in
the environment-variable table -- the `STRANDS_MESH_RESUME_BACKOFF_S` row
already cited the first as if it were listed. The forward bound is the tighter
one and its effect is asymmetric: a receiver whose clock is 6s behind the
operator reads a correct, correctly-signed resume as future-dated, refuses it,
and refuses every retry identically, so it stays stopped until its clock is
corrected or the bound is widened on every peer. `docs/mesh.md` showed
`emergency_stop()` with no documented way back. Adds rows for the three
`_resume_*` knobs, a recovery section beside the `emergency_stop()` call, and
tests pinning both the behaviour and that the knobs stay documented. Each bound
is documented against the clock direction it actually governs: a receiver
*behind* the operator is refused as future-dated by the forward-skew bound,
while only a receiver *ahead* of the operator is refused as stale by the
freshness window, so widening one does not clear the other's refusal.
