### Added: fleet emergency-evacuation example (three-phase protocol, benchmark-scored)

`examples/fleet/04_emergency_evacuation.py` drills the evacuation protocol a
plain e-stop cannot provide - a frozen robot in a corridor is itself the
blocking hazard. Phase 1: an injected alarm broadcast aborts every rollout
fleet-wide, rate-limited and audited. Phase 2: each robot runs a pre-validated
deterministic retreat (scripted base/joint setpoints behind the
`EvacuationWorld` seam, so the Isaac adapter drops in later) to muster, ordered
by corridor distance - closest to the path first, never an LLM decision.
Phase 3: mesh lockout engages only after the path is asserted clear, and
resume goes through the HMAC override protocol with operator approval - a
wrong-code or declined resume leaves the lockout engaged. A
`DeclarativeBenchmark` scores the run: any robot re-entering the
clearance-inflated corridor fails it, and it passes only when the personnel
proxy reaches the exit unimpeded and the abort met its deadline. The incident
report is built deterministically from the signed audit log, and the mesh ACL
template gains a read-only `dashboard` role (observe presence/health/safety/
camera; no command-plane grants). (#2183, epic #2179)
