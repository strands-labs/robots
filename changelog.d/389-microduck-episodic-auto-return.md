### Added: episodic behaviors in `MicroduckPolicyBundle` auto-return on their own

`MicroduckPolicyBundle` used to have one switching mode -- the velocity gate
between a `move_key` and an `idle_key`. Pollen's reference `infer_policy.py`
also carries a second, orthogonal mode: **timed episodic behaviors**
(`kick_left` / `kick_right` / `roulade`) that run for a fixed duration and
then auto-return to a default skill. Without this, a caller who did
`bundle.switch("kick_left")` stayed on `kick_left` forever, feeding the
walking/standing observation contract to an episodic policy that expects to
end within ~1.2s.

Two new constructor arguments cover the gap. `episodic_skills` maps a skill
name to a duration in seconds; every entry must name a held policy and every
duration goes through the shared `positive_finite_number_error` domain.
`default_skill` is the revert target the bundle returns to when a timer
expires -- refused if it names no held policy, and refused if it is itself an
episodic skill, because an auto-return that re-arms its own timer never
terminates.

`bundle.trigger(name)` arms an episodic behavior; the tick loop then runs it
until its timer reaches zero, at which point the bundle reverts to
`default_skill`. The timer counts down at `1/control_frequency` per tick, read
off the same `Policy.control_frequency` seam every other consumer uses; a
bundle asked to run an episodic behavior without being told its clock refuses
loudly rather than assuming 50Hz. A running episode inhibits the velocity gate
so a walk command mid-kick does not preempt the episodic skill, and `reset`
clears the FSM before the next rollout.

That `reset` normalisation is scoped to a bundle that declares episodic
skills. A bundle that declares none is left exactly as it was: the active
skill is the caller's to choose there, `switch` is the only thing that moves
it, and the rollout calls `policy.reset(seed=...)` once per episode -- so
normalising regardless of whether an episodic behavior was ever declared
would have discarded an explicit `switch` at every episode boundary of a
multi-episode run, with nothing reported.
