### Fixed: the Microduck velocity gate refuses a goal it cannot read, before the selection moves

`MicroduckPolicyBundle`'s `switch_on_velocity` gate reads the per-call
`target_velocity` and picks `move_key` or `idle_key` from its magnitude. It was
the third reader of that well-known goal key in this family and the only one not
held to a domain - the other operand of its own comparison, `switch_on_velocity`,
goes through `positive_finite_number_error` at construction, and the child's
`MicroduckPolicy._apply_command_kwargs` consults `finite_vector_error` for the
same key one layer down.

The child's guard states two reasons for its own existence, and both applied to
the gate. A non-numeric value surfaced as a bare "could not convert string to
float" out of the gate's own `np.asarray` - a `TypeError` for a mapping, where
every documented refusal in this family is a `ValueError` - naming neither the
bundle the caller called nor the parameter it passed. And a non-finite component
made the magnitude `nan`, which is `>=` nothing, so the gate silently selected
`idle_key`: a caller asking the robot to move at a `nan` velocity got the
standing skill.

The selection moving is what made that a defect rather than a lost tick. The
child refused the tick a line later, against a name the caller never used, but
the gate had already written `self._active` - and the gate is skipped on any tick
that carries no velocity, so the flip outlived the failed call and was still
choosing the skill on every tick after it. A component count outside the two the
child documents behaved the same way: the gate read a magnitude out of `[0.5]`,
or out of the first three of `[0.5, 0.0, 0.0, 99.0]`, moved the selection, and
only then did the tick refuse the width.

The gate now asks the two questions the tick itself will ask - the shared
per-component vector domain, then `TARGET_VELOCITY_WIDTHS` - before it reads a
magnitude and before it moves anything, so a refused tick leaves the active skill
exactly as it was and the refusal names the bundle the caller called. Both
questions are the child's own helper and the child's own constant, so the two
readers cannot drift apart on what a velocity is; the regression asserts that
equality over the whole grid rather than as two lists. An absent velocity is
still simply "no goal this tick" and leaves the selection alone, which is why the
early return for it stays above the guard.
