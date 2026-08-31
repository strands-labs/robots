### Fixed: the `MicroduckPolicyBundle` velocity gate arbitrates between its pair instead of selecting into it

`switch_on_velocity` exists to choose between a `move_key` and an `idle_key` by
`|twist|` each tick, and both the module docstring and the parameter's own
documentation say it selects *between* them. It read the magnitude whatever skill
was active, so an explicit `switch(...)` — or `select=` on a previous tick — to
any third skill was undone by the very next tick that carried a
`target_velocity`, before the skill that was asked for had been ticked once.

There was no value a caller could have passed to stay, because every skill the
provider ships beyond the pair reads the same twist slots for something that is
not a velocity. `alpha_sitstand` is the case that makes it a motion fault rather
than a lost tick: for that policy `twist[0]` is a posture flag, `1` sit and `0`
stand, with the same policy sitting, holding and standing back up. So its
documented sit command has magnitude 1.0 and routed to `move_key`, handing the
walking policy the flag as a 1.0 forward velocity, and its stand command has
magnitude 0.0 and routed to `idle_key`. Both directions of the only control that
policy has left it unreachable, and neither reached it even to be refused. The
same applies to `roulade`, the `ball_kick_*` family and `alpha_ground_pick`,
which read phase and behaviour encodings there.

Pollen's `infer_policy.py` draws the boundary in the same place. Its
`_update_policy_session` is documented "Switch between walking and standing
sessions based on vel_cmd magnitude" and returns early for each of its non-pair
modes — `ground_pick_mode`, `sit_mode` ("Don't switch while sitting"),
`slope_mode` and an active `behavior_mode` — before it computes the magnitude.
Our gate carried only the first of those five guards, the one checking that both
sessions are loaded.

The gate now returns early when the active skill is neither of its two keys, one
statement after that existing check and before the magnitude is read. Nothing
about the pair changes: a bundle holding only `move_key` and `idle_key` — the
two-skill shape the documentation's own example builds — behaves exactly as
before in both directions and at the threshold itself, an absent key still
leaves the gate inert, `select=` still wins per tick, and selecting a gate key
again hands arbitration straight back.

Nothing graded it because the only cell exercising the gate builds a bundle
holding exactly the pair, where the missing guard cannot change an outcome: with
no third skill available to be active, "arbitrates between the pair" and "selects
into the pair from anywhere" agree on every input.
