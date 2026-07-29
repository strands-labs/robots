### Fixed: the all-missing `robot_state_keys` message now recommends a binding that resolves

When none of the configured `robot_state_keys` appear in the observation, the
diagnostic ended in one fixed remedy for every caller, whose worked example was
`embodiment='so101'`. On real SO-101 hardware that example is self-defeating:
`so101` is the SIM embodiment and declares numeric MuJoCo actuator names
(`'1'..'6'`), while a lerobot `SOFollower` reports `shoulder_pan.pos` and
friends -- so an operator who followed the advice verbatim re-entered the
identical branch and was handed the identical advice, on exactly the path the
message's own stated trigger ("a robot/sim that reports named joints")
describes. The embodiment they needed, `so_real`, was never mentioned. The cause
sentence had the matching flaw: it asserted the generic-key explanation
("`joint_0..joint_N` were paired with ...") unconditionally, so it also
mis-described the `'1'..'6'` case, where the configured keys are named rather
than generic. The remedy is now DERIVED from the observation in hand -- it names
the embodiments whose declared `state_keys` this observation actually satisfies,
so applying the suggestion cannot land back in the same branch -- and the cause
is decided from the configured keys instead of asserted. Swapping one fixed name
for another could not have fixed this, because no single name is sufficient:
`so_real`, `koch_real` and `omx_real` declare identical state keys, so an
observation of those six `.pos` names genuinely cannot choose between them.
Ambiguity is therefore reported rather than resolved -- all matching names are
offered, since guessing would trade a guidance loop for the wrong robot -- and
when nothing matches, only `set_robot_state_keys()` is pointed at, with no
embodiment value named. Both documented mechanisms stay named in every branch, so
the long-standing contract that this diagnostic points at both `embodiment=` and
`set_robot_state_keys()` is unchanged. Behaviour is otherwise untouched: the same
resolved ordering, the same `generic_state_keys_used` telemetry, the same
warn-once, and `strict_keys=True` still raises -- only carrying guidance that
works.
