### Changed: a mapping action is clipped into the joint limits once per coercion, not twice, and each clip is one vector operation

`coerce` clipped the held or measured joints a mapping leaves out, then clipped the whole vector again: a run's
`{joint: value}` tick clipped three times (the tick's coercion twice, the write's guard once). The first clip now runs
only when `clip=False` (an agent run, whose own values must reach the safety chain unclipped), so such a tick clips
twice. `clip_to_limits` reads bound vectors computed once per `Description` instead of looking each joint up, about
3x faster (1.4 us to 0.5 us for six joints). Results are unchanged, a NaN target is still refused, and `Robot.command`
still clips every write.
