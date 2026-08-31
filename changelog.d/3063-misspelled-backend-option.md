### Fixed: a misspelled backend option is refused instead of silently dropped

A simulation backend constructor accepts `**kwargs` as a *tolerating* sink: a
name it cannot bind is dropped, which is what lets one call carry another
backend's options (`num_envs`, `device`) and resolve against whichever backend
is selected. That contract is deliberate and pinned.

It covered a name *some* backend reads, and said nothing about a **misspelling
of a name the receiver itself reads** -- so such a name was byte-identical to
omitting the argument. `Robot("so101", defualt_timestep=0.001)` integrated the
physics at the 2 ms default (half the requested rate) and reported success;
`Robot("so101", positon=[0.5, 0, 0])` spawned the robot at the origin. The
second is the same "absorbed by `**kwargs`" failure the factory's
`position`/`orientation`/`keyframe` pass-through was written to end, reaching
the parameter *names* instead of their values.

The residual names are now screened against the receiver's **own** signature
(`own_keyword_names`, derived rather than listed, so it cannot go stale) and a
close match is refused with a `TypeError` naming the parameter meant. Names
that are neither bound nor close to one are logged at DEBUG, so a genuine
cross-backend option is visible rather than silent.

Deriving the accepted set per receiver -- rather than from a union of every
backend's options -- is what keeps plugin backends working: a plugin's option
names are not statically knowable, so a union would refuse one plugin's option
for a caller targeting another. For the same reason the match threshold is
stated rather than inherited from `difflib`, whose 0.6 default would refuse
`timestep` (0.667 against `default_timestep`, whose substring it is) -- a real
plugin option. Measured, the two populations are far apart: typos of the MuJoCo
constructor's parameters score 0.889-0.968, cross-backend names at most 0.667,
and a test pins that gap rather than the number.

Applied at both backend constructors and at the sim branch of `Robot(...)`,
each screening against its own names, so a genuine backend option still passes
through the factory untouched.
