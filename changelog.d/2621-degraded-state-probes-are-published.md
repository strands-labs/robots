### Fixed: a degraded state probe names itself in the published snapshot

Every section of a `strands/{peer_id}/state` snapshot is optional, because a robot may be hardware, sim,
both or neither, so an absent section is ambiguous: a robot with no joints and a robot whose joint read
just raised publish the same thing. A failing probe was reported in the peer's own log, which closed half
of that and could not close the other half, because the observer that has to explain the absence is on
another machine. A fleet view could only tell the two cases apart by reading that peer's log.

It could not always do even that. `_read_state` returns `None` when only `peer_id` and `t` survived and
the state loop publishes nothing for a `None`, so a hardware peer whose one section was `joints` stopped
publishing state entirely for as long as its bus was contended - while its presence heartbeat kept
advertising it, since `_presence_loop` does not read `_read_state`. There was no message on the wire to
inspect at all.

A probe that fails now names itself in a `degraded` block keyed by category, carrying the exception's type
name as `reason` - the discriminator `_warn_read_state_once`'s own docstring names as selecting the
operator's next move, so a contended `ConnectionError` and an uncalibrated `RuntimeError` are told apart -
its message as a `detail` bounded like the topic's other caller-supplied strings, a `failures` count and
`for_seconds`. The entry is removed on the tick the probe answers again, and the clear happens where the
read returned rather than where the probe was skipped, so a sim peer with no motor bus is not an
`hw_joints` recovery and an arm that has gone away keeps its last diagnosis. `for_seconds` is a duration
and is measured on `time.monotonic()`; the stamp it comes from stays off the wire.

A diagnosis is something to report, so the silenced peer above publishes its fault instead of vanishing.
The log is unchanged: the once-per-category warning gate arms for the life of the peer, so a probe
flapping at `STATE_HZ` costs the same one warning it always did while the wire tracks every transition.
