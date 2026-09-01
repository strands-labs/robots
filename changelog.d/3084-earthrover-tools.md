### Added: the EarthRover's agent verbs, one table-driven module

Six ``@tool`` verbs over the EarthRover driver in a single module -
``rover_move``, ``rover_stop``, ``rover_lamp``, ``rover_state``,
``rover_camera``, ``rover_speak`` - ported from the scout-the-rover bundle
where the transport ran against the real rover. One ``_VERBS`` accessor table
and one shared handle judgement carry the whole family, the shape the g1
consolidation locked: adding a verb is adding a row plus its function, never
a new module file.

All six are re-exported at the package root, so a tool loader reaches them at
the address every other tool answers to (``strands_robots:rover_move``) rather
than needing a submodule path.

Two contracts are worth naming. A timed ``rover_move`` (twist held for a
bounded ``duration_s``) ends in a forced stop and reports **both** halves - a
move whose trailing stop never reached the SDK comes back as an error saying
the rover may still be rolling, because "the twist was sent" is not "the
rover stopped". And ``rover_lamp`` documents what the wire imposes: the SDK
carries the lamp inside the one ``/control`` twist frame, so a lamp write is
a zero-twist write and the rover stops - said in the docstring rather than
discovered in the field.

Reads answer honestly: ``rover_state`` summarises battery, signal, heading
and GPS in one line ("GPS no fix" when there is none) with the full snapshot
alongside, an empty cache is a refusal naming the remedy, and
``rover_camera`` returns the frame as an image content block the agent can
actually see. Driver envelopes pass through verbatim - the driver already
words its refusals better than a wrapper could.

Skipped on purpose: scout's dataset, memory, telegram and voice tooling -
owned by the lerobot and dataset surfaces already in the tree, or specific
to the scout application rather than the rover.
