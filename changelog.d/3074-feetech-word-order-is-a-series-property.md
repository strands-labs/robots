### Fixed: the Feetech word order and position full scale are named as STS/SMS-series properties, and decided once

`Goal_Position`'s 12-bit full scale and the low-byte-first order the package
frames it in belong to the STS/SMS **series**, not to the register and not to
Feetech. The vendor SDK reverses that order on a per-model protocol number, so
an SCS-series servo reads the same two bytes as the byte-swapped value:
`position=1023`, which is full scale on an `scs0009`, reaches it as 65283.

Every surface that stated one of those numbers while naming only the
manufacturer now names the series -- `serial_tool`'s `position` and `velocity`
schema entries (which is the whole of what a model planning a call knows about
the domain), the refusal that quotes the ceiling back, and
`FeetechDriver.tool_spec`'s description, which described itself as an
"SCS protocol" driver while framing STS/SMS words.

Both numbers are now decided in the codec that frames the register:
`MAX_GOAL_POSITION`, `encode_word` and `decode_word` in
`strands_robots.drivers.feetech.protocol`, with the order named rather than
spelled as shifts. The bus, `serial_tool` and `pose_tool` read them; previously
the two-byte split appeared 6 times across those three modules and the full
scale 8 times, so a correction for a different series could move one and leave
the rest behind. `encode_word` also refuses a value two bytes cannot carry
instead of masking it into a different, reachable command.

No byte this package puts on the wire changes for an STS/SMS servo, and cells
graded against `scservo_sdk` under both protocol numbers pin that. Addressing an
SCS-series servo remains out of scope: it needs a second word order and a second
full scale, not a scale option, which is why no surface offers one.
