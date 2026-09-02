### Fixed: a WebAuthn challenge's TTL is measured on a clock a correction cannot move

`_stash_challenge` stamped each challenge record with `time.time()` and three
decisions read that stamp back: the TTL gate in `_pop_challenge`, the sweep the
next stash performs on the way in, and the order `_evict_oldest` drops entries
in when the table hits its cap. All three are durations this process decides on
its own -- it writes the stamp and it reads it -- so one NTP correction,
`date -s` or resume from suspend moved every one of them.

Measured against the real store with the default 300s TTL: a challenge 301
seconds old was **accepted** across a `-1h` step, extending the window that
nonce stays replayable by the size of the step, while a challenge one second old
was **refused** across a `+1h` step and then swept out of the table by the next
caller's request -- so an unrelated client's arriving login ended the operator's
in-flight ceremony. Across a backward step the cap also inverted: an entry
stamped *after* the correction sorts oldest, so "drop the oldest" dropped a newer
challenge and kept a staler one.

The stamp is now `time.monotonic()`, under a `t_mono` key that names its clock
domain the way the safety subsystem does. The absolute stamps in the same module
are deliberately unchanged: a session token's `iat`/`exp` and a credential's
`created` name a point in time that a browser or an operator correlates with
something off this process, and belong on the wall clock.

The source scan that grades this idiom walked `strands_robots/tools` alone,
which is a root narrower than the shape it grades -- it read clean while the one
offender in the tree sat in the dashboard. It now walks the whole package.
