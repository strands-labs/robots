### Fixed: a failed G1 bring-up releases the subscribers it already built

`G1Driver.connect_eagerly` builds a `DDSSubscriberSet`, subscribes four topics in
a loop, and then initialises the command publisher. Each `subscribe` call starts
a `ch_reader` daemon thread and matches a CycloneDDS reader; the set is a local
variable, and the driver only takes a handle on it (`self._subs`) once the whole
bring-up has succeeded. Three of the four failure exits past the constructor
returned their reason without closing it, so those readers stayed matched and
those threads stayed looping for a driver that reports itself unconnected -- and
with nothing left holding the set, nothing could close them afterwards.

Measured on all four exits, with a recording endpoint standing in for the SDK's
`ChannelSubscriber`: `start()` refusing had subscribed nothing, so it owed
nothing; an unresolvable IDL class had subscribed one topic and closed none; a
refused subscriber had subscribed two and closed none; and the publisher's own
failure had subscribed four and closed four. That last branch already closed the
set inline, which is what made the other three read as a deliberate choice
rather than an omission.

`cleanup()` cannot recover them either. It closes `self._subs`, which a failed
bring-up never assigns, so it is a no-op for exactly the state this leaves
behind -- and `connect_eagerly`'s own docstring invites the retry that makes it
compound ("the driver stays usable, so a caller can fix the cause and call
again"), which accumulates one more matched reader per attempt.

Every failure exit past the constructor now routes through one `_abort_connect`
helper that closes the set and records the reason, so the release cannot drift
from the report the way an inline close on one branch already had. The reason a
caller sees is unchanged on all four exits, and a successful bring-up still
hands its live subscribers to the driver.
