### Fixed: a Booster T1 endpoint opened outside the shared DDS lock is now named

The channel set is constructed under `_DDS_INIT_LOCK` (#3067). What graded that
was a pair of cells driving the *start* of the construction - the open blocks
while a competing holder owns the lock, and the lock is free once the open
returns - plus the package-wide source rule over endpoint-creating calls.

Neither sees *which* constructions the lock actually covers. Moving only the four
channel opens out of the critical section, the plausible shape of a later
refactor, leaves both cells green because the block still begins under the lock,
and leaves the source rule green because that rule derives its vocabulary of
endpoint-creating operations from the Unitree infrastructure modules: it knows
`Init` and not the Booster SDK's `InitChannel`, `InitChannelWithName`,
`InitWithName`, or its `B1LowStateSubscriber` / `B1LowCmdPublisher` /
`B1BatteryStateSubscriber` / `B1FallDownStateSubscriber` constructors. Nine of
the eleven constructions this method performs are outside what it can report.

The lock state is now recorded at every one of them, for both the unnamed and the
named-robot spelling, so an endpoint opened outside the critical section is named
whatever the vendor calls it. Two further cells cover the failure path, which is
the one that leaks: a lock still held after a channel refused to open deadlocks
every later endpoint construction in the process, and the release of a partial
set is pinned as happening outside the lock, which is what the driver's docstring
already claims and nothing measured.
