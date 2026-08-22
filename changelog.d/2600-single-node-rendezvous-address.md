### Fixed: a single-node elastic launch rendezvouses on an address a peer can reach

`elastic_launch_callable` left two rendezvous values for torch to resolve, and both of
torch's fallbacks are addresses no peer can dial.

`RendezvousStoreInfo.build` resolves `MASTER_ADDR` itself when `local_addr` is None, as
`addr = local_addr or socket.getfqdn()`. On a host whose `getfqdn()` answers with a
reverse-DNS PTR name - the name of an address rather than a hostname - that name has no
forward lookup, so the agent publishes an address the worker store's client can never
resolve. The endpoint fell back to the literal `localhost:0`, whose port is not an
address a client can dial and whose host resolves to two different stacks wherever both
are configured. Both resolutions happen inside libtorch's C++ socket code, so a wrong
value there is a wait that no Python timeout, `pytest-timeout` signal or rendezvous
budget can end: the run parks on "Rendezvous'ing worker group" with no error at all.

Both are derived now. A single-node launch pins `127.0.0.1` and picks a concrete free
port; a multi-node launch with no endpoint is refused with the reason rather than falling
back to a loopback address that can only rendezvous with itself; a multi-node launch
keeps torch's own resolution, because the address really must be reachable from the other
nodes, but says so when the resolved name is a reverse-DNS artifact instead of leaving
the operator to watch a silent hang. `STRANDS_TRAIN_LOCAL_ADDR` and
`STRANDS_TRAIN_RDZV_TIMEOUT_S` are the operator overrides.

Each rendezvous phase now carries a bound in the unit the backend reading it expects:
`read_timeout` for the c10d store, `join_timeout` for its handler, and `timeout` for the
static backend. Previously only the last of the three was set, leaving the two the c10d
path reads at torch's own 60 and 600 second defaults.

Every unusable spelling of the timeout env var falls back to the default rather than
raising, including the non-finite `inf` and `1e999`: converting with `int(float(raw))`
raises `OverflowError`, which a `(TypeError, ValueError)` handler does not catch, so an
operator's typo would have turned a training launch into a traceback.
