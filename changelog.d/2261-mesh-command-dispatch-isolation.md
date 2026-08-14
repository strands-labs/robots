### Quality: pin the mesh inbound-command listener's parse-and-dispatch contract

`Mesh._on_cmd` is the callback every remote command arrives on, and it answers by
returning -- a refused sample and a dispatched one are indistinguishable to the
caller, so the whole listener was invisible to the suite. The cases around it
drove `_exec_cmd` (the dispatch body) and the wire validator on either side
instead, leaving three properties of the listener itself unpinned.

3 test functions (6 cases) in `tests/mesh/test_mesh_rpc.py` now drive it with
a raw sample the way a subscription would. They pin that valid JSON which is not
an object -- an array, a string, a number, `null` -- is dropped before any field
is read and leaves neither a dispatch nor a response behind; that a well-formed
command from another peer reaches `_exec_cmd` with the decoded payload and
answers on the requester's turn-scoped key; and that the dispatch runs on its own
daemon thread named after the peer, so one wedged command can neither stall the
subscription nor hold interpreter exit open.

Five plausible regressions were applied to the listener and run against both
arms. Four are caught here and are invisible to all 41 pre-existing cases in
that module: dropping the non-mapping guard, making the dispatch thread
non-daemon, dispatching synchronously on the callback thread, and dropping the
peer scope from the thread name. The fifth, the self-echo guard, is already held
by the pre-existing cases, which is why these deliberately do not restate it.

No library behaviour changes.
