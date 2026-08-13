### Fixed: a failed Zenoh session close is recorded instead of reported as a clean one

`release_session()` and `_atexit_cleanup()` are the only two `close()` calls in
the module that defines `zenoh_error_types()` -- whose docstring names `close`
among the operations it covers, and excludes programmer errors so they "surface
loudly instead of being swallowed by a best-effort cleanup path". Both caught
bare `Exception` and recorded nothing, so a close that failed was
indistinguishable from one that succeeded: `release_session()` emitted only
`INFO Zenoh mesh session closed`, byte-identical to the healthy path, and
`_atexit_cleanup()` emitted nothing at all. A `TypeError` / `AttributeError`
from a mis-shaped session object was swallowed there too.

Both now catch the module's own `zenoh_error_types()` and log the failure --
WARNING in `release_session()`, where the success line would otherwise
contradict the record, and DEBUG at exit, where there is no claim to contradict.
The "session closed" line is emitted only when the close completed. A failure
outside the documented transport surface propagates, matching how
`Mesh.stop()` already treats its `undeclare` calls.
