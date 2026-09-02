### Fixed: `lerobot_train` no longer erases the record of a training session it could not inspect

`SessionManager._load_sessions` dropped a session whose pid exists but whose
process could not be probed (`psutil.AccessDenied` - a run started under `sudo`
and later listed as the invoking user), while keeping a session whose pid was
already gone. It retained the dead and discarded the live.

Because `add_session` and `remove_session` load, modify and write the store
back, and that store is the only place a detached run's pid is recorded, the
omission reached disk on the next training session started or stopped - after
which the process went on holding the GPU with no supported way left to stop it.
Before that it was already invisible: `list` did not show it, `status` and `stop`
reported "not found", and `remove_session` could not delete it on request.

The load step now drops nothing. It classifies a record as running or finished
and reports at `WARNING` any whose process it could not inspect;
`remove_session` is the only thing that removes one. A finished run reaped
between the two probes (`NoSuchProcess`) is retained like any other finished run,
since which of those two paths a finish takes is a race rather than a difference.
Presence in the store was never the running claim - `list` and `status` derive
that from the pid when asked - so retaining a record cannot over-report it.

This also makes the `stopped: null` report reachable from `lerobot_train`: an
exit that could not be observed is reported as unknown, where previously the
record was gone before `stop` looked it up.
