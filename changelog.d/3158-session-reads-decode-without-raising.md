### Fixed: the lerobot session tools read a store and a detached log on a decode policy that cannot raise

`lerobot_train` and `lerobot_teleoperate` each read back two files whose
bytes they do not control, and both reads were strict UTF-8.

The detached child's log is opened as the subprocess's `stdout`/`stderr`
sink, so its contents are whatever the child wrote to the fd it
inherited - a native library printing latin-1, a progress renderer, or a
multi-byte character torn in half when the run was killed all produce a
byte that is not UTF-8. Tailing that file in `status` raised
`UnicodeDecodeError`, which is a `ValueError` and therefore passed
`lerobot_train`'s `except OSError` - the handler written to keep the rest
of the report - and reached the action's outer guard. The whole report
was replaced by `Tool execution failed: 'utf-8' codec can't decode byte
...`, discarding the pid, the uptime and the running verdict, all of
which were computed before the log was opened. `lerobot_teleoperate`
catches broadly and kept its report, but printed the codec message where
the tail should have been.

The session store was read the same way, under
`except (OSError, json.JSONDecodeError)`. That handler answers the two
failures the store was expected to have - it is gone, or it is not JSON -
and an undecodable byte is neither, so the degradation both modules
document ("a store that cannot be read degrades to empty rather than
raising") did not happen. Every action that consults the store failed
instead, `stop` included, and both modules record that this store is the
only place a detached process's pid is written down.

Both files are now read with `errors="replace"`, and the writes name the
same encoding as the reads. Damage stays in the field that carries it: a
pid is ASCII either way, so a record damaged in a session name still
names its process and `stop` still signals it, and a log tail is shown
with U+FFFD marking where the child's output stopped being text.
Substitution rather than omission is asserted, because a byte dropped
silently changes the text an operator is reading without saying so.

The existing pin for this store wrote `"{not json"` - valid UTF-8, and
the one corruption shape the handler could already take. The undecodable
shape is pinned beside it in each store's own test file, and
`tests/tools/test_status_reports_a_log_it_did_not_encode.py` drives the
log tail through both tools' `status` action. The clean-log and
not-JSON-store cells pass on both trees, so they read as controls rather
than as casualties of the change.
