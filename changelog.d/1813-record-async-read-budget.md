### Fixed: `lerobot_camera(action="record")` now reads with the budget the caller set

`timeout_ms` is documented as the budget for this tool's asynchronous reads, and
every handler that selects that read path hands the caller's value to
`async_read` - except `_record_video_sequence`, which the dispatch never passed
the value and which read with a fixed `1000` instead. It was the only handler
taking `async_mode` that did not also take `timeout_ms`, and that omission
propagated into the validation table: because no handler consumed the option,
`record` was deliberately left out of the guard's `timeout_ms` rows, so the value
was neither honored nor checked.

Measured on a UVC camera at 640x480, `action="record"` with `timeout_ms` in
`{default, 1, -5, 9000}` produced byte-identical output every time - 12 frames,
5971 bytes - while the sibling `action="capture"` honored `timeout_ms=1` on the
same device and refused `-5`. A camera that needs longer than a second for a
frame therefore aborted the recording at 1000 ms and reported a timeout against a
budget nobody chose, and raising `timeout_ms` to cover it changed nothing; a
caller needing a tighter bound got a loop that blocked for a second per frame.

The handler now takes `timeout_ms` in the position its four siblings put it,
passes it to the read, and carries the guard row, so an unusable budget is
refused before the camera opens and names the action. A caller who sets nothing
still records with the same 1000 ms budget as before. Two structural pins keep
the two halves together: every handler that selects the asynchronous read must
take a budget, and every action whose handler takes one must be validated for it.
