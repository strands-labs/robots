### Fixed: a camera recording reported as stopped is one whose recorder thread has exited

`Simulation.stop_cameras_recording` asked the daemon recorder loop to exit,
joined it with a 5 s budget, and then discarded the outcome -- `Thread.join`
returns `None` whether or not the thread finished. A loop still inside `render`
when the budget expired therefore produced three answers that all read as
success: the MP4 was encoded from a frame buffer the live loop was still
appending to (the flush walks each buffer twice, so the encoded clip and the
reported frame count could describe different lists), the recording was
deregistered so no later call could re-join it, and `Already recording` no
longer refused a second recorder on the same cameras -- putting two capture
threads on one camera set. `get_cameras_recording_status` answered `[idle]`
about that live thread.

The join outcome is now read and reported. An expired join returns
`status="error"` with `stopped=False` and the per-camera buffered frame counts,
encodes nothing, and leaves the recording registered so a later
`stop_cameras_recording()` re-joins the loop and flushes it.

Both `start_cameras_recording` and `start_cameras_recording_synchronous` now key
their refusal on the registration rather than on the `running` flag the loop
outlives. Only a flush deregisters a recording, so a registered one always holds
frames nothing has encoded -- including after an expired join's loop finally
leaves `render` and exits, a state no liveness read can see. A `start` in that
window used to replace the registration and discard those frames while reporting
success, which is the recovery the expired-join error had just promised. The
status verb names all four phases (`recording`, `stopping`, `unflushed`, `idle`)
in its text and as `phase` in its JSON block, alongside `running` and
`thread_alive`; `[idle]` now means only that no buffer is left to encode.
