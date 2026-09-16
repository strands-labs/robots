### Added: the dashboard shows the fleet and runs a simulated robot behind the e-stop

`/api/fleet` reads the registry (`list_robots`) and, when the mesh extra is
importable, the in-process peer table - it never joins the mesh as a side
effect of opening a page. `/api/sim` starts `Robot(name, mode="sim")` in one
worker thread per session (the renderer's GL context is thread-bound and
`MjData` is not shareable), capped at four, and serves the camera as MJPEG
through `rendering.video.mjpeg_frames` plus a `/ws/telemetry` snapshot stream.

The e-stop is `safety_state.Lockout`, folded exactly as that module's tests
describe: `/api/safety/estop` freezes every session and latches `locked`;
every route that would move a sim checks `proves_clear` and refuses with 423;
`/api/safety/resume` leaves the state `unknown`, because a resume is a request,
and the first command a session then accepts is `note_command_accepted`, the
proof. Stopping a session is never refused.

Building an engine is not instant - a model compile plus a renderer - so a
session sits in `starting` for a moment, and an e-stop inside that window
reaches it too: `freeze_all` selects every session that can still step rather
than only the ones already running, and a create whose build overlapped the
e-stop is refused with 423 and dropped instead of being offered as the accepted
command that would report the lockout clear again.

The latch holds against a command that was already in flight. A route admits a
joints or reset request while the lockout is clear, the worker applies it a tick
later, and the red button can be pressed in between: folding that command in as
`note_command_accepted` reported the lockout `clear` while every session sat
frozen, and the next create was admitted. The check and the fold now happen
under the lock the e-stop latches under, so such a request answers 423 and the
latch stands. A queued command that would move the robot is refused when the
worker reaches it rather than applied, which is the promise `sim_session` states.

`/api/sim/{id}/joints` also refuses a non-finite target: `json.loads` accepts the
bare `Infinity`, `-Infinity` and `NaN` tokens and `isinstance(v, float)` admits
them, so the domain the route declares now says finite - the same ingress
`3242-settings-non-finite-numeric-domain.md` documents for the settings store.
