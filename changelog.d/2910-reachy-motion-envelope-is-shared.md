### Fixed: both Reachy consumers hold the same axis to the same travel envelope

`strands_robots.tools.reachy` exists for one reason, which its own package
docstring states: it holds "what the *two* Reachy consumers must agree on and
neither owns: the motion envelope". Only one of those two consumed it.
`ReachyDriver.send_action` ran `envelope_error` over its action dict; the Device
Connect driver's three movement RPCs ran `finite_number_error` and stopped
there. So for the same physical robot `look(pitch=200)` reported `success` and
sent a head pose built from 200 degrees of pitch on an axis whose travel is
+/-40, `body(yaw=400)` reported `success` and sent `{"body_yaw": 6.98}` — 400
degrees in radians — on an axis whose travel is +/-160, and
`send_action({"head_pitch": 200})` refused both while naming the limit.

The exclusion was argued rather than overlooked. `_motion_domain_error` said the
reachable workspace "is the daemon's to enforce -- it depends on hardware this
library does not model", and that was true when it was written: the reason landed
on 2026-08-07 and `MOTION_ENVELOPE_DEG` landed on 2026-08-26, nineteen days
later, in a package that imports no transport and no driver and is importable
with no Reachy attached. The reason is a claim about what the library can model,
and a later change gave the library the model.

Why nothing caught it: the two surfaces spell the same axis differently. `look`
takes `pitch`/`roll`/`yaw` where the envelope keys
`head_pitch`/`head_roll`/`head_yaw`, and `envelope_error` ignores a key it has no
limit for — so handing it the RPC's own keyword dict bounds nothing and reports
no error. `_ENVELOPE_AXIS_BY_PARAM` is that mapping, and a test grades it against
the live limits so an axis added to the envelope cannot be silently left
unmapped.

Finiteness is still asked first, so an unusable value is named by the caller's
own parameter spelling rather than by the axis it maps to, and a travel
comparison against `nan` — which `abs(nan) <= 40` makes `False` — is never the
message. Per-axis travel is the half that transfers: the envelope's head-body
yaw coupling limit bounds `head_yaw - body_yaw` and needs both values in one
call, which this RPC surface does not offer, and that stays with `send_action`.
The millimetre offsets and the antenna angles carry no envelope entry and remain
finiteness-only, held from both sides so this cannot grow into a bound the
envelope never declared.
