### Fixed

The no-OpenGL-context error now answers for the host reading it, and the camera
recorder no longer reports a render REFUSAL as a camera that is merely "cold".

`render`, `render_depth` and `get_frame` each carried their own copy of one
sentence for every host: "Install EGL or OSMesa for offscreen rendering:
apt-get install libosmesa6-dev". macOS has neither EGL nor OSMesa -- MuJoCo
renders through CGL there, and there is no apt -- so a Mac operator was sent to
install a package that does not exist for their machine while the real cause
kept its cover. Measured on one tree with `sys.platform` set each way, the text
`render` returns is byte-identical: the platform is never consulted.
`no_gl_context_message` now owns the sentence and names the two macOS causes,
which need opposite actions -- a CGL context needs a window-server session,
which a process started by launchd, cron or a bare ssh login does not have; and
a context that worked earlier in the same process was lost rather than missing,
which a fresh process fixes and no install does. Linux keeps the packages, where
they are exactly right, and the advice now lives in one place instead of three.

The recorder thread's warmup had one verdict for every camera that never
produced usable output: "still cold after 30 attempts ... first captured frames
may show gradient artifact". A render can instead come back as a structured
error RESULT rather than a frame, and because that is a returned dict and not a
raised exception the warmup loop's `except` branch never saw it -- so the reason
was dropped at every log level and the recording finished as `status="success"`
with zero frames, no MP4, and an artifact whose only hint was an error count.
No number of attempts fixes a missing GL context, so "cold" is not a slower
version of the same story and its stated consequence understates a recording in
which every frame is missing. Warmup now records the refusal per camera, reports
refused cameras apart from genuinely cold ones, and carries the reason out
through the artifact as `render_refused` and a line in the result text, so
`frames: 0` arrives with its cause attached.

A host that cannot render still cannot record: `status` and `frames` are
unchanged. What changes is that the operator learns why. A genuinely cold camera
keeps the message it always had, and a batch containing both kinds gets both
lines.
