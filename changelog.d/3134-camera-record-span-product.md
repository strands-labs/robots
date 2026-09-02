### Fixed: `lerobot_camera` refuses a capture span its rate records no frame in

The `record` action's loop bound is a product, `int(fps * capture_duration)`, but
`capture_duration` was validated by `positive_finite_number_error`, which reads
that factor alone. Every span below one frame period -- anything under `0.0333`
at the default `fps=30` -- is therefore positive, finite, and makes that bound
zero: the loop body never runs, and the tool returns `status="success"` with
`Frames: 0` while its `Saved:` line names a 258-byte MP4 that no decoder will
open. Measured over 27 recordings at `fps` 10/30/60, 12 reported a complete
recording whose file decodes to zero frames.

Which side of the line a span falls on is not a property of the span, so
`_numeric_option_error` now reads the two factors together once each has passed
its own domain, and the refusal quotes both of them along with the shortest span
that would work. Refused rather than floored to one frame, for the reason the
horizon guard in `Simulation._validate_duration` gives for the same product: a
recording that cannot be honored as asked is a caller error, not a value to
silently substitute. The shortest span each rate can honor still records.

`preview_duration` is deliberately not paired with `fps` this way. The preview
loop is bounded by a `time.monotonic()` deadline whose first iteration always
runs, so a short preview displays a frame rather than none; pairing it would
refuse a preview that works.
