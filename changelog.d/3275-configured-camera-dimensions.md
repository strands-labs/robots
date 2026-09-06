### Fixed: a configured Isaac camera resolution takes the same domain as the render argument

`IsaacConfig.camera_width` / `camera_height` are the defaults `add_camera` and the
render family use when a call states no `width` / `height` of its own. Those
arguments are graded on the shared pixel floor
`strands_robots.utils.positive_count_error`; the fields were graded by a
hand-rolled `< 1` pair, so one resolution got two verdicts depending on where the
caller spelled it. `True`, `640.0`, `640.5`, `nan` and `inf` were stored and then
spent as an array dimension, raising `TypeError` out of `render`, whose contract
is a `{"status", "content"}` dict; `"640"` and `None` raised `TypeError` from the
comparison itself, naming neither the field nor a usable resolution. The fields
now read the same domain the call sites do, and each is graded separately so the
message names the one to fix.
