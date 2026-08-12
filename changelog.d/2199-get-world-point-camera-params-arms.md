### Quality: pin the failure arms of `get_world_point`'s camera-params read

`SimEngine.get_world_point` makes two backend reads -- `get_frame`, then
`get_camera_params` -- and answers for each independently so the method can
keep its documented never-raises envelope. Only the first read's failure was
ever driven: both handlers behind the second one, and the base class's own
`get_frame` / `get_camera_params` defaults behind them, were unexecuted
anywhere in the suite.

The second read can fail on input the call has already accepted and a frame it
has already rendered. MuJoCo's orthographic free camera is the reachable
instance: it rasterizes normally, so `get_frame` returns a full RGB + depth
pair, but an orthographic projection has no pinhole `K`, so
`get_camera_params` refuses. `get_world_point`'s `Returns:` enumerated its
failure causes and named neither backend read, so a caller had no way to
anticipate a refusal that arrives after a successful render.

Adds behaviour tests for both arms -- every member of the handled exception
tuple, the missing-path report, that the backend's reason survives into the
envelope, and that the two reads stay distinguishable -- plus the orthographic
case end to end against a real MuJoCo render with a perspective control on the
same scene. The `Returns:` enumeration now names both reads and the
orthographic cause. No executable line changes.
