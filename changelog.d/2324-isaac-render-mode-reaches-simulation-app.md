### Fixed: an Isaac render_mode now selects the renderer it names

`create_simulation("isaac", render_mode=...)` stored the value and gated a few
code paths (headless refusals, camera warmup, the native-resolution upscale)
but never forwarded it, so `SimulationApp` always launched with the default
real-time renderer and `rtx_pathtracing` was silently a no-op renderer-wise --
the silently-dropped-kwarg pattern. `create_world` now maps the mode to the
documented `renderer` launch key (`rtx_realtime` -> `RayTracedLighting`,
`rtx_pathtracing` -> `PathTracing`; `headless` selects nothing), and because
`SimulationApp` is create-once per process, a later request the existing
singleton cannot honour is reported as a warning naming the dropped keys
rather than silently returning an app configured otherwise. Two adjacent dead
helpers are deleted per repo rule 10: `_configure_renderer` (carb-settings
DLSS mitigation the RTX pipeline re-asserted away every render tick -- never
called, only cited by a comment that read as if it were active) and the
backend `_add_lighting` (never called; scene lighting is owned by the scene
author). (#2324)
