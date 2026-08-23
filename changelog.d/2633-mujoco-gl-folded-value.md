### Fixed: `MUJOCO_GL` is read the way MuJoCo reads it

MuJoCo folds `MUJOCO_GL` with `.lower().strip()`, reads `disable`/`disabled`/`off`/`false`/`0` as
"build no GL context at all", accepts a platform-dependent set of backend names, and raises
`RuntimeError` at import for anything else. Two sites re-derived that vocabulary loosely, so both
answered about a spelling rather than about a backend.

`strands_robots doctor` compared the raw value against `egl`/`osmesa`/`glfw` and read everything else
as unset. `MUJOCO_GL=EGL` renders through EGL and was refused with "MUJOCO_GL not set and no display
detected"; a disabled GL context and a value MuJoCo refuses at import both read as unset, so on any
machine with a display they passed. On macOS the same fall-through recommended `egl`, which MuJoCo
refuses there. Verdicts now name the value MuJoCo reads, refuse a disabled context and an unaccepted
value on their own terms, and only ever recommend a value that platform accepts.

The MuJoCo GL backend gated its NVIDIA EGL vendor ICD staging on an exact `egl`, so `MUJOCO_GL=EGL`
selected EGL while skipping the guarantee that keeps glvnd off Mesa `llvmpipe` - the silent CPU
software-rasterizer fallback. A whitespace-only value is also no longer mistaken for a preference,
so a headless host is auto-configured rather than left on GLFW.

Every value the previous comparison recognised keeps its verdict, byte for byte.
