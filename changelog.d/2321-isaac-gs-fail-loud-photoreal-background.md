### Fixed: the isaac_gs photoreal background no longer silently demotes to the procedural gradient

`examples.isaac_gs.background.resolve_background` demoted *any* 3DGS failure
-- gsplat not importable, the CUDA rasterizer disabled (which is what a plain
`pip install gsplat` produces in the Isaac container), a scene download/load
error -- to the flat procedural `PanoramaBackground` with only a logged
warning, so a user who asked for the photoreal path got the beige gradient and
a log line, and the repo's own gallery shipped the fallback. A GS background
that cannot initialize now raises `RuntimeError` carrying the pre-built-wheel
install hint; the demotion is opt-in via `allow_fallback=True`
(`--allow-fallback` on `render_demo.py` and `app.py`). `GsplatBackground` also
validates `ply_path` at construction (exists + readable) so a wrong path
surfaces where it was supplied instead of at the first frame inside an app's
catch-all; `allow_fallback=True` covers that construction failure too (an
explicit `--gsplat-ply` with a typo'd path demotes instead of crashing),
while the default posture keeps the precise `FileNotFoundError` /
`PermissionError` rather than wrapping it in the install hint. (#2321)
