### Fixed: a cached photoreal backdrop is no longer served to a camera whose clip planes moved

`HybridCompositor` caches each `BackgroundRenderer.render(cam)` result and
reuses it while only the robot moves. The key covered four of `CameraParams`'
six fields - name, image size, pose and intrinsics - and omitted `znear` and
`zfar`, which the backdrops consume: `PanoramaBackground` fills its entire
depth buffer with `cam.zfar`, and `GsplatBackground` hands both planes to the
rasterizer as its clip planes.

MuJoCo derives both planes from `model.stat.extent`, which the compiler
recomputes from the scene bounds, so any scene change (`add_object`,
`attach_bodies`, `load_scene`) moves them while a fixed named camera keeps the
pose, intrinsics and size the key did read - the entry then stood for a camera
the engine no longer described, and the depth test judged the scene against a
far plane it did not have. Geometry beyond the stale plane was composited away:
measured on a MuJoCo tabletop that gained a distant landmark, 5810 pixels
(the whole landmark, at 186 m, against a backdrop parked at 50 m) were replaced
by backdrop, and the same code produced a different frame depending only on
whether the cache was warm.

The key is now derived from every `CameraParams` field, so a field added later
participates on arrival. Callers do not have to invalidate anything after a
scene change; `clear_caches()` remains for releasing memory or for forcing a
re-render after mutating a background renderer in place.
