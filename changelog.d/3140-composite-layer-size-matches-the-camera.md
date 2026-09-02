### Fixed: a composited frame is the size its own camera parameters describe

`HybridCompositor.render` aligned its layers by truncating to the shortest one
(`h = min(fg_rgb.shape[0], bg_rgb.shape[0])`) while the returned
`CompositeFrame` still carried the `CameraParams` the call resolved. A layer
shorter than that camera therefore produced a frame the camera cannot
describe: on a real MuJoCo scene requested at 320x240, a background capped at
64 px returned a 64x64 composite reporting a 320x240 camera whose principal
point (160, 120) lies outside the returned image -- wrong for any 3D-to-2D
use, at a size nobody asked for, with nothing saying so.

A short layer is reachable from either side, and both are extension points.
`BackgroundRenderer` is a public `Protocol` accepted by
`HybridCompositor(background=...)` and `set_background()`, so any third-party
rasterizer that caps its output supplies one; and `IsaacSimEngine.get_frame`
already decided this exact question the other way for the foreground, raising
on a size it cannot render "rather than silently dropping the requested size".

Every layer -- both `get_frame` buffers and both `BackgroundRenderer` buffers
-- is now held to the size the resolved camera declares and refused with a
`RuntimeError` naming the side, the buffer, both sizes and the remedy, for the
reason the no-depth refusal beside it already states: silent wrong output is
forbidden. A conforming render is unchanged, byte for byte.
