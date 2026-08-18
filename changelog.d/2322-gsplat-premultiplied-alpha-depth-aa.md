### Fixed: a partial-coverage splat pixel is no longer darkened twice, mis-depthed, or rendered without its trained AA compensation

`GsplatBackground.render` treated gsplat's output as straight-alpha when it is
premultiplied: compositing `rgb * alpha + fill * (1 - alpha)` weighted splat
color by `alpha^2`, a 2.0x darkening at half coverage (grey halos at splat
edges, sparse regions, the zenith). The accumulated `RGB+D` depth was also
never divided by alpha -- the class docstring said it was -- so a soft splat at
4.0 m reported ~2.0 m and the compositor's z-test could wrongly occlude robot
pixels behind it. And the SPZ loader parsed then discarded the header's
antialiased training flag (bit 0; the curated `tabletop.spz` ships `flags=1`),
so mip-splatting-trained opacities rendered in `classic` mode without their
compensation. `render()` now composites with the premultiplied over-operator
(`rgb + (1 - alpha) * fill`), alpha-normalizes depth to metric (promoting
essentially-empty pixels to `zfar` rather than an eps-division phantom), and
passes `rasterize_mode="antialiased"` for AA-flagged `.spz` assets. (#2322)
