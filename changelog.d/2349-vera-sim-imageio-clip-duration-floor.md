### Fixed

- `[vera-sim]` declared a bare `imageio` with no version bound, admitting releases the clip
  encoder it installs cannot use. `encode_clip` passes the GIF writer a per-frame duration in
  milliseconds, which is imageio's reading only from 2.28.0 on: below 2.16.0 the `imageio.v2`
  import raises immediately after `require_optional` has reported imageio installed, 2.16.0 -
  2.19.0 raise out of `mimsave`, and 2.20.0 - 2.27.0 read the duration as seconds, silently
  encoding the clip 1000x too slow while returning a valid GIF that the encoder's
  "the encoder wrote no clip" check accepts. The extra now states the same
  `imageio>=2.28.0,<3.0.0` range as `[sim-mujoco]` and `[sim-isaac]`.
