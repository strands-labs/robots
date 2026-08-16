### Fixed

- `add_object(shape="mesh")` on the MuJoCo backend no longer echoes the `size` it
  discards. A mesh consumes no `size` component, so the result asserted an extent
  the call never applied - an omitted vector reported a 5 cm object for an asset of
  any size, and an explicit one read as honoured. The success text now reports the
  extent read back off the compiled geom and names the collision geometry, which
  for every mesh geom is its convex hull rather than the triangles that render.
  Primitive shapes are unchanged.
