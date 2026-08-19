### Fixed

- **simulation/terrain**: `generate_heightfield` measures `resolution` against the shared
  `positive_whole_number_error` domain before generating anything, so the grid-cell count that
  sizes the returned field is checked the way `difficulty` already is. A bare `int(resolution)`
  truncated a fractional count silently - `39.7` returned 1521 floats for a number whose square is
  1576.09 - accepted a string outright, and let `None`/`[]`/`inf` raise `TypeError`/`OverflowError`
  from inside `int()`, outside the `ValueError` every other terrain refusal uses. The `>= 2` floor
  is unchanged.
