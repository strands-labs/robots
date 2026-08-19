### Fixed
- **rendering**: `render_environment_map`, `bake_environment_map` and
  `environment_map_cache_path` check `face_size` / `equi_w` / `equi_h` against the
  shared `positive_whole_number_error` domain before rendering anything, and
  normalize them with `int()` afterwards. `equi_w=0` previously returned a
  zero-pixel map after six background renders, leaving `derive_key_light` to
  report the scene as black and advise a search flag that cannot fix it; an
  integral float such as `equi_w=32.0` - which that domain accepts - raised a bare
  `TypeError` from `numpy`, and in the cache path spelled a second filename for
  pixels already baked.
