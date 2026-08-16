### Fixed

- Raised the `robot_descriptions` floor in the `[sim]` extra from `>=1.11.0` to
  `>=1.23.0`. The built-in registry names a `robot_descriptions` submodule for 57
  robots and that package gains one module per newly packaged robot, so twelve of
  the thirteen releases the old range admitted were missing at least one of them -
  1.11.0 lacked 27, including `so100` and `so101`. Those robots declare no
  `asset.source` fallback and the auto-download path has no Menagerie clone
  fallback, so a resolve inside the old range left them unable to fetch a model
  file at all.
