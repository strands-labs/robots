### Fixed: two asset paths are refused rather than silently ranked

With `mjcf_path` live alongside `urdf_path` and `usd_path`, a call naming two was
newly ambiguous, and the `elif` chain would have dropped the loser with nothing
said. `add_robot` now refuses the combination and quotes every path it was given.
