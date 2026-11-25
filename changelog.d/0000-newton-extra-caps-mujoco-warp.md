### Fixed: the newton extra installs a mujoco-warp and a mujoco that work together

A fresh `uv pip install "strands-robots[mujoco,newton]"` resolved `mujoco-warp` 4.0.1, whose `mujoco-core` 4
overwrote the pinned `mujoco` 3.x, so `Simulation()` failed with
`OSError: libmujoco.so.3.17.0: cannot open shared object file`. The `newton` extra now pins `mujoco-warp>=3.11,<4`
and `mujoco>=3.3.4,<4`, the same `mujoco` cap as the `mujoco` extra.
