### Fixed

- **simulation/mujoco**: the dense mass matrix is now built on MuJoCo builds that
  ship the CSR joint-space inertia without the `mju_sym2dense` conversion (3.5
  through 3.9). The CSR rung of the `get_mass_matrix` path called that symbol
  unconditionally although MuJoCo exports it only from 3.10, so it raised
  `AttributeError` inside the declared `mujoco>=3.2.0,<4.0.0` range. MuJoCo's own
  conversion is still used wherever the binding exports it; where it does not, the
  stored lower triangle is expanded through the `M_rownnz` / `M_rowadr` /
  `M_colind` index arrays, bit-exact against `mj_fullM`.
