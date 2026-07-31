### Fixed: `[sim-mujoco]` declares the QP backend its IK solve needs

`qpsolvers` is a solver-agnostic front end that ships no solver of its own, so
the extra shipping the `move_to` Cartesian transport primitive declared
`qpsolvers` without any backend and got one only through `mink`'s own
`qpsolvers[daqp]` pin. With no backend installed `mink.solve_ik` cannot run and
`move_to` returns `IK bridge unavailable: No qpsolvers backend is installed`,
whose remedy advises installing `strands-robots[sim-mujoco]` - the extra that
did not declare one. `[sim-mujoco]` now declares `qpsolvers[daqp]` explicitly,
matching how `[cosmos3-sim]` already declared its backend for the same
mink + qpsolvers stack. `daqp` is both the bridge's first preference and the
backend `mink` pulls, so the resolved package set is unchanged.
