### Fixed: `create_simulation("newton", solver=...)` refuses a solver that cannot drive a robot

Newton resolves eight solver names and only three of them integrate a rigid
articulated body. The other five were accepted and then failed in two ways a
caller could not act on: `vbd`, `style3d` and `mpm` raised from inside Newton
naming a `ModelBuilder` the caller never touched, while `xpbd` and
`semi_implicit` built and stepped without moving a joint, so `add_robot`,
`send_action` and `step` all reported success over a frozen world. Measured
against a two-hinge arm, a commanded 0.9 rad target moved `featherstone`,
`kamino` and `mujoco` by 0.899 rad and left `xpbd` and `semi_implicit` at
0.0 rad, and stepping under gravity alone left them at 0.0 rad as well.

Naming one of the five is now reported where the caller names it, before any
world, model or solver is built, with the reason and the solvers that can drive
a robot. `describe()["available_solvers"]` reports the accepted names only, and
the solver table in `docs/simulation/newton.md` no longer lists two frozen
solvers as the articulated ones nor a working one as particle-only.
