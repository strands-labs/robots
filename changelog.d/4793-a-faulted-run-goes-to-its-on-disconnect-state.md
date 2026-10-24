### Changed: a run that ends in a fault puts the body in its declared on_disconnect state

When `Robot.run` or `Simulation.run` ends with `stopped_by="fault"` (a policy that can no longer act, a NaN
target, a lost remote server), each of the body's controls is written to its declared `on_disconnect` target
(`hold`, `zero`, `safe` or `apply`) before the run returns. Before, the body stayed at its last command, so a
real arm held wherever the policy left it. In lock-step only the robot that faulted is moved; the rest keep
running. A sim body whose controls all hold is unchanged.
