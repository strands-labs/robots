### Changed: a real robot on a zenoh mesh any machine may join is observe-only

A real robot mounted on a plain `zenoh://` mesh (no config file) used to take `act` and `hold` from any machine on
the network, with only an audit warning. The peer id in a request is just a claim there, so another machine could
also keep the robot's watchdog from noticing that its real mover had gone. Such a robot now refuses `act` and `hold`
with `RequestRefused` (reads, `stop`, `estop` and `resume` are unchanged), and the audit warning says why.

To move it over the mesh, give zenoh a config file that says who may join (`?config=` or `STRANDS_ZENOH_CONFIG`), or
vouch for the network with `zenoh://?trusted=1` or `STRANDS_ZENOH_TRUSTED=1`. Simulated robots are unchanged.
