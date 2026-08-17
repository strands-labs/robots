### Fixed: the `robot_mesh` gateway survives its own configuration and closes itself

`STRANDS_MESH_GATEWAY_DISCOVERY_WAIT_S` was slept on straight from the
environment, so a non-numeric value raised inside the best-effort handler that
builds the gateway and surfaced as `gateway mesh unavailable` -- naming neither
the variable nor the typo -- while `inf` blocked forever holding the gateway
lock, so the call never returned and no later call could take the lock either.
It is now resolved through the shared numeric domain, honouring `0` as "do not
wait" and falling back to the default for anything a sleep cannot take. The
gateway is also the one mesh with no owner to stop it, so it held an open
session plus its heartbeat and state threads and stayed advertised as a live
peer for the process lifetime; it is now closed at exit. Both knobs are
documented.
