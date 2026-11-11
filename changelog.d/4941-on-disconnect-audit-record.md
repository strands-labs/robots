### Changed: every `on_disconnect` applied leaves one structured audit record

- `apply_on_disconnect` writes one `strands_robots.audit` warning for every call, whoever makes it (the mesh
  watchdog, a local `Watchdog`, a run that faulted): `rpc="on_disconnect"`, tenant, device, the mover's `peer`, the
  `reason`, what it `applied` (`hold`, `targets`, or nothing while the e-stop is latched) and the `outcome` (`ok`, or
  the error's type). An apply that fails is recorded too, then raised as before.
- The mesh watchdog's record names the mover it recalled the arm from and whether it went quiet or vanished.
