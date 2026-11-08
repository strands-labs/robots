### Fixed: a refused mesh move no longer arms the watchdog, conflicting rpc answers are refused, and a zenoh mesh can require authentication

- **Watchdog:** a mesh `act` the robot refused (an unknown argument, or a refusal from its safety chain) used to arm
  the watchdog for whatever peer id the request claimed. Its timeout then applied the robot's `on_disconnect` to the
  hardware. Now a refused motion leaves the watchdog as it was. Arguments are checked before the watchdog hears of
  the request, and every `on_disconnect` the watchdog applies leaves a `strands_robots.audit` warning. A control-shadow
  target the safety chain refuses leaves it as it was too, and another peer's refused requests never reset the quiet
  clock of the peer that last moved the robot.
- **Conflicting answers:** `invoke` used to take the first reply that matched the request, so a peer that answered
  first could forge a robot's observation or error. It now raises `RemoteError` when one request gets different
  answers. `stop` and `estop`, which every holder answers, are unchanged.
- **Authentication:** a zenoh mesh takes a zenoh config file (`zenoh://?config=<path>` or `STRANDS_ZENOH_CONFIG`) for
  TLS with client certificates, user and password authentication, or access control. Mounting a real robot on a mesh
  without one logs an audit warning. The mesh never overrides the file's own scouting settings. TLS guards only `tls/`
  links, so "Securing the mesh" in `docs/mesh.md` shows a file that allows only `tls` and turns scouting off.
