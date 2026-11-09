### Changed: a real robot takes remote LeRobot actions only over an authenticated channel

A LeRobot session's frames and action chunks carry no identity, so anyone sharing the link (a zenoh router's
clients, or whoever answers at the server's address) could read the cameras and answer for the server. A real robot
now takes actions only over a `tls`/`quic` link whose zenoh config does mutual TLS
(`Policy("zenoh://tls/host:7447/d", zenoh_config="robot.json5")` or `STRANDS_ZENOH_CONFIG`), or from a server on
the same machine, which is connected to at its literal loopback address. `server_config(..., zenoh_config=)` gives
the server its certificates. Host names and numeric host forms are read consistently (IPv6 zones, IPv4-mapped,
6to4, trailing dots).

Migration: a real robot pointed at a plain `zenoh://gpu-box:7447/d` on the LAN is now refused with a
`PermissionError`. Add certificates on both ends, or keep the old behavior with `trusted=True` or
`STRANDS_ZENOH_TRUSTED=1`. Simulated robots are unchanged.
