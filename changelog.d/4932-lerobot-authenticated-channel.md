### Changed: a real robot takes remote LeRobot actions only over an authenticated channel

A LeRobot session's frames and action chunks carry no identity, so anyone sharing the link (a zenoh router's
clients, or whoever answers at the server's address) could read the cameras and answer for the server. A real robot
now takes actions only over a `tls`/`quic` link whose zenoh config does mutual TLS
(`Policy("zenoh://tls/host:7447/d", zenoh_config="robot.json5")` or `STRANDS_ZENOH_CONFIG`), or from a server on
the same machine, which is connected to at its literal loopback address. `server_config(..., zenoh_config=)` gives
the server its certificates. Host names and numeric host forms are read consistently (IPv6 zones, IPv4-mapped,
6to4, trailing dots).

A real robot on a `tls`/`quic` link now also needs LeRobot's robot role: its zenoh config must present a
certificate named `robot-client` and the session joins as a router client (`zenoh_mode="peer"` is refused).
`python -m strands_robots.policy.lerobot zenoh-config --deployment D --out DIR [--host H]` writes a CA, the router,
`policy-server` and `robot-client` certificates and LeRobot's three configs for D: run `zenohd -c DIR/router.json5`
(mutual TLS, discovery off, deny-by-default ACL by certificate name on `lerobot/inference/v1/deployments/D/**`),
`server_config(..., endpoint="tls/H:7447", zenoh_config="DIR/server.json5")` joins that router as a client, and the
robot uses `zenoh_config="DIR/robot.json5"`. Another certificate from the same CA, or a second robot's, can no
longer answer for the server. The `lerobot` extra now includes `cryptography`.

Migration: a real robot pointed at a plain `zenoh://gpu-box:7447/d` on the LAN is now refused with a
`PermissionError`. Run `zenoh-config` and a router with its `router.json5`, or keep the old behavior with `trusted=True` or
`STRANDS_ZENOH_TRUSTED=1`. Simulated robots are unchanged.
