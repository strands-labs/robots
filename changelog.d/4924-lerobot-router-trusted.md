### Changed: a real robot takes a remote LeRobot policy from off its own LAN only once you vouch for it

`Policy("zenoh://tls/host:port/<deployment>")` (and `quic`, `ws`, `wss`) reaches the policy server through a zenoh
router, and a plain `zenoh://host:port/<deployment>` link to an internet host reaches it straight across the network.
LeRobot's session frames and action chunks carry no identity: any client of the router can read the robot's frames
and answer for the server, and a plain internet link is reached with no server authentication at all. A real robot
now refuses any link off its own LAN with a `PermissionError` before any frame goes out, until the operator vouches
with `Policy(..., trusted=True)` or `STRANDS_ZENOH_TRUSTED=1`. A loopback or LAN link, and simulated robots, are
unchanged.
