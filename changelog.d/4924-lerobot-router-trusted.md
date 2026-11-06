### Changed: a real robot takes a remote LeRobot policy through a zenoh router only once you vouch for it

`Policy("zenoh://tls/host:port/<deployment>")` (and `quic`, `ws`, `wss`) reaches the policy server through a zenoh
router. LeRobot's session frames and action chunks carry no identity, so any client of that router can read the
robot's frames and answer for the server. A real robot now refuses such a link with a `PermissionError` before any
frame goes out, until the operator vouches for the router's clients with `Policy(..., trusted=True)` or
`STRANDS_ZENOH_TRUSTED=1`. Plain `zenoh://host:port/...` links and simulated robots are unchanged.
