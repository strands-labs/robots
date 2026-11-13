### Fixed: a remote policy reached through a tunnel at 127.0.0.1 gets the internet's settings

A LeRobot policy server reached at a local-looking address (`127.0.0.1`, a LAN address) is now measured after the
handshake: when the round trip is above 20 ms, as through an `ssh -L` tunnel to a far machine, the session gets the
same 30 s timeouts and JPEG frames as a server at a public address, instead of LeRobot's 5 s and raw frames.
`Policy("zenoh://...", network="lan" | "wan")` decides it explicitly; `policy.network` and `policy.rtt_s` show what was
chosen. Which links a real robot may use is unchanged.
