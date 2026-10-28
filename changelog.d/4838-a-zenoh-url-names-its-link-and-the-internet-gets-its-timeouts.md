### Added: `Policy("zenoh://tls/host:7447/<deployment>")` — any zenoh link, and defaults that survive the internet

`zenoh://host:port/<deployment>` stays `tcp/`; `zenoh://<link>/host:port/<deployment>` (or `zenoh+<link>://…`) names the
locator a server actually listens on (`tls`, `quic`, `ws`, `wss`, `udp`) — the form a hosted lease hands out, written as
given. Before, `zenoh+tls://…` was "unknown form" and `zenoh://zenoh.example:7447/d` silently built `tcp/…`.

When the endpoint is not on this machine or its LAN, the client's handshake/action/startup timeouts default to 30 s
instead of LeRobot's 5–10 s (a first call wakes the model and crosses the WAN with two frames), frames go as JPEG, and a
`tls`/`quic`/`ws` link speaks to the router as a `client`. Once the server has described itself, the action timeout also
covers its own `limits.action_deadline_s`. Anything the caller names by keyword wins over every default.
