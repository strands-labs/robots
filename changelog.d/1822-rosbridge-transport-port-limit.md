### Fixed: the rosbridge surfaces refuse a port their transport cannot address

`use_rosbridge(port=65535)` raised a bare `AssertionError` out of a function
annotated `-> dict[str, Any]`. 65535 is a legal TCP port, so the shared 16-bit
domain (`strands_robots.utils.tcp_port_error`) accepted it and the tool dialed -
but the WebSocket transport behind roslibpy builds its URL with
`assert port is None or (type(port) == int and port in range(0, 65535))`, and
`range(0, 65535)` stops one short. The assert carries no message, so an agent
driving the tool got an exception where every other refusal is a result dict,
and the failure named neither the tool nor the parameter. Ports `1`, `5555` and
`65534` were unaffected.

`use_rosbridge` and `RosbridgeRobot` now refuse the port the transport cannot
carry, with a message naming the surface, the parameter and the addressable
range. The refusal happens before the backend probe, so it reads the same
whether or not roslibpy is installed and no socket is dialed to discover it, and
`RosbridgeRobot` refuses at construction rather than at first use - it forwards
every call through `use_rosbridge`, so such a port is a dead bridge.

The shared 16-bit domain is deliberately *not* narrowed to match: the bound
belongs to one transport, not to the port space, and `gr00t_inference` still
accepts the whole range. The divergence and the autobahn behaviour it rests on
are pinned together, so a future release that accepts 65535 fails the premise
test rather than leaving a silently over-strict guard behind.
