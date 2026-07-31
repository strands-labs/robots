### Fixed: a rosbridge port that is an `int` subclass dials instead of raising

`use_rosbridge` passed the caller's `port` straight to the `roslibpy` client,
whose WebSocket URL builder gates it on type *identity* rather than `isinstance`
(`type(port) == int`, autobahn/websocket/util.py:85). Every `int` subclass was
therefore refused at every value - including the default `9090` - and refused
with a bare `AssertionError` carrying an empty message, raised from the client
constructor and so outside the `try` that reports an unreachable bridge. An
`IntEnum` port read from a settings module named neither the tool, the parameter
nor the value, out of a function annotated `-> dict[str, Any]`. `RosbridgeRobot`
inherited it: it accepted the port at construction and then failed on every
call, because all of its I/O forwards through this tool.

The port is now normalized to a plain `int` at the one place a client is
constructed, ahead of the connection cache read. The value is legal and dials
exactly as the equal plain `int` does, so it is carried rather than refused. The
shared 16-bit port domain (`tcp_port_error`) is unchanged - it is `isinstance`
-based on purpose, and this was a type-identity defect, not a range one.

Normalizing before the cache read also removes an order dependence that made
the defect look intermittent: the cache is keyed on `(host, port)` and an
`IntEnum` hashes equal to its value, so any earlier plain-`int` call to the same
host and port reused that client and never reached the URL builder. A reproducer
that happened to dial with a plain `int` first reported that everything worked.
