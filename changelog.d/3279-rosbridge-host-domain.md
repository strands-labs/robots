### Fixed: the host half of a rosbridge address is graded before a pattern is offered it

`use_rosbridge` and `RosbridgeRobot` interpolate a caller-supplied host and port
into one websocket address, `ws://<host>:<port>`. The port half was graded twice -
by `tcp_port_error`, the domain every port in the package shares, and then by the
transport's own ceiling - while the host half went straight to the transport's
allowlist pattern. A pattern can only be offered a string, so every non-string
host (`9090`, `True`, `b"localhost"`, a list from a JSON tool call) raised
`TypeError` out of `re`: out of a tool whose every other refusal is a result
dict, and out of a constructor documented to report a malformed host as
`ValueError`. The message named neither the tool nor the parameter - the same
defect the transport's port ceiling already records, with `re`'s text in place of
a bare `assert`.

Both surfaces now read `dial_host_error` ahead of the allowlist, the same two
stages the port half beside them has. The set of hostnames accepted is unchanged:
the shared domain refuses none of the hostnames the allowlist admits, and the
allowlist keeps narrowing what the domain admits, because which hostnames this
ROS transport accepts is its own posture rather than the shared domain's. The
truthiness check that used to catch `""` is gone; the domain owns that value and
names the address it cannot build.
