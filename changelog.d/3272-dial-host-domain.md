### Fixed: the host half of a dialled websocket address is graded the way its port is

Every caller-supplied port this package dials is held to one shared domain,
`strands_robots.utils.tcp_port_error`, because an unusable port is not refused by
the transport - it is applied, and surfaces later as an unreachable server that
implicates the service the caller was trying to reach. The host beside it is
interpolated into the same `ws://{host}:{port}` and was held to nothing on two of
the three surfaces that build one, so a value that is not a host was resolved
rather than refused - and the resolution discarded the port the shared domain had
just approved. `host="127.0.0.1/foo"` parses as host `127.0.0.1`, path
`/foo:<port>` and port **80**; `host="ws://127.0.0.1"` parses as host `ws` on
port 80; `host=""` builds no URI at all and raises `InvalidURI` past the
`OSError` channel these clients convert into an actionable hint; and a non-string
is dialled verbatim, `None` as the DNS name `"none"`.

The rule now lives beside the port domain it is the other half of, as
`strands_robots.utils.dial_host_error`, and `RemotePolicy`,
`Cosmos3Policy` and `VeraConfig` all read it - so one address cannot be refused
by one client and dialled by the next. A stated `endpoint=` and an injected
`client=` still own their own address, and `"0.0.0.0"` stays accepted as the way
to reach a server bound on every interface. The ZMQ policy clients keep their
transport's verdict: `tcp://` is not a URI, and zmq refuses each of these
spellings at `connect` with the whole address in the message.

Reaching that verdict reads the value's own characters, so the read is made
behind the module's guarded-read layer and an unreadable host is refused rather
than raised past: a `str` subclass whose `startswith` raises used to escape as a
`RuntimeError` out of a call documented to report through `ValueError`. The
refusal text is rendered through the shared renderers, so it cannot itself fail
on the path that exists to answer an unusable value with text.
