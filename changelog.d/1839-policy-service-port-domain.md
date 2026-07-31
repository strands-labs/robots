### Fixed: a policy provider refuses a service port it cannot dial

The four providers that reach a policy service over TCP built the endpoint by
interpolating a caller-supplied `port` with no validation: `groot` and `moveit2`
into `tcp://<host>:<port>`, `cosmos3` into `ws://<host>:<port>`, and
`lerobot_async` into a gRPC target. Every one of those transports connects
lazily, so a port outside the 16-bit range was not refused by the socket -
`create_policy("groot", port=99999)` returned a ready policy whose endpoint was
`tcp://localhost:99999`, and the mistake only surfaced later as an inference
timeout that implicated the server. `port=2.7` produced `tcp://localhost:2.7`;
`cosmos3` and `lerobot_async` accepted `nan`, `None` and `[8000]`, yielding
`ws://localhost:None` and `127.0.0.1:[8000]`.

Each provider now validates the port against the shared
`strands_robots.utils.tcp_port_error` domain - the same one the `gr00t_inference`
tool that *starts* the GR00T service and the mesh bridges already use - before
any transport is constructed. The check is scoped to the branch that dials: a
`groot` local-mode policy, a `cosmos3` policy given an injected client or the
diffusers backend, and a `lerobot_async` policy given `server_address` are
unaffected.
