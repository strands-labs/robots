### Fixed: `gr00t_inference` refuses a numeric option it cannot honor

`port`, `denoising_steps` and `timeout` were forwarded unvalidated from the
agent tool into a detached `docker exec` command line and a port-poll loop,
neither of which reports a value back to the caller. `port=nan` reached the
inference server as `--port nan` and the container mapping as `-p nan:nan`;
`denoising_steps=0` asked a diffusion policy for zero denoising steps; and
`timeout=0` skipped the poll loop entirely, so a service that had started and
opened its port was reported as having "failed to start", while `timeout=inf`
never gave up at all.

Each option is now held to the shared domain for its kind before any container,
checkpoint or socket work begins, and only for the actions that read it - `list`
scans a fixed set of ports rather than the caller's, and `lifecycle="teardown"`
opens no socket, so neither is refused for a value it ignores. The 16-bit TCP
port range moves to `strands_robots.utils.tcp_port_error`, which `use_rosbridge`
and `RosbridgeRobot` now share, so the same port cannot be refused by one
transport onto a service and accepted by the next.
