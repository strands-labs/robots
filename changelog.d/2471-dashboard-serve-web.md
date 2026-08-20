### Added: fleet dashboard --serve-web serves a live Rerun web viewer for headless hosts

`examples/fleet/dashboard.py --serve-web` serves the Rerun web viewer and the
live log stream from the dashboard process instead of spawning a native viewer
window, so a browser - locally or over an SSH tunnel - watches the fleet table
and audit timeline update live on a headless or remote host. It binds
`127.0.0.1` by default per the repo's network-exposure convention (the rerun
CLI's own default is `0.0.0.0`, so the address is always set deliberately);
`--bind` opts into wider exposure and the startup output says so, and
`--web-port` / `--grpc-port` default to Rerun's own 9090/9876. The startup
message prints the ready-to-open `?url=rerun%2Bhttp...` URL plus the one-line
tunnel recipe. Because `--serve-web` is an explicit ask, a missing rerun-sdk
fails with the install hint rather than silently degrading to the terminal
renderer, and the served child is the native rerun-cli binary itself rather
than the `python -m rerun` wrapper - terminating the wrapper orphans the
server and leaks both ports (measured on rerun-sdk 0.26.2). The read-only
mesh surface is untouched: the web viewer changes the render transport only.
