### Fixed

- **examples/fleet**: the `--serve-web` startup message classifies the bind
  address by class rather than by equality with the default, so any loopback
  address (`127.0.0.0/8`, not only `127.0.0.1`) gets the SSH tunnel recipe
  instead of a network-exposure warning, and the recipe forwards to the address
  the server is actually on. A hostname stays unclassified, matching the Rerun
  CLI's own IP-literal-only `--bind` domain, and an IPv6 literal is bracketed
  wherever a port follows it. The default bind is byte-identical.
