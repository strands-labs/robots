### Fixed

- `Robot(...).run()` printed `<peer-id> is online. Ctrl+C to stop.` whether or not
  the Device Connect runtime came up. The mesh is stopped *for* Device Connect
  before the bring-up, so a bring-up that failed - most often an absent
  `[device-connect]` extra - left the process reachable over no transport at all
  while announcing the device as online, with only a warning two lines above the
  claim. The status line now reports what actually started and names which
  transport the process no longer has; the warning names the extra to install when
  that is what is missing, rather than the internal module whose import failed.
  Keeping the process alive on a failed bring-up is unchanged, and the success
  path is byte-identical. The foreground tests drove the shipped import unstubbed,
  so whether they exercised the success or failure branch was decided by whether
  the optional extra happened to be installed; they now substitute the integration
  module and open no sockets.
