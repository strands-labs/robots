### Fixed: a missing required hardware parameter is refused by name, with the bus

Eight of lerobot's sixteen robot types declare `port` with no default, so
forgetting it is the commonest caller mistake on the hardware path.
`_create_minimal_config` let the resolved dataclass raise and wrapped whatever it
said, so the refusal read `SOFollowerRobotConfig.__init__() missing 1 required
positional argument: 'port'` -- a lerobot internal, naming neither the parameter
the caller supplies nor a single device on the host. `mode="auto"` had already
enumerated the candidate ports one frame up, logged them, and returned only the
string `"real"`, so `mode="auto"` and `mode="real"` produced the identical message
and the probe's work was invisible.

The mirror-image mistake was already reported properly: a kwarg the dataclass
does not declare is refused with the accepted field names, and the docstring says
why -- it catches a typo like `prot=` at config-build time rather than as a
delayed connection failure with no kwarg in sight. Both halves of "the caller got
a kwarg wrong" now report the same way. A missing port additionally names this
host's servo-bus candidates and their USB serial numbers, because a port path is
a position on the bus that the kernel may renumber across a replug while the
serial number does not move. The scan answers a missing `port` only: naming
serial devices for a missing `remote_ip` would point a network robot's caller at
the wrong bus entirely.

Which serial device is a robot's motor bus is now decided once, in
`strands_robots._serial_discovery`. That rule was inline in `_auto_detect_mode`
-- the keyword list, the WCH CH34x / FTDI vendor ids, the exclusions -- and the
refusal needs the same answer, so a second copy would let the two disagree about
what a robot is. Enumeration there is best-effort, because pyserial is not a
declared dependency of this package: an absent import or a libusb hub glitch
reports no devices rather than replacing a precise refusal with an `ImportError`.
