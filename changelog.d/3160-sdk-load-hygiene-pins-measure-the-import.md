### Tests

- The two g1 SDK-load-hygiene pins now measure the import in a clean
  interpreter instead of reading process-wide `sys.modules`. A process-wide read
  answers "has anything here imported `unitree_sdk2py`", and two sibling tests
  import the real SDK for good reasons - one decides at module scope whether to
  install a stub, the other grades the wire format against the real IDL - so on
  a host that has the SDK installed the read was already non-empty when a pin
  ran (101 modules in a `tests/drivers` run, 115 in `tests/tools/g1`) and both
  pins failed naming the wrong module. A subprocess asks the question the names
  ask, and with a guarded eager import planted in `g1_actions` the new pins fail
  for it and pass for clean code in both contexts, where the old in-suite
  verdict was the same either way. Production code is unchanged: a clean
  interpreter importing `strands_robots.tools.g1.g1_actions`,
  `strands_robots.tools.g1.use_unitree` or `strands_robots.drivers.g1` loads no
  SDK module.
