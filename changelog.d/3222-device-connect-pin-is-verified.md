### CI: the device-connect source pin is measured in the test env rather than announced

`test-lint.yml` redirects `device-connect-edge` and `device-connect-agent-tools`
to `arm/device-connect@<ref>` when a pull request touches
`strands_robots/device_connect/`, and printed `Pinned device-connect packages to
...` as its only trace. Hatch creates the test env silently, so the job log
carried no evidence of what the suite's interpreter resolved - while the outer
`pip install -e ".[all,dev]"`, which reads no `UV_OVERRIDE` and installs nothing
the suite imports, printed the published wheel's download in full view. #3222
read that as the pin never being consulted; measured, it is consulted, and the
defect was that CI could support the claim in neither direction. A new step runs
`scripts/check_device_connect_source_pin.py` through `hatch run` between env
creation and the suite: it reads each distribution's PEP 610 `direct_url.json`
and fails, naming the origin actually loaded, unless both come from the
announced repository and ref. Pinned by
`tests/test_device_connect_source_pin_is_verified.py`.
