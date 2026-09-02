### Fixed: a documented `driver="strands"` call is graded against the driver it builds

`tests/test_docs_robot_attribute_reads_resolve.py` resolved a documented
attribute read against `strands_robots.hardware_robot.Robot` whenever the
robot's registry entry declared no `hardware.driver`, ignoring an explicit
`driver="strands"` in the documented call - which is the documented way to reach
the native driver of exactly such a robot, and so the only spelling available to
a robot whose driver is registered by a package rather than declared in the
registry. Every read on that object was graded against the wrong class: a
correct line was reported as an `AttributeError`, and a line that really would
raise on the driver was not reported at all. The grader now asks
`strands_robots.drivers.resolve_driver` rather than re-deriving the rule, so the
explicit keyword, the registry declaration and the default are honoured in the
factory's own precedence.
