### Added: `Robot(name, driver="memory")`, a hardware body with no arm attached

`MemoryDriver` (`strands_robots.robot.drivers`) is a bus with no wire: a write lands at once and the next read
returns it, with every target logged in `writes` and `holds` / `estops` counted. The operator gate (`confirm=`),
`approve=`, the e-stop and the watchdog's real-body rule run on it exactly as on an arm, so docs, notebooks and CI
show and test them without writing a `Driver` by hand. It takes no `port=` and no `cameras=`.
