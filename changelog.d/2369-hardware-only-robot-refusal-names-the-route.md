### Fixed

`add_robot` and `Robot(name, mode="sim")` now say when a registered robot is hardware-only.
Nine `registry/robots.json` entries declare a LeRobot `hardware` backend and no simulation
`asset`; those names took the unknown-name branch, so a correctly spelled request was
answered with a spelling suggestion and the cause went unmentioned. The suggestion pool was
the whole registry, so the correction offered could itself be hardware-only - following the
sole remedy offered for `earthrover` led to `hope_jr` and then back again. The refusal now
names the LeRobot type and the `Robot(name, mode="real")` route, and suggestions are drawn
from `list_robots(mode="sim")` so every offered name is one the backend can spawn.
