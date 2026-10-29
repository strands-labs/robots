### Changed: a robot's id is the name you wrote

`Robot("g1").id` is now `"g1"` (it was `"unitree_g1"`, the registry name), so `sim.robots`, `sim.run({...})`, the
dashboard's `/api/robots/g1/...`, mesh selectors and agent tools all take the name the caller typed. `robot.name`
stays the model's name (`"unitree_g1"`); `id=` still wins; a hardware Robot's default id does not name a LeRobot
calibration file, only an explicit `id=` does. Docs: `Simulation()` accepts `physics="newton"` and `num_envs > 1`
with it; the LeRobot matrix admits 12 of 24 policy families.
