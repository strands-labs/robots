### Fixed

`randomize(randomize_positions=True)` (MuJoCo backend) now writes the perturbed
pose to `model.qpos0` as well as `data.qpos`, so an object-position
randomization survives the `reset()` every rollout performs before its first
step instead of being reverted before the policy observes it. The offset is
measured from each object's commanded pose, so repeated per-episode calls draw
independent offsets inside `position_noise` rather than compounding into a
random walk, and the result text reports how many objects were perturbed.
