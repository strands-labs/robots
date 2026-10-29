### Fixed: what an agent is told about the arm and the world

`act` tells a model each joint's unit and range (`shoulder_pan`: rad, -1.92..1.92) from the manifest, so its first move
needs no `describe`. Every rpc reads its docstring's first paragraph, in plain text (no RST markup, no `self`), and its
arguments carry the rest; `contacts` has a description. `add_object` says `size` is metres from the centre out (MuJoCo
half-sizes; the default is a 10 cm cube). `sim.remove(name)` takes a `name` (a robot, object or camera), not `robot`.
A robot's `run.policy` offers only the presets that drive it: `so101` is offered none, `unitree_g1` `holosoma`,
`microduck` `microduck`; `presets.json` records each preset's `robots`.
