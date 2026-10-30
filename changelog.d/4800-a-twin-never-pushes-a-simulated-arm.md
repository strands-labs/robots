### Fixed: a twin never pushes a simulated arm or mirrors a NaN or huge reading; close() waits for the step in progress

A hardware Robot's twin spawns where a simulated robot of its type stands, and since `observe()` places its
joints there it pushed that robot off its targets. In MuJoCo the twin now touches the floor where its plain
body would, and nothing else. A NaN reading, or one beyond MuJoCo's 1e10 (`mjMAXVAL`), is no longer mirrored:
in the twin's `qpos` it made MuJoCo reset the whole default world, its time and every robot in it.
`Simulation.close()` now waits for a step another thread is taking, and that thread's next `act()` or
`reset()` raises `RuntimeError` instead of a `TypeError` from MuJoCo; a closed world refuses `camera()`,
`randomize()` and `render()` instead of compiling again, and a real robot carries on when its twin's world
closes.
