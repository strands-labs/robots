### Fixed: one message that edits the world and renders it no longer races

strands' default `ConcurrentToolExecutor` starts every tool call of one assistant message at once, so `camera c`
then `render c` could render first (`KeyError: no camera 'c'`) and the retried `camera c` then hit
`camera 'c' already exists`. Calls on one world — `sim.tool` and the tools of the robots in it — now run one at a
time in the order the model sent them, under any executor. `Simulation.camera()` and `render()` also check a name
under the world lock, so a concurrent edit cannot land between the check and the change. Two `run` calls in one
message now run one after the other (each through `approve=`) instead of the second being refused as a clash.
