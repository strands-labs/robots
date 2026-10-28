### Added: a remote policy runs the way it was trained — sequential (hold-for-observation) or splice

`Policy("zenoh://host:7447/<deployment>")` now resolves an **execution** at attach: `"splice"` is LeRobot's engine
unchanged (the next chunk requested one turnaround before the queue empties, from a moving arm, merged with
`chunk_merge`/RTC settings); `"sequential"` sends a frame only when the queue is EMPTY, holds the last target
until the whole chunk arrives, then plays one row per tick — what a model trained on whole chunks from still frames
expects. `execution="auto"` (default) takes the server's word: a descriptor `execution` field when it has one, else
`"splice"` only for a deployment that advertises an RTC contract (`rtc_horizon > 0` / `rtc_model_space_v1`),
`"sequential"` otherwise; `Policy(..., execution="sequential"|"splice")` overrides. `policy.execution` names the one
running, `policy.calls` lists every chunk a sequential session received (wall seconds, rows, server durations).

A policy may now return `None` from `act()`: a **hold**. The loop writes nothing that tick, the body keeps its last
target, a world still advances its period, the dataset keeps its frame (state + the last action), and the pacer moves
its grid one period on — a deliberate wait for the network is `RunResult.holds` / `held_seconds`, never a
`deadline_miss`. `Perturb` passes a wrapped policy's hold through. Before, the only way to run a chunked WFM
sequentially on a real arm was ~130 lines of user code around LeRobot's client, and every wait was booked as a miss.
