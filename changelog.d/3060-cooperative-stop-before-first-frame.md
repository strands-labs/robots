### Fixed: a cooperative stop issued before a rollout's first frame halts the rollout

`start_policy` returns in about a millisecond; the executor worker reaches the
rollout's first frame hundreds of milliseconds later. `policy_running` - the flag
`stop_policy` lowers and the `on_frame` hook reads - was raised by
`_make_run_policy_hook`, which runs on that worker. For the whole launch window
the flag therefore read `False` while `_active_policy_robots` already listed the
robot, so a stop in the window was answered `Was not running on '<robot>'` and
then overwritten by the worker's own raise: the rollout ran to its full duration
having reported that it stopped. Measured on `so101` with a 100-frame rollout
stopped at +100 ms, all 100 frames executed after the stop; the same window
swallowed a stop issued while the rollout was still queued behind a busy
executor.

The launching thread now claims the robot - the blocking `run_policy` on its
caller's own thread, `start_policy` before its submit - recording the stop count
it observed at that instant. The hook raises the flag only while that claim is
still current, and a stop moves the count, so nothing can raise the flag back
behind a stop. A rollout with no launcher claim, i.e. a caller driving
`PolicyRunner` with this hook directly, is still claimed in the hook on its own
thread. `start_policy` submits the shared rollout body rather than the public
blocking entry, which is the half that must not re-claim the robot; the body
forwards to the same `SimEngine.run_policy`, so no default changes.

`stop_policy` now derives its verdict from the same rollout registry
`list_policies_running` reads, so the two surfaces cannot report opposite facts
about one robot at one instant. `Was not running on '<robot>'` stays reserved for
the genuinely idempotent case, where nothing is in flight at all.

Every stop path goes through one durable seam, `SimRobot.request_policy_stop`,
rather than assigning `policy_running = False`: the `stop_policy` action,
`remove_robot`, teardown, and the Device Connect `stop` and `emergencyStop`
handlers. They cannot drift to different answers about whether a rollout was
halted, and `remove_robot` inside the launch window was joining a worker that had
already raised the flag back - it timed out against a rollout it had asked to
stop, and no longer can.

`reset()` still lowers the flag for every robot and the hook still raises it for
the next episode, which is what keeps a multi-episode rollout's own
episode-boundary reset from reading as a stop of the rollout that issued it.
