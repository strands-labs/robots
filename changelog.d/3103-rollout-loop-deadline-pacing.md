### Fixed: a rollout runs for the wall clock it was asked for, whatever a step costs

`duration` is documented as wall-clock seconds and `fast_mode=False` as
real-time pacing. Both were false by the cost of one step: the three rollout
loops slept `1 / control_frequency` AFTER each step, which is a delay where a
rate needs a deadline, so the wall clock a step spent on inference, the physics
substeps, a render for the video and the recorder's frame write was added to the
period instead of subtracted from it and the loop ran at `1 / (period + work)`.

Measured on a MuJoCo so101 rollout asking for 2.0s, before -> after:

| case | achieved | achieved Hz | asked Hz |
| --- | --- | --- | --- |
| free policy, 50 Hz | 2.15s -> 2.00s | 46.4 -> 50.0 | 50 |
| 10 ms inference, 50 Hz | 3.15s -> 2.00s | 31.8 -> 50.0 | 50 |
| 30 ms inference, 30 Hz | 3.90s -> 2.00s | 15.4 -> 30.0 | 30 |

Sim time was exact in every row (2.0s of integration), so neither the physics
nor a recorded dataset's timebase was wrong - only the wall clock the caller was
promised, and the rate anything watching the rollout saw.

`PolicyRunner.run` and both backends' `run_multi_policy` now pace on
`strands_robots.mesh.pacing.Ticker`, which the mesh's publish loops were
converted to for this exact argument. A step whose work fits inside the period
is fully absorbed; a step that overruns drops the missed deadlines rather than
chasing them, so a slow step is followed by a gap instead of a burst of
back-to-back setpoints. `fast_mode=True` is unchanged and unpaced.

The mesh's pacing inventory scans for the `stop_event.wait(period)` spelling of a
delay and so could not see these three: they paced with `time.sleep(period)`.
`tests/simulation/test_rollout_loop_paces_on_a_deadline.py` covers that second
spelling for the simulation package, deliberately without requiring the call to
sit inside the loop body - the runner's pace lived in a per-step helper, and a
loop-scoped version of the check named the two backends and missed the runner.
