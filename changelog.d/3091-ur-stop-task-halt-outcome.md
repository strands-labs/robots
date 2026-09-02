### Fixed: a UR halt that reports success is a halt the rollout thread has left

`URDriver.stop_task()` signals the rollout, waits for its thread and then
decelerates the arm with `servoStop`. The wait is bounded at two seconds and a
caller-supplied policy blocking on a remote inference call outlasts any bound, so
the case that matters is the one where it expires. Two things were true of it.

`_Rollout.join` was annotated `-> None` and swallowed the timeout, so the
envelope carried `stopped=True` for a thread still inside the loop - while
`get_task_status()` reported `running=True` in the same instant. One driver, two
envelopes, opposite answers about whether the arm is under a task, on the one
path an operator reaches to stop it.

The loop also read its stop event at the top of a step and not again after the
policy returned, so the setpoint that policy call was computing went out *after*
the `servoStop`, measured landing 0.59 ms later. An arm an operator was told had
stopped resumed moving.

`join` now returns whether the thread left the loop, the step re-reads the stop
event after the policy returns and before the setpoint goes out, and `stop_task`
reports an error envelope with `stopped=False` and a reason naming the timeout
when the thread is still in there. The arm is decelerated either way; what is
gone is the claim. `G1Driver` and `Go2Driver` - the fleet's other two drivers that
roll a policy out on a thread - already held both contracts, so the third one is
now graded on the same relation rather than on its own weaker one.
