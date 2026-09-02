### Fixed: a session reported stopped is one whose process is confirmed gone

`lerobot_teleoperate` and `lerobot_train` both implement `action="stop"` as
SIGTERM, a fixed two-second sleep, SIGKILL if the pid still exists - and then an
unconditional `"**Session Stopped**"` plus an unconditional
`remove_session(...)`. Sending SIGKILL is not the process exiting. The kernel
delivers it asynchronously, and a task inside an uninterruptible wait (a serial
ioctl on the teleoperation bus, a stalled CUDA or network call in a training
step) stays in the process table until that wait returns.

Driving the real verb against a real child that ignores SIGTERM, the process was
still running at the moment the tool reported it stopped in 5 of 5 trials - and
the record had already been dropped. That store is the only place a detached
session's pid is written down, so the process kept driving the robot with no
supported way left to stop it.

Both verbs now capture the process identity before signalling anything, so the
escalation is aimed at the process they found rather than at whatever holds the
pid two seconds later, and they report the outcome they can establish: success
with `stopped: true` once the process has left the table, an error with
`stopped: false` when it is still there, and an error with `stopped: null` when
this user may not inspect it and neither answer is available. The record is kept
in both error cases, which is what keeps the session stoppable on the next
attempt. The rule lives in one place (`strands_robots.tools._process_stop`) so
the two verbs cannot drift apart on it.

The confirmation replaces the fixed sleep, so a session whose process exits
promptly now stops in 0.002s instead of 2.001s. The sibling teardowns this
aligns with - `gr00t_inference._stop_service`, which rescans the port after the
escalation, and `policies.vera.server_runner.stop`, which waits after each
signal - are unchanged.
