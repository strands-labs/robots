### Fixed: report a teleop auto-accept that could not answer the calibration prompt

`lerobot_teleoperate(action="start", auto_accept_calibration=True)` answers the
child process's calibration prompt by writing two newlines into its stdin from a
background thread. A write that failed - a closed pipe, a child that exited
first - was swallowed with a bare `pass`, so the two outcomes were
indistinguishable: the start result read `status="success"` and reported
"Session Started" either way, the session store reported a live pid either way,
and nothing was logged. The operator was told the session had started while the
child sat at an unanswered prompt.

The failure is now reported at WARNING, naming the session, the reason, and where
to look. A write that *succeeds* stays silent, as the parameter's documented
posture requires. Every other error handler in this tool already reported its
failure; this one was the exception.

Also pins the `stop` half of the session-lookup refusal (`status`'s was covered
and `stop`'s was not, so the two could have drifted apart), and pins that the
store's pid pruning is what makes the tool's "No PID found" refusal unreachable.
