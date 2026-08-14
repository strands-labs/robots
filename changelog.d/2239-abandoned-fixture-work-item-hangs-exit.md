### Quality: a fixture work item a test gives up on no longer hangs the run

`Robot.start_task` submits the rollout to `Robot._executor`, and the fixtures
that build that executor for a `Robot` assembled via `__new__` wait for the
submitted work with `future.result(timeout=...)`. When that wait gave up, the
work item was still running on a **non-daemon** `ThreadPoolExecutor` worker, and
`ThreadPoolExecutor` registers an interpreter-exit hook that joins every worker
it started -- `shutdown(wait=False)` does not detach one already running. So the
process could not exit: pytest printed the verdict and the job then hung,
delivering a red test as a hung job rather than a failed one. Measured on that
shape, pytest reported `1 failed in 2.03s` and a 45s wall clock had to kill the
process.

The three test modules that build such an executor and can abandon its work
item now use `tests._daemon_executor.DaemonThreadExecutor`, which keeps
`ThreadPoolExecutor(max_workers=1)` semantics -- one worker, submissions
serialized, a real `Future`, `shutdown(wait=True)` draining -- with a daemon
worker the interpreter does not join. The verdict is unchanged; only the hang is
gone. A structural check keeps a new abandoning fixture from reintroducing it.
