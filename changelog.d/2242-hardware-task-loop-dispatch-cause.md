### Fixed: a hardware rollout dispatch failure reports its own cause

`Robot._run_control_loop` chose between running the rollout on this thread and
running it on a worker thread by wrapping `except RuntimeError` around both the
`asyncio.get_running_loop()` probe and the nested dispatch. Only the probe's own
`RuntimeError` means "no loop is running".

`Robot.stream(action="execute")` is an `async def` that calls
`_execute_task_sync`, so the nested branch is that surface's live path. A
`RuntimeError` raised there - a thread the pool cannot start, an executor
already shut down - landed in the same handler, whose `asyncio.run` is invalid
by construction on exactly that branch. The caller was told
`asyncio.run() cannot be called from a running event loop` instead of the cause,
and the `task_runner` coroutine the handler built was left un-awaited.

The probe is now separate from the dispatch, so a dispatch failure propagates
with its own cause and nothing is retried on a thread that cannot serve it.
