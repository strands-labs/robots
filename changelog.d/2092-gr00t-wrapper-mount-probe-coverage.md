### Quality: the wrapper-mount probe behind the deterministic GR00T skip is now tested

`_container_has_wrapper_mount` decides whether an already-running container may be
handed back for a `deterministic=True` start. A container started without the wrapper
mount cannot serve one - the exec'd `python /srv_wrap.py` dies inside the container long
after `start_container` returned success - so the probe reports the mount absent and the
caller fails fast with an actionable "recreate with `force=True`" error.

The two lifecycle tests that reach that branch replace the probe with a stub returning
`True`/`False`, so they pinned what the caller does with each answer and nothing about
how the answer is produced: the probe's whole body was unexecuted. Its contract is now
pinned - a mount at the wrapper path is detected, a path that merely contains it (say
`/srv_wrap.py.bak`) is not, and every docker invocation failure (non-zero exit, missing
binary, OS error) reports the mount absent rather than passing silently - together with
both lifecycle outcomes driven through the real probe instead of a stub.

No library behaviour changes.
