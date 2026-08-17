### Fixed: `gr00t_inference(action="stop")` signals every service pid and reports whether the port was actually freed

Port teardown escalated SIGTERM then SIGKILL with `subprocess.run(..., check=True)`
per pid. The pid list comes from a scan, so a pid can exit before it is signalled;
`kill` then exits non-zero ("No such process") and the raised `CalledProcessError`
aborted the sweep, leaving the remaining pids unsignalled and skipping the SIGKILL
escalation. In the `docker exec` branch - tried first - the exception was swallowed
by a `continue`, the host fallback found nothing for a containerised process, and the
call returned `{"status": "success", "message": "No service running on port N"}` while
the sidecar still held the port; `restart` discards that result, so the next bind
failed while naming the bind rather than the surviving process. Signalling is now
best effort per pid, and the port is rescanned after the escalation: `"success"` means
nothing is left holding it, and an owner that survives both signals is reported as an
error naming the port and the surviving pids.
