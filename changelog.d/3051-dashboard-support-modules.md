### Added: the `strands_robots.dashboard` support modules -- settings, log redaction, task timeouts and a bounded cache

Four modules the operator dashboard needs before it has a server, each a leaf
that imports no dashboard sibling, so each is reviewable and usable on its own
rather than only once the whole package is present:

- `settings` -- the resolved configuration tree, overrides layered over file
  values over env defaults, with per-key coercion and an override scope that a
  caller can isolate rather than leak into the process.
- `log_redaction` -- secret registration and log scrubbing, so a token that
  reaches a log line is replaced rather than published.
- `task_timeout` -- the ack budget for a task command and the verdict to report
  when one times out. A timeout says the command was delivered and the robot may
  be about to move, because the alternative reading -- that nothing happened --
  is the one that gets an operator hurt.
- `ttl_cache` -- a bounded time-to-live cache. A TTL that only stops *serving* an
  entry is not a bound, so this one prunes by age when read, and by age and then
  insertion order when written.

All four import nothing outside the standard library. They are reachable through
the `[dashboard]` extra all the same, because the package gate in
`strands_robots/dashboard/__init__.py` runs before any submodule and refuses the
whole package when the server dependencies are absent -- one refusal naming the
extra, rather than a bare `ModuleNotFoundError` per module that reads as a broken
virtualenv.

No public entry point is added. A `main()` on the package would reach `cli`,
which reaches `server`, and neither lands here -- an exported callable that
raises `ImportError` is worse than an absent one, so it arrives with the modules
that supply it.
