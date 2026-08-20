### Fixed

- **mesh**: the three replay-cache critical sections no longer parse the environment while holding their lock. Inbound-command dedup resolved the freshness window, the forward-skew tolerance and the eviction bound inside `_cmd_replay_lock`, and both safety handlers resolved the eviction bound inside theirs; each is an `os.getenv` plus a validating parse that also logs on an unusable operator value. All three now resolve into locals before taking the lock, matching the hoist the safety handlers already did for the two float tunables.
