### Fixed

- **A `policy_port` is refused by a provider that reads none.** `start_task` /
  `execute_task` validated a supplied `policy_port` on the shared
  `tcp_port_error` domain and then handed it to `_get_policy`, which forwards
  `port`/`host` to whichever provider was named. Ten of the fifteen registered
  providers declare no `port` keyword: six take `**kwargs` and swallowed it, so
  a rollout ran a policy that never dialed the caller's server and still
  reported success, and four take none and raised `TypeError` from inside
  `create_policy` - on the executor thread, after the arm was energized and
  after `start_task` had already answered "Task started". Both surfaces that
  default `policy_provider="mock"` while accepting a port (the Device Connect
  `execute` RPC and the `robot_mesh` tool) reached the silent case.

  The registry already answered this, in two fields for two questions:
  `requires` lists the keywords a caller must supply, so it judges a *missing*
  port, and `config_keys` lists the keywords the provider understands, so it is
  the only one that can judge a port that *was* supplied. Reading `requires` for
  both left the supplied direction unjudged. `provider_reads_a_port` and
  `port_reading_providers` read `config_keys`; `_get_policy` now asks the same
  helper instead of reading the field itself, and the missing-port behaviour is
  unchanged.
