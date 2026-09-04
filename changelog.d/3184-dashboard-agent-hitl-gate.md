### Added: a HITL gate pauses agent tool calls that would move real hardware

The operator dashboard's agent runs with a `BeforeToolCallEvent` hook
(`strands_robots.dashboard.agent_hitl`) that pauses the agent on a tool call
whose action can put a *real* robot in motion, instead of refusing it. The
pause is an SDK interrupt, so an operator's "yes" resumes the same turn and the
tool executes; a "no" declines that one call. Stopping is never gated --
`stop`, `emergency_stop`, `status` and `peers` are deliberately absent from
`MOTION_ACTIONS`, and `robot_mesh` is absent because it raises its own
SDK-native interrupt on every physical action, so listing it would ask the
operator twice for one command.

`peer_is_physical` (`strands_robots.dashboard.agent_motion`) decides whether a
peer moves real hardware, so a sim-only fleet is never gated. Operator
responses are recorded through `strands_robots.tools._hitl_audit`.

The gate resolves *which* robot a call would move from a trusted source rather
than from the model's own input, since the `peers` action that lists a
simulated peer's name is deliberately ungated and so that name is always within
the model's reach. A proxy tool's per-build binding names its target; a
direct-serial tool addresses its declared `port`; only a tool that declares
`target` itself is read from that field, so the gate and the tool always resolve
the same peer. `task_post_allowed`'s `confirmed` flag is likewise checked
against `strands_robots.utils.boolean_flag_error` on the path that reads it,
rather than by truthiness, so a confirmation spelled as a string is refused
instead of selecting the confirmed posture.

