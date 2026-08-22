### Fixed: the mesh forwards the locomotion goal it documents

`SimEngine.run_policy` documents four issue #300 well-known goal keys -
`target_pose` / `target_joints` / `target_velocity` / `world_update` - and offers
the mesh as the remote analogue: "the local-sim analogue of the mesh `tell()`
path, which already forwards these keys". The mesh carried three of them.
`target_velocity` was absent from the wire validator's field allowlist and from
`Mesh._SIM_WELL_KNOWN_POLICY_KWARGS`, so telling a peer to walk left the goal
behind at two layers.

It is the one goal key that is a whole command rather than a refinement. WBC and
`wbc_gait` read `[vx, vy, omega]` and nothing else; with no goal they hold a
standing balance, so a fleet operator asking a humanoid to walk got a rollout
that reported `status="success"` and did not go anywhere. Neither layer says a
word about it: `validate_command` builds its output key by key from an allowlist,
so an unlisted field is dropped rather than refused, and the dispatcher then
builds `policy_kwargs` from its own tuple.

Both providers are reachable over the mesh by construction - the policy-provider
allowlist is derived from the registry, and the comment on that set names WBC as
the provider a hand-maintained copy had already omitted once. `docs/policies/wbc.md`
promises the route in as many words: "sharing the non-VLA goal vocabulary so a
command can flow through `run_policy` / mesh `tell()` without coupling to a
backend".

Dated: the three-key forward set is from #304 (2026-06-06), the first
`target_velocity` consumer from #483 twelve days later, and the docstring that
claims mesh parity from 2026-06-29. Correct when written, never brought along.

The wire now validates `target_velocity` on `target_pose`'s per-component domain -
finite, in range, a bool refused by name rather than read as a 1 m/s command - and
bounds the component count purely as DoS defence, the role `MAX_TARGET_JOINTS`
plays. It does not fix the arity, because the receivers disagree: WBC needs at
least three components and reads the first three, MotionBricks reads a planar
`[vx, vy]`. A fixed length here would refuse a shape a shipped receiver accepts,
and each one already names its own requirement, so a two-component velocity
crosses the wire and WBC answers "target_velocity must have at least 3 elements
[vx, vy, omega], got 2".

The set both layers carry is now graded against `run_policy`'s own docstring
rather than restated, in both directions, so a fifth goal key is held to the same
parity on arrival and an undocumented `target_`-shaped key is still dropped.

One invariant needed widening rather than extending. The dispatcher has two sinks
and no way to ask a provider which it reads, and that choice was pinned as "no
provider names a goal key on its constructor" - true of cuRobo and MoveIt2, and
not of WBC, whose constructor `target_velocity` is a documented static default.
What makes `policy_kwargs` the correct sink there is precedence: a per-call value
overrides the constructor one. Were that reversed, a peer told a new direction
would answer with the direction it was built with, so the precedence is now pinned
directly.
