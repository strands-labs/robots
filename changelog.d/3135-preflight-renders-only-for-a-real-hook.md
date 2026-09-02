### Fixed: the policy preflight renders the scene only when a hook will read the keys

`_preflight_policy_config` sourced the preflight hook's `observation_keys` from
`get_observation` without `skip_images`, so every rollout paid a full every-camera render
before its control loop - and, for `start_policy`, before its first cooperative-stop check -
could begin. `preflight_policy` then discarded those keys for any provider that leaves
`Policy.preflight` at its default no-op, which is 14 of the 15 shipped providers
(`lerobot_local` is the only one with a real hook). Measured with `mock` in MuJoCo headless,
the preflight rendered one frame per scene camera plus the model camera - 7 frames in a
6-camera scene - and a 3-robot fleet took 0.474s to go quiet after `stop_policy` against
0.020s for one robot, because the preflights serialize across the executor workers before any
rollout loop starts. On software GL those renders are ~100x a GPU's, which is where a
fleet-wide abort deadline is missed.

The observation is now looked up only when the resolved class actually overrides `preflight`.
`policy_overrides_preflight` answers that without instantiating the class, and
`_overrides_preflight` is the single implementation of the rule, shared with `preflight_policy`
so the two cannot drift apart. `requires_images` is deliberately not the predicate: it is an
instance property, so an uninstantiated class hands back the truthy `property` object even for
`MockPolicy`, which declares `False`; and an overriding `preflight` needs the camera keys that
`skip_images` omits - exactly the routing information it validates. `lerobot_local`'s
documented camera-routing check keeps the full observation and is unaffected.
