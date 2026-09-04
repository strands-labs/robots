### Fixed

- **Remote inference: a served policy's `required_bodies` now reach the host that
  supplies them.** `Policy.required_bodies` is the "policy declares, runtime
  supplies" contract, and both of its halves run on the machine driving the
  rollout: the runtime resolves the names once up front, refuses one the scene
  does not contain, and merges each body's world pose into every observation. The
  `ready` handshake did not advertise the declaration and `RemotePolicy` did not
  override it, so a policy behind a WebSocket declared nothing and both halves
  were skipped - no pose was ever merged, and a body name the scene lacked was
  never reported, leaving a rollout that reported success. `PolicyServer` now
  advertises the declaration of the whole tree it serves (so a wrapper does not
  hide the policy inside it) and `RemotePolicy` mirrors it, alongside the
  `requires_images`, `execution_horizon`, `actions_per_step` and `supports_rtc`
  it already mirrored. The walk and its type refusals moved to
  `strands_robots.policies.base.collect_required_bodies`, read by both the
  runtime and the handshake, so the two surfaces cannot report a different set
  for one tree. A malformed declaration is refused when the server is built
  rather than on every client handshake.
