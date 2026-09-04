### Fixed

- A policy's declared `required_bodies` survive being wrapped.
  `PolicyRunner._resolve_required_bodies` read the declaration off the object it
  was handed, and a wrapper is a different object than the policy inside it, so
  a child's declaration was reported as absent. Both halves of the contract went
  with it: the per-tick merge that puts `body.<name>.{pos,quat,lin_vel,ang_vel}`
  into every observation, and the up-front check that a declared body resolves in
  the scene. On a real rollout a tracker declaring one body received 0 of its 4
  pose keys through `PersistentPolicy` and through `CompositePolicy`, and a body
  the scene does not contain -- refused before the first step when the policy is
  driven bare -- ran to completion reporting `status="success"` instead, which is
  the "300 steps of a silently absent key read as a zero pose" the up-front check
  exists to prevent. That reaches the one shipped declarer: `ProtoMotionsPolicy`
  declares its anchor link because the observation carries the floating base only
  and `base_quat` is the pelvis, which the waist joints separate from
  `torso_link`; and both wrappers hand their child the observation this method
  decides the shape of, `PersistentPolicy` verbatim and `CompositePolicy`
  filtered. The declaration is now collected over the whole policy tree through
  the shipped `iter_policy_tree` walk, which is what `Policy.children` exists to
  answer -- "so one probe walks to the policy that answers, instead of every
  probe having to learn the name of every wrapper" -- so a wrapper is honored
  without having to re-declare anything, including a wrapper around a wrapper.
  Every refusal now names the policy that declared the body rather than the
  wrapper it was reached through, so it points at the class to change. A policy
  that declares nothing still sees the backend observation untouched, wrapped or
  not, and a body named twice in one tree still yields one key set.
