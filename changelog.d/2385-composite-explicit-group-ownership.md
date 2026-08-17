### Fixed

- `CompositePolicy` now treats an explicit `upper_joints` group as exclusive on every tick.
  A defaulted `lower_joints` kept every name the lower policy emits, so a lower policy whose
  action space reached the upper group also commanded it: when the upper policy emitted that
  name too the merge raised, and when it was silent on it the lower policy's value was
  written to a joint the caller had assigned away, reported as success. The lower policy
  commanding into an explicit upper group is now refused whether or not the upper policy
  emits that name, naming the contested joints, both policies and the remedy. Precedence
  between two defaulted groups is unchanged, and `WBCPolicy` emits no arm joints so the
  shipped locomotion pairing is unaffected.
