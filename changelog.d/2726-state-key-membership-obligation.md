### Tests

- **policies**: the `robot_state_keys` contract file classifies four providers as
  already total without the shared name-list domain - every G1 joint they drive
  is resolved by name inside the caller's list, so a malformed shape fails that
  membership check instead - and drove three of them. `ProtoMotionsPolicy` was
  classified in a one-line addition to the exempt set by the pull request that
  introduced the provider, and the file's own prose still said three, so the
  property that classification asserts was never measured for it. Measured over
  `tests/policies` beforehand, 4499 passing: making its setter fall back to the
  canonical joint order for any shape that is not a joint list left 4499 passing
  and 0 failing, and deleting the membership check outright failed exactly one
  test, the well-formed partial list. None of the five malformed shapes was
  refused by anything. The by-name section is now a table derived from that set,
  the guard the domain-owning half already had, so classifying a provider as
  already-total carries a behavioural obligation rather than exempting it from
  one: all four are driven over the same four malformed shapes, a one-shot
  iterator, a refusal that binds nothing, and the layout each resolves.
  `ProtoMotionsPolicy` also gains the duplicate-name pin its sibling has,
  strengthened for the path that reads the caller's list - the emitted key set
  stays the canonical 29 and the value handed to the tracker comes from the
  first occurrence.
