### Added

- `Policy.children` declares the policies a wrapper delegates to, and
  `iter_policy_tree` walks that tree, so a runtime capability probe can reach the
  concrete policy inside a wrapper instead of type-testing the wrapper. The walk
  reads `children` with `getattr`, so a duck-typed policy object that does not
  subclass `Policy` answers a probe as "no match" rather than raising.
