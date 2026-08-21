### Fixed

- **mesh**: `_load_acl_cached`'s docstring now describes the ACL cache that ships. It
  named the superseded pre-#218 `is_default_acl_in_use` + `resolve_acl` pair as its two
  callers (three functions call it, and only `snapshot_acl` is wired from outside the
  module), said callers "get the same dict object" (every return path deep-copies, so
  they share contents and not identity), and claimed to close the `Mesh.start` TOCTOU
  window that `snapshot_acl` attributes to its thread-local snapshot while calling this
  identity-keyed tier a by-design refresh window. The caller census and caller names are
  now graded against the module's own AST, so a fourth reader of the cache is reported
  when it is added.
