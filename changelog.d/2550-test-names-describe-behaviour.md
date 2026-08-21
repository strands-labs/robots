### Changed

- **tests**: no test class or function is named for the release or review round that
  produced it. `TestHardwareConfigV040Followups` and `TestEnsureCaV040Followups` named
  a shipped release and a review trail rather than the behaviour they verify, which is
  what a maintainer scanning for coverage of a behaviour actually reads - and `V040`
  reads as historical once the release is past, which invites skipping the class. The
  hardware class was three unrelated review items, so no single behaviour name covers
  it: it is split into `TestForwardableKwargDropIsReported`,
  `TestThirdPartyPluginGuardIsNarrow` and `TestLerobotExtraShipsAnAarch64VideoDecoder`.
  The CA class shares one subject and is renamed to
  `TestEnsureCaFilesystemDiscipline`. No assertion changed, and the `#NNN` issue
  references stay in the per-test docstrings. A repo-wide guard now refuses a release
  token (three or more digits after a `v`) and a review-round token
  (`Followup`/`Followups`) in a test-case name, while deliberately accepting a one- or
  two-digit version, which names a data format under test rather than a release - the
  nine such names in the tree are pinned as controls so the rule cannot widen into "no
  digits in test names".
