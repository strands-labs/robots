### Changed: an approved agent run loads exactly the approved bytes for registered checkpoints and Hub checkpoints

- **Registered checkpoints:** a LeRobot checkpoint passed as a policy object (`tool(policies={"pick": Policy(dir)})`)
  reads its files when it attaches, after the operator's yes. Its plan now carries the SHA-256 of those files, and
  the run loads a private, checked copy (as a checkpoint path already did); files changed after the yes refuse the
  run. The developer's object is not changed.
- **Hub checkpoints:** the plan's `files` is `owner/repo@<commit>`, resolved when the operator is asked, and the run
  loads that commit even if the revision moves. A revision that cannot be resolved (offline) refuses the run before
  anyone is asked; a full 40-character `revision=` needs no lookup.
- **Order:** an agent's network-address spec, and a Hub checkpoint `STRANDS_POLICY_REPOS` does not list, are refused
  before the operator is asked (and before any commit lookup), not after.
