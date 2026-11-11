### Changed: an agent loads a policy folder from this disk only if no other user could have written it

- **Why:** an agent may name a local checkpoint or dataset folder, and a checkpoint rebuilds what its own JSON names
  while a dataset is the motion a robot replays. A folder another local user created or may write (say in `/tmp`)
  loaded like the operator's own.
- **Now:** under `from_agent()` (robot and world tools, mesh, dashboard runs), every file the loader reads and every
  folder down to it must be owned by this user or root and not writable by group or others; a sticky folder such as
  `/tmp` may sit above it. Symlinks are followed hop by hop and each folder on the way is checked. An approved run on
  a real robot also checks each file on the open file it hashes and copies. Refused before anything loads.
- **Unchanged:** your own `Policy(path)` and `robot.run(path)`, handles you register with `tool(policies=...)`, and
  folders you own under your home folder or `/tmp`.
- **Migration:** a checkpoint in a group-writable shared folder: `chmod -R go-w` it (or copy it into a folder you own),
  or register it as a handle.
