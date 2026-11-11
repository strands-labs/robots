### Fixed: a recording is written and deleted only in a directory no other account can move

A dataset path was checked once, by name, and LeRobot then wrote the take by that name, and a dropped take was
deleted by it. If a directory on that path could be renamed by another account, a symlink swapped in after the check
sent the writes, or the delete, to a directory outside the target. Now `Episode.open` walks the dataset's path one
directory at a time without following symlinks, right before the dataset opens. It refuses (`PermissionError`) a
symlink on the path, or a directory owned by another account or writable by others without the sticky bit. Missing
parent directories are created during the walk. A dropped take deletes its directory only if its path still leads
to the same directory the take created. This applies to `record=` and the dashboard's Record tab, for directories and
`hf://` targets (LeRobot's cache). Behaviour change: a target under a shared directory that others can write to is now
refused; record into a directory only you can change (`chmod go-w`), or one with the sticky bit. A directory you may
search but not list (a `711` `/home`) is walked like `cd` walks it, so it does not refuse.
