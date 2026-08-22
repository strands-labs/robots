### Fixed

`gr00t_inference` now refuses a protected host directory under every name that
reaches it. `_check_volume_safety` resolved the candidate mount through symlinks
and compared it against a literal blocklist, so it compared a directory against
a name. Docker mounts the directory, so a protected directory reachable under a
second name was refused under one spelling and admitted under the other. On a
host whose protected directory is reached through a symlink whose target escapes
the blocklist - macOS ships `/etc -> /private/etc`, and a Linux server with a
separate data volume may ship `/home -> /mnt/home` - the symlink spelling was
refused and the target spelling, along with every file beneath it, was admitted.

The admitted spelling reached two sinks, and the second is the one that matters:
`_check_hf_local_dir_safety` delegates to the same helper, and `hf_local_dir` is
an agent-supplied string that `_download_checkpoint` writes to on the host with
no docker mediation. So the reachable surface was an agent-named write into a
protected directory, not only an operator bind mount.

Both sides of the comparison are resolved now. A protected path that is not a
symlink contributes one entry, so on a host whose protected paths are real
directories the set is the blocklist and every verdict is the one it was; where
`/bin`, `/sbin` and `/lib` are symlinks into `/usr`, the three entries they add
were already covered by `/usr`. The invariant is stated for every entry in the
shipped blocklist, parametrized, so an entry added later is covered without
anyone remembering this.
