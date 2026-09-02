### Fixed: a mesh scan is memoised for one resolution pass, not for the process

`_has_meshes` cached its answer on `(directory, st_mtime)` while answering for
the whole subtree. A directory's own `st_mtime` does not move when a file
appears inside one of its subdirectories, so that key cannot observe the thing
it caches, and no later `stat` can tell the two states apart.

`resolve_model_path` is the caller that re-checks after `auto_download_robot`,
and downloads land meshes in the subdirectory the MJCF declares through
`<compiler meshdir=...>`. The re-check recomputed an identical key, was served
the pre-download answer, and fell through to its "no meshes available"
fallback - discarding a successful download and preferring a mesh-less
candidate whose XML will fail to load in MuJoCo.

The memo now belongs to the caller and spans a single pass over candidate
locations, with a fresh one for the pass after a download. Caching is opt-in
per scan, so it cannot outlive a change the resolver itself made. This keeps
the saving the cache was justified by - candidates sharing a directory are
walked once - and removes both the process-lifetime staleness and the
unbounded growth that keying on a float mtime produced.
