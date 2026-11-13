### Fixed: the dataset-directory walk closes every directory fd it opens, on each path out

`claim()` (and `remove()`'s re-check) walk a dataset path one directory fd at a time with `O_NOFOLLOW`. Each fd is now
opened through a small context manager and entered on the walk's `ExitStack`, so it is closed when the walk unwinds on
any path, including the symlink refusal and a missing parent. `Robot.set_gains` now returns `None` on both branches
instead of returning the body's `set_gains` result on the hardware-gains path; its signature was already `-> None`.
