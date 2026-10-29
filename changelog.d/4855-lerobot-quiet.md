### Fixed: recording and loading a LeRobot checkpoint no longer flood the terminal

`record=` now writes its episodes without the SVT-AV1 encoder banner (~25 lines per episode), libav's
`[mp4 @ ...]` notices or the `datasets` `Map:` progress bar; a caller's own `SVT_LOG` still wins, and the
progress-bar and libav settings are put back after each write. `Policy(checkpoint, device=...)` and `server_config`
no longer warn that the checkpoint's saved device (e.g. `cuda` on a Mac) is unavailable, since the device asked for
replaces it, and no longer print "Loading weights from local directory".
