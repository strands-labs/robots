### Quality: drive every backend's report that the LeRobot dataset stack is unavailable

`start_recording` resolves its recorder out of `strands_robots.dataset_recorder`
before building a dataset, and reports failure rather than raising. Three
different causes land there and each one has a different remedy: the lerobot
extra is absent (install it), the module did not import (repair a partial
strands-robots install), or the module imported but supplied no
`DatasetRecorder`. All three backends carry that block, so there are nine cells
- and only three had ever run: MuJoCo drove two, Newton one, Isaac none.

The nine are now driven, so each cause keeps its own diagnosis on each backend,
together with the two properties that make those diagnoses worth having: they
stay pairwise distinct, and the plain-MP4 fallback each one recommends really
exists on the backend that recommended it. `recording.py` reaches 99% on MuJoCo
and Newton and 91% on Isaac.
