### Fixed: a recording rate `DatasetRecorder.create` cannot honor is refused

`DatasetRecorder.create(fps=...)` writes the recording rate into the dataset
metadata and was unchecked. LeRobot rejects only `fps <= 0`, so a fractional
`2.7`, a `nan` or an `inf` created the dataset and then saved **zero frames** with
`create`, `add_frame`, `save_episode` and `finalize` all returning normally;
`fps=True` recorded a 1 fps dataset (an `int` subclass acting as a 1); and
`fps="30"` dead-ended in a bare `TypeError: '<=' not supported between instances
of 'str' and 'int'` naming neither the parameter nor the method.

The rate is now refused on the same shared `positive_whole_number_error` domain
every backend's `start_recording` already applies to the `fps` it forwards to
`create` unchanged - one rule reached from both surfaces, differing only in
whether a refusal is returned as an error envelope or raised as `ValueError`. The
check sits in the guard block that already holds the schema column names and the
frame shape, so it precedes both the lazy `lerobot` import and the on-disk target
that `overwrite=True` removes: a refused `overwrite=True` call leaves an existing
dataset intact.
