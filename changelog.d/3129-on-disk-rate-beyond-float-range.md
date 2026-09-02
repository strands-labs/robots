### Fixed: an on-disk recording rate beyond the float range is unreadable, not an exception

`recorder_dataset_fps` reads a dataset's declared frame rate and is documented to
answer `None` when the dataset "does not report a usable whole rate", so that "an
unexpected LeRobot layout must not block a valid resume". It resolves the rate
through `float`, because `numbers.Real` -- the predicate it classifies on, so every
spelling the fps domain accepts is read -- carries no ordering against `int` and no
`int()` overload. That conversion could raise, and the raise escaped the contract.

The rate arrives off disk, where no domain has been asked. `meta/info.json` is
JSON, whose integer literals are unbounded, and LeRobot's `fps` field is an
unenforced dataclass annotation, so `LeRobotDataset` opens such a dataset without
complaint and hands the value straight to the reader. Measured through
`start_recording` against a recorded LeRobotDataset (SO-101 in MuJoCo, 1 episode,
30 frames, 30 fps) whose on-disk `fps` was edited to a 401-digit integer:

    on-disk fps      recorder_dataset_fps      start_recording
    30                                 30     success
    29.97                            None     success
    0                                None     error: fps must be positive, got 0
    1e400 (-> inf)                   None     success
    10**400          raises OverflowError     error: Dataset init failed:
                                              int too large to convert to float
    -10**400         raises OverflowError     error: fps must be positive, got -1000...

The dataset had opened -- LeRobot returned its metadata without complaint -- so the
reported subject was wrong, and the message names neither the field nor a remedy,
while the fractional and infinite rates beside it resumed as the unreadable
layouts they are.

Resolving the rate through `float` converts before the sign can be tested, so a
negative rate of that magnitude reaches the conversion too. That sign does not
change what `start_recording` reports -- LeRobot refuses it as it opens the dataset
-- so it is a raise the reader owes its own callers rather than a verdict the
surface got wrong, and it is pinned at the reader for that reason.

The conversion is now made inside a `try` handling `OverflowError`, `TypeError`
and `ValueError` -- the same exceptions, for the same reason, as
`requested_rate_mismatch_reason`, the other guard in this module asked before any
domain has classified its rate. A rate beyond the float64 range is reported as the
unreadable layout it is, in either sign.

The statement in #3124's fragment that neither guard needs a `try` is corrected in
the same change: it holds for the guard asked after the domain, not for the one
that reads its rate off disk.
