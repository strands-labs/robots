### Internal: the LeRobot v3 writer types its episode against a Protocol instead of importing it

`recording/lerobot_v3.py` read five fields off the `Episode` it serves and imported the class to name them, while
`episode.py` imports the writer at record time -- a two-module cycle. The writer now declares a `Recording` protocol
with those fields and drops the `episode` import, so the one edge is `episode -> lerobot_v3`; an import-linter
contract pins it. No public surface changes; LeRobot stays lazily imported.
