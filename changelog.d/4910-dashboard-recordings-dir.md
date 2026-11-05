### Fixed: the dashboard's Record tab writes datasets only inside its recordings directory

`POST /api/record/start` used to resolve any `target` the page sent and create or resume a LeRobot dataset there. An
aborted take deleted that directory. Now a target must be a name or path inside `STRANDS_ROBOTS_RECORDINGS` (default
`~/strands-robots/recordings`), after `..` and symlinks are resolved, or an `hf://owner/name`. Anything else is
refused with 403. The page's default target moved from `/tmp/strands-recording-<time>` into that directory. Library
`run(record=...)` is unchanged.
