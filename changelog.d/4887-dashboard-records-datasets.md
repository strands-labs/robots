### Added: the dashboard records LeRobot v3 datasets

A Record tab in the dashboard picks robots, their cameras, a task, a target and an fps, and records whatever moves
them (sliders, a policy run, the chat) through the package's own `recording.Episode`, so the dataset is the one
`robot.run(record=...)` writes. `POST /api/record/start` begins, `POST /api/record/episode/save|discard` keeps or
drops the current take, `POST /api/record/stop` ends it, `GET /api/record/status` reports frames, achieved fps,
frames dropped against the target fps and the saved episodes, and `GET /api/record/options` lists what can be
recorded. A REC dot in the top bar shows a live take. Several robots record one dataset each (`target/<robot id>`).
The recorder only observes and never uploads.
