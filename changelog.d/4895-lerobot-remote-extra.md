### Fixed: a LeRobot without its remote extra is refused before the first step, with the pinned install line

LeRobot imports `msgpack` at the first encode, so a LeRobot installed without `[remote]` used to pass
the import check and then fail inside LeRobot with `pip install 'lerobot[remote]'`, a PyPI line whose release has
no remote inference. Attaching a LeRobot policy now checks `msgpack` and `zenoh` first and names the one pinned
LeRobot line. The acceptance checks skip by name, instead of failing, when LeRobot's package refuses
to import for a missing extra (`lerobot.datasets` without `datasets`).
