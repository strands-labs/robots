### Fixed

- **tools/lerobot_teleoperate**: a teleop session whose process exists but cannot be inspected (`psutil.AccessDenied`, e.g. a session started under `sudo` for serial-port access and later listed as the invoking user) is no longer erased from the session store. The prune is written back to disk and that store holds the only copy of the detached process's pid, so dropping the record left the session unlisted, unstoppable, and still driving the arm. A process reaped mid-probe (`psutil.NoSuchProcess`) is still pruned, and a retained record is now reported at `WARNING`.
