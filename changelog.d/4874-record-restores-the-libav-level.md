### Fixed: after `record=`, libav logs where you had it, not to the terminal for the rest of the process

LeRobot's encoder replaces PyAV's libav callback with libav's own stderr printer after every video, and `record=`
only reset that printer's level to INFO when it finished, so for the rest of the process every libav notice reached
the terminal: one `No accelerated colorspace conversion` line per decoded frame when reading back through PyAV on
macOS, and an `av.logging.set_level()` that routed libav into Python logging was bypassed. Each write now hands PyAV
back the level it found (`None` by default, which drops libav's log), so a process that recorded logs like one that
never did, and notebook 02 no longer quiets libav again after each `record=`. A caller who had installed libav's own
printer with `av.logging.restore_default_callback()` gets PyAV's callback back as well: PyAV cannot say which
callback is installed.
