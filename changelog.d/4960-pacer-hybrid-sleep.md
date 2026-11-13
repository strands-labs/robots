### Fixed: a 100 Hz run on hardware no longer misses deadlines because a sleep woke late

Between steps the pacer slept the whole wait in one `sleep`, and the operating system often wakes a sleeper a
millisecond or two late (more on macOS, with occasional spikes). With a few ms of work per step at 100 Hz that
late wake alone pushed writes past their deadline: about 1 in 500 ticks on an M-series Mac, up to 9 ms late. The
pacer now sleeps until 2 ms before the target and spins the rest, yielding to other threads each turn and
checking the e-stop latch, so an e-stop still ends the wait at once. Only the host's monotonic clock spins; a
robot built with its own `clock=` waits exactly as before.
