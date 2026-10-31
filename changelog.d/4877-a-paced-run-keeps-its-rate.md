### Fixed: a paced run keeps its rate when its sleeps wake late, and a recording's setup is no deadline miss

A paced run (hardware, or `realtime=True`) timed a write it held to one period after the previous one from when
its sleep woke, so every held tick added the host's oversleep to the period: at 50 Hz, sleeps waking 1.5 ms late
wrote at 46.5 Hz with no deadline miss counted, each write acting on an observation about a period old. A held
write now counts as written on its floor: on hardware with `deadline_misses == 0`, `RunResult.seconds` is
`steps / hz_target` to within one period, plus any time a hold waited past its period; two writes may come closer
than a period by one oversleep, however long; and a tick a late wake starts so late that its write lands past the
next tick's start is a miss, so a heavily loaded host now counts misses (and `on_miss="fault"` ends the run) where
it used to slow down. `record=` opens its dataset before the first tick and before a policy's gains go on, so its
setup no longer makes tick 2 late (`on_miss="fault"` ended a process's first recording there), and a dataset the
run cannot append to is refused before tick 1 writes.
