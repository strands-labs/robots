### Fixed: the e-stop resume path measures `lockout_elapsed_s` on the monotonic clock

The duration is no longer reconstructed from two wall-clock reads, so a clock
adjustment while the fleet was held (an RTC-less robot's first NTP sync, a VM
resumed from suspend, an operator correcting the clock) can no longer distort --
or make negative -- the field the safety audit trail keeps to answer how long the
fleet was halted. The envelope's `t` remains a wall-clock instant.
