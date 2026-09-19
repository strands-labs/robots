### Fixed: no operator is asked while the rclpy transport lock is held

`use_ros` consulted the operator gate inside `with _backend.lock:`, so
`interrupt()` blocked for the length of a human decision while holding the
process-wide rclpy executor lock every other caller of the transport has to
take: an unrelated `echo` on the same graph - the odometry read of a second
`RosBridgedRobot`, a scan - waited out the prompt. The gate is now consulted
above the lock, as `use_rtps` and `use_rosbridge` already do, each verb under a
condition mirroring its own required-argument check, so an incomplete call is
still reported without asking an operator and every refusal message is
unchanged.
