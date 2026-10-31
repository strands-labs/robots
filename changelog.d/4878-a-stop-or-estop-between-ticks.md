### Fixed: a stop or e-stop between ticks ends a paced run before another write; `on_disconnect` lands at the fault

A `stop()` that landed while a paced run (hardware, or `realtime=True`) waited for its next tick was seen only after
one more observe, inference and write. An e-stop in that wait, cleared before the next tick wrote, let that tick
write a target decided after the press, and so did an e-stop pressed and cleared during a policy call. Both loops now
check the e-stop, then `stop_event`, again after the wait, and before its write a tick checks whether its robot's
e-stop has been pressed since the run's first tick, resumed or not, and if so writes nothing. A run that ends in a
fault puts the body in its `on_disconnect` state as soon as the fault is seen, not up to a period later, so an e-stop
pressed in that period no longer keeps a control whose `on_disconnect` is not `hold` from being sent its target.
