### Fixed: PD gains survive every scene change, go on only after a run's checks, and are never NaN, infinite or negative

`set_gains` wrote kp and kd into the compiled MuJoCo model, and every scene change rebuilds it, so a holosoma G1
walking at 0.5 m/s fell after a render. The backend now writes them back after every compile. A run applies its
policy's gains only after its last check, so a run refused by its checks leaves the gains it found, and a run that
starts under a latched e-stop on any of its robots applies none and ends at step 0, even if resumed before step one.
`set_gains` and `run()` refuse a NaN, infinite or negative gain (a NaN made MuJoCo reset the whole world every step),
and `set_gains` refuses an actuator with no name, whose write could land on another robot's.
