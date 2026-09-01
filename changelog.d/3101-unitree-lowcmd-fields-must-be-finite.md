### Fixed

- **drivers/g1, drivers/go2**: every field a ``LowCmd_`` carries to a motor - the
  position target ``q``, the gains ``kp`` / ``kd``, the velocity feed-forward
  ``dq`` and the effort ``tau`` - is now held to the shared
  ``finite_number_error`` domain before it is written. Both builders coerced with
  a bare ``float()``, which accepts ``nan``, ``inf``, the string ``"nan"`` and
  ``True``, so a diverged policy published a fully populated frame with the slot
  enabled and a valid CRC and the motor controller integrated an unrepresentable
  target. Ten of the twelve native drivers already applied this domain to their
  action values; these two applied it only to the constructor's
  ``battery_floor_pct`` and ``run_policy``'s ``duration``. A non-finite action
  now takes the control loop's existing ``exit_reason="policy"`` lane instead of
  reaching the wire. NumPy scalars are still accepted.
