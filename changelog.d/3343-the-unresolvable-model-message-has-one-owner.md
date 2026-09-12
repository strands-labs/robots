### Changed: the unresolvable-model message has one owner

`MuJoCoSimEngine._unknown_model_msg`'s three-way diagnosis - a typo (with close
matches over the sim-loadable registry), a hardware-only registry entry, or an
asset that is simply not downloaded - moved to
`strands_robots.simulation.base.unknown_model_msg` now that the Isaac backend
resolves names too. The MuJoCo method delegates and its text is byte-identical; a
second inline copy is how two backends come to diagnose one registry differently.
The one backend-specific sentence, the discovery hint, is a parameter.
