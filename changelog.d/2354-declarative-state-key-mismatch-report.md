### Fixed

A declarative `embodiment=` now reports a `state_keys` mismatch with the same registry-checked remedy the generic `robot_state_keys` path gives. An all-missing binding previously returned the observation untouched and reported nothing at all, so the only signal was a downstream error that knows nothing about embodiments; a partly-missing binding warned but asserted a single cause and offered no remedy. Both now name the declared keys, the keys the observation carries, and an embodiment the registry confirms binds that observation. Nothing about the packed state vector changes.
