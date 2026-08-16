### Fixed

`NewtonSimEngine.describe()` advertised `set_timestep` as `"(dt: float) -> dict"`
while the method has always been `set_timestep(timestep: float)`, so a caller
following the discovery surface hit `TypeError: got an unexpected keyword
argument 'dt'`. The advertised string now names `timestep` and its unit, and the
existing rule that every advertised parameter must be a real parameter now
covers every backend that overrides `describe()` - previously only the ABC and
MuJoCo were checked, because the check needed a constructible engine and the
Newton and Isaac runtimes cannot be installed in the same environment.
