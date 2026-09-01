### Fixed: an absent optional dependency is reported in `ImportError.name`, not only in prose

`ImportError` carries a keyword-only `name` field so a caller can read which
module could not be imported without parsing the message. The interpreter always
populates it; code that constructs the exception itself populates it only where
it passes `name=`, and no site in the package did -- including both raise sites
of `require_optional` and `require_optionals`, the mechanism every optional
dependency in the package is required to go through and the path 48 call sites
take.

Anything having to tell "an optional extra is absent here" from "this dotted path
does not exist" was therefore left reading the chained exception and, failing
that, matching the reported text. `require_optionals` raises after the `except`
block that probed the modules has exited, so it carried no chain either and the
message was the only thing left.

Measured across the package, 21 of the 28 constructed raise sites were already
reachable through a chained `ModuleNotFoundError` and 7 reported the module in
prose alone. Those 7 now name it at the source: `require_optional` reports the
module it was asked for, `require_optionals` the first missing one in the order
given, and the four hand-rolled probes in `strands_robots/rtps/idl`,
`strands_robots/training/reward.py`,
`strands_robots/policies/vera/server_runner.py` and
`strands_robots/simulation/mujoco/backend.py` report `cyclonedds`,
`lerobot.rewards`, `vera` and `mujoco` respectively.

Two facts are kept rather than traded. `name` is the module that was *requested*,
and the chained `ModuleNotFoundError` is retained, so a module that is present
but whose own imports are not still reports both what was asked for and what the
interpreter could not find. `name=` is metadata rather than text, so every
install instruction is byte-identical to before.
