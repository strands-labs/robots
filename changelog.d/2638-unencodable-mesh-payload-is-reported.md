### Fixed: a mesh payload that can never reach the wire is reported, not dropped at DEBUG

`Mesh.publish_safety_event` writes one event to two sinks, and on a payload the
JSON encoder refuses the two disagreed. The audit half recorded a
`sig="SERIALISE_FAILED"` poison record and logged at ERROR; the wire half
published nothing and said so only at DEBUG. At the default log level an operator
therefore had a forensic trail asserting a `critical` collision event had been
raised, with no peer having received it.

Both encode sites absorbed the encode into the handler that absorbs a wire
failure, so a permanent failure and a transient one were reported identically.
`MeshTransport.put` already scoped its tolerance to a transient failure - a closed
session, a dropped broker, a socket-level write - which the caller's next tick
retries. A payload the encoder refuses is not transient: it fails the same way
forever, and 8 of 9 realistic payload shapes are affected, including a
`np.float32` sensor reading and an `np.zeros(3)` pose vector.

The encode now happens outside that handler at both sites, and the permanent case
is reported at ERROR once per topic through one shared reporter, so a bridge's two
legs word it identically and a 50 Hz publisher with a broken payload builder emits
one line rather than one per tick. The transient path is unchanged. Hoisting the
encode above the `awscrt` import on the MQTT leg also separates an absent
`[mesh-iot]` extra from a bad payload, and `BridgeTransport.put`'s docstring no
longer cites a serialisation error as an example of something that propagates
through it - measured with the real legs, it does not.
