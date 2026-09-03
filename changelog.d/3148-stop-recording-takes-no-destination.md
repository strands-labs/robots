### Fixed: `stop_recording` accepts no destination it will not write

`Simulation.stop_recording` took an `output_path` in its first positional slot
and discarded it, documented as an unused legacy argument. The dataset root is
chosen once, at `start_recording(root=...)`, and the recorder has been writing
there for the whole episode, so by the time this call runs there is nothing left
to redirect: a destination here can only be dropped.

It was reachable from an agent. `output_path` is a published field of the
simulation tool schema, and the dispatcher's unknown-parameter refusal reads the
method signature -- so the field bound here, the dataset was finalized at the
recorder's own root, and the call reported `status="success"` about a path that
stayed empty. A positional python call was swallowed the same way, answering
"Was not recording." to `stop_recording(some_path)`.

Nothing else in the tree wanted the parameter: `describe()` already advertised
the method as `(push_to_hub=..., bucket=..., run_id=...)`, and the schema field's
own list of sinks names `render`, `export_xml` and the rollout drivers rather
than this one. It is removed, so both entry points now refuse a destination --
the dispatcher by name (`Unknown parameter 'output_path' for action
'stop_recording'. Valid: ['bucket', 'push_to_hub', 'run_id']`), python with a
`TypeError` -- and the refusal precedes dispatch, leaving an open recording
intact rather than finalized elsewhere under a success. The API table in
`docs/simulation/overview.md`, which advertised `output_path` as the method's
only parameter, is corrected to the three keyword arguments it does take.

Honouring the path instead of removing it would have been new behaviour rather
than a fix: the parquet and per-camera MP4s are already on disk under the
recorder's root, so a destination given at stop time could only mean a copy or a
move.
