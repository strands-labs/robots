### Fixed: the fleet evacuation incident report attests the records it shows

`examples/fleet/04_emergency_evacuation.py` scoped its audit read to the run
(`read_audit_log(since=run_start - 1.0)`) and then called
`verify_audit_integrity()` with no argument, which re-reads the whole log. For
an example that log is the developer's real
`~/.strands_robots/mesh_audit.jsonl`, because examples deliberately do not
redirect `STRANDS_MESH_AUDIT_DIR`, so the report's header and its timeline
described different record sets. Measured with 4000 records of prior history in
the log and 5 events from the run, the rendered report read
`Audit integrity: ok=False (signed=5/4005)` above a five-row table. The `ok`
value is the worse half: history written before `STRANDS_MESH_AUDIT_PSK` was
configured is unsigned, and an unsigned record is a forgery by definition once a
PSK is set at verification time, so a completely successful evacuation reported
tamper evidence. Scoped to the records shown, the same run reports
`ok=True (signed=5/5)`.

`build_incident_report` now pairs them itself - `integrity` defaults to a
verdict over the records it was handed - so the unscoped call is gone from the
file rather than merely given an argument, and a caller cannot reintroduce the
mismatch by omission. The explicit two-argument form still works.
