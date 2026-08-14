### Added: fleet example 05 - work-order ingress mapped onto capability manifests

`examples/fleet/05_work_order_dispatch.py` (epic #2179, PR5): structured work
orders arrive on a JSONL file queue speaking business vocabulary only
(`{order_id, material, operation, qty, from, to, due}`), and the translation
to robot vocabulary is two-stage. Stage one is deterministic - schema
validation plus hard-constraint filtering (site, payload, fixture, zones)
against per-robot, per-site capability manifests, with an explicit NACK back
onto the queue carrying a machine-readable reason (the failing constraint per
robot, with required and actual values) when no feasible robot exists. Stage
two is the agent's territory - choosing among the capable robots and
sequencing multi-step orders - and is floored by `guard_choice`: a pick
outside the feasible set is refused and falls back to the deterministic
selector, so a chooser can never invent a capability. Dispatch goes over the
mesh behind a human-in-the-loop gate (`STRANDS_MESH_HITL_ACTIONS=none` for
CI), the `order_id` is threaded through the signed audit log end to end, and
a structured completion/failure event lands back on the queue. The shared
manifest schema ships as `examples/fleet/capabilities.py`, also consumed by
example 01 (#2180). The smoke test drives the whole core through the
transport seam with the audit log confined to `tmp_path`.
