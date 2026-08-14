### Fixed: `export_xml(output_path=...)` validates its LLM-supplied destination

`export_xml` is one of the MuJoCo simulation's agent-callable actions, so its
`output_path` arrives from a tool call exactly as `render`'s does, but it wrote
that path straight to `open(output_path, "w")`. A `..` segment escaped the
requested directory, a symlinked target was followed and overwrote whatever it
pointed at, and shell metacharacters and backslash separators were accepted,
each reporting `status=success`. `strands_robots.simulation.safe_output` exists
for this class of sink and its module docstring enumerates them; `export_xml`
was absent from that list in prose and in behaviour.

The write also sat outside the method's `try`, so an `OSError` from the
caller's path escaped the result envelope entirely: a missing parent directory
raised `FileNotFoundError` and a directory destination raised
`IsADirectoryError`, past a method documented to return a result dict.

`output_path` now routes through the shared `validate_output_path` guard before
writing, the write is atomic so a crash mid-export cannot truncate an existing
file at the destination, and an `OSError` at the sink is reported through the
envelope (a missing parent is created; the message names the destination rather
than the internal temp filename). Confinement is guards-only: an absolute
destination is still accepted, preserving this sink's historic contract, because
`safe_output` documents `render`'s sandbox as applying to a "newer,
sandboxed-by-design feature". The success text now reports the resolved path
rather than the raw argument, which could name a file that was not written.

The `output_path` description in the agent tool schema said "Trajectory/video
export path", naming neither `render` nor `export_xml`; it now states each
sink's destination and the guards that apply to all of them.
