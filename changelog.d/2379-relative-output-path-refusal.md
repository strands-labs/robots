### Fixed

A confinement refusal from `validate_output_path` now matches the class of path the
caller gave. A relative `output_path` / `output_dir` is resolved against the process
CWD, so under confinement it could only be refused - and the refusal reported that
CWD-absolute path and offered the sink's absolute-path opt-in, a remedy for an input
that was not given. Following it lifted the refusal by disabling confinement, so
`render(output_path="views/front.png")` reported success and wrote the PNG under the
CWD, leaving the sandbox empty. The relative wording now states where the path was
resolved from and quotes the sandbox-anchored destination to pass instead, keeping
the opt-in as a last resort; an absolute destination outside the sandbox keeps its
existing message. Which destinations are accepted is unchanged. `tool_spec.json` also
now publishes the rule that a bare filename is written into the render sandbox, so a
caller reading only the schema no longer has to know and spell the sandbox path.
