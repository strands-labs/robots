### Fixed

A sandbox confinement refusal now names the environment variable that lifts it
instead of the pattern `*_ALLOW_ABS`. Artifact sinks confine an LLM-supplied
output path to a sandbox root, and each sink has its own opt-in spelling
(`STRANDS_ROBOTS_RENDER_ALLOW_ABS` for `render(output_path=...)`,
`STRANDS_ROBOTS_VIDEO_ALLOW_ABS` for `run_policy(video=...)` and
`start_cameras_recording(output_dir=...)`), but the refusal reported the glob, so
acting on the one message whose job is to state the next step meant grepping the
package for the name. The name was available at every call site and discarded:
only the resolved boolean reached `validate_output_path`. It is now threaded
through as `allow_abs_env` and quoted, along with the alternative that needs no
variable at all (a path under the sandbox); `video_sandbox_args` returns the
spelling beside the flag it derives from it, so the value read and the name
reported cannot drift apart. The confinement decision is unchanged, and refusals
the opt-in cannot lift (`..` traversal, shell metacharacters, backslash
separators, a symlinked target) still do not advertise it.
