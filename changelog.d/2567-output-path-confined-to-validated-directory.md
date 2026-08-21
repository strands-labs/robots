### Fixed: a caller-named capture or pose file cannot be written outside the directory that was validated

`validate_save_path` guards the directory a tool writes into, but the file was
composed afterwards from separate caller-supplied parameters and joined onto it,
and `os.path.join` walks back out of its first argument whenever the second asks
it to. So the `..` traversal that guard refuses in `save_path` was reachable
through `lerobot_camera`'s `filename` or `format`, or `pose_tool`'s `robot_id` --
the halves of the path nothing checked -- and the write reported success at
whatever location it had reached.

A new `resolve_output_path` helper beside `validate_save_path` resolves the
composed name and refuses one that does not land inside the resolved directory,
which also means a symlink inside it cannot be used to step outside. It is a
containment check rather than a character allowlist, so every name that worked
before still works: an unlisted extension such as `tiff` stays honored, per the
decision recorded in #2559, and so does a filename containing a space.
