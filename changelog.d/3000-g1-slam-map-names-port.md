### Added: a SLAM map name is decided before it is joined onto a path

The neon bundle's SLAM save/load path (`cagataycali/neon-the-g1/tools/g1_slam.py`)
routes an agent-authored map name through one containment check: it joins the
name onto `~/maps`, resolves the result, and tests containment with
`str(path).startswith(str(root))`. A string prefix is satisfied by a *sibling*
directory whose name begins with the root's name, so `../maps-evil/pwn` is
admitted and lands outside the root entirely, while `sub/dir` is admitted inside
it but under a subdirectory the top-level `glob("*.npz")` listing cannot report --
a map saved and then invisible to the caller who saved it.

`g1_slam_map_name_admits` decides the same question before any join, on the name
alone: admitted only when it is a single path component that is not a traversal
token. Containment is then structural rather than tested -- `root / f"{name}.npz"`
for an admitted name cannot be anywhere but directly inside `root`, whatever the
root is and whatever the caller resolves afterwards. Six rules, each reporting
its own id beside the bundle's own `invalid map name` verdict: a missing name, a
bool (`str(True)` is an ordinary stem, so a flag would name a map after itself),
a non-string, the empty string (it makes the file `.npz`, whose stem is `.npz`
rather than the empty string, so the listing reports a name that cannot be loaded
back), a name carrying a path separator or NUL, and `.` or `..`.

Both separators are refused on every platform rather than only the deciding
host's, because a map name travels -- authored by an agent, written on the robot,
listed on an operator's laptop, asserted in CI -- and a rule keyed on `os.sep`
would admit `a\b` on Linux as an ordinary filename and then have it name a
directory the first time the same string reached a Windows path.

`g1_list_slam_map_name_rules` names the whole rule set, the unexpanded map root,
the suffix a writer appends, and the suffixes the listing enumerates in the
bundle's own chaining order. Neither verb touches the filesystem, so the answer
does not move with `$HOME` and is available before the map directory exists.
