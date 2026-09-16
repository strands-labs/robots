### Changed: a relative scene path names `~/.strands_robots/scenes`, for both `export_xml` and `load_scene`

`export_xml` resolved a relative destination against the process working
directory, so the most natural agent call - `export_xml {"output_path":
"scene.xml"}` when asked to save the scene - dropped the file wherever the
process was started, the user's git checkout included, while the same agent's
`render` was confined to `~/.strands_robots/renders`. A relative destination (a
bare name or one with directories) is now anchored to the scenes directory
(`STRANDS_ROBOTS_SCENE_ROOT`, default `~/.strands_robots/scenes`); an absolute
path is written as given, as before. Every guard (traversal, symlinked target,
metacharacters) runs on the anchored destination, and the success text reports
the resolved path.

`load_scene` reads that directory too, so the documented round trip - export,
then reload the name you exported - holds in the spelling an agent uses. A
`scene_path` is still read AS GIVEN first, so a relative path that resolves
against the working directory keeps resolving there; the scenes directory is
searched only when nothing is at the caller's path. A scene in neither place is
refused with both directories named.

Scripts that passed a relative `export_xml` destination and read the file back
from the CWD need the returned path or an absolute destination.
