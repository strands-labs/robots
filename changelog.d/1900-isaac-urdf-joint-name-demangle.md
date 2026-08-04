### Fixed: Isaac reports a URDF's own joint names, not USD's mangled forms

The Isaac URDF importer transcodes any joint name that is not a valid USD
prim identifier - the `robotstudio_so101` URDF names its joints literally
`"1"`..`"6"`, and since a USD identifier cannot start with a digit they
import as `tn__1_`..`tn__6_` - and that mangled form leaked through every
public surface keyed by joint name: `robot_joint_names`, `get_observation`
keys, and `send_action` dict resolution. Isaac was self-consistent, but the
same URDF on MuJoCo keeps `"1"`..`"6"`, so cross-backend consumers (e.g. a
cuRobo planner built from the same URDF fed `scene.joint_names` from the
Isaac sim) mismatched joint vocabularies. `add_robot(urdf_path=...)` now
re-reads the URDF's movable joint names with stdlib XML and deterministically
maps the importer's `dof_names` back onto them (bootstring `tn__` transcoding
and legacy `TfMakeValidIdentifier` substitution both recognised; ambiguous
decodes keep the USD name), so every backend reports the same vocabulary for
the same URDF. The `usd_name -> urdf_name` map is recorded on the robot's
bookkeeping entry for diagnostics. (#1900)
