"""Bundled MJCF task objects (articulated containers + receptacles).

The task objects are small, self-contained MJCF files shipped inside the
package - the first-party asset set for contact-rich container tasks
(open a carton, pour proxy contents into a receptacle). They are loaded
through the EXISTING scene APIs rather than a new one:

* ``sim.add_robot(name=..., urdf_path=task_object_path("hinged_carton"))``
  attaches the object into a live scene with its bodies/joints namespaced
  under ``name/``. This is the route that makes an articulated object's
  joints observable by the joint predicates (``joint_above`` /
  ``joint_below`` / ``joint_progress``) - ``get_observation`` is
  robot-scoped, and the scene-scope joint resolution in
  :mod:`strands_robots.simulation.predicates` walks every registered
  entity.
* ``sim.load_scene(task_object_path(...))`` replaces the whole world with
  the object alone, which is only useful for inspecting an asset.

The catalog is the set of ``*.xml`` files in this directory:

    hinged_carton   - carton whose lid opens on a hinge (``cap_hinge``, rad)
    sliding_carton  - carton whose lid slides open (``cap_slide``, m)
    open_tray       - rigid open-top receptacle (no joints)

Proxy contents are NOT part of the assets: spawn them per task as small
spheres via ``add_object`` (see ``examples/17_pour_task.py``), so the count,
size, and start pose stay task-authoring decisions.
"""

from __future__ import annotations

from pathlib import Path

from strands_robots.utils import safe_join

_TASK_OBJECTS_DIR = Path(__file__).resolve().parent


def list_task_objects() -> list[str]:
    """Names of the bundled task objects, sorted.

    Each name resolves through :func:`task_object_path` to an MJCF file
    shipped inside the package.

    Returns:
        Sorted task-object names (without the ``.xml`` suffix).
    """
    return sorted(p.stem for p in _TASK_OBJECTS_DIR.glob("*.xml"))


def task_object_path(name: str) -> str:
    """Absolute path of a bundled task-object MJCF, by catalog name.

    The name is validated against the catalog before it touches the
    filesystem: an unknown name is refused with the valid set (so a typo
    surfaces as an actionable error, not a downstream "file not found"),
    and a path-traversal component (``../``) is refused by
    :func:`strands_robots.utils.safe_join` before the containment check.

    Args:
        name: A catalog name from :func:`list_task_objects`, e.g.
            ``"hinged_carton"``.

    Returns:
        Absolute filesystem path of the MJCF file, as a string (the type
        the scene APIs' ``urdf_path`` / ``scene_path`` parameters take).

    Raises:
        ValueError: If ``name`` is not a non-empty string, escapes the
            asset directory, or names no bundled task object.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"task_object_path: 'name' must be a non-empty str, got {name!r}")
    path = safe_join(_TASK_OBJECTS_DIR, f"{name}.xml")
    if not path.is_file():
        raise ValueError(f"task_object_path: unknown task object {name!r}. Valid: {list_task_objects()}")
    return str(path)
