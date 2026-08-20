"""The bundled task-object catalog resolves names to package MJCF, and nothing else.

:mod:`strands_robots.simulation.task_objects` ships the articulated-container
assets (hinged/sliding carton + open tray) inside the package. These tests pin
the catalog surface: every listed name resolves to an existing MJCF file, an
unknown name is refused with the valid set (not a downstream file-not-found),
and a traversal component cannot escape the asset directory. The assets
themselves are checked for well-formedness here with stdlib XML only - loading
them into a real physics scene is
``tests/simulation/mujoco/test_pour_task_smoke.py``'s job.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from strands_robots.simulation.task_objects import list_task_objects, task_object_path


class TestCatalog:
    def test_ships_the_container_task_set(self):
        assert {"hinged_carton", "sliding_carton", "open_tray"} <= set(list_task_objects())

    def test_every_listed_object_resolves(self):
        for name in list_task_objects():
            path = Path(task_object_path(name))
            assert path.is_file()
            assert path.suffix == ".xml"

    def test_unknown_name_is_refused_with_the_valid_set(self):
        with pytest.raises(ValueError, match="hinged_carton"):
            task_object_path("no_such_object")

    def test_traversal_is_refused(self):
        with pytest.raises(ValueError):
            task_object_path("../robots")

    @pytest.mark.parametrize("bad", ["", None, 7])
    def test_non_name_is_refused(self, bad):
        with pytest.raises(ValueError, match="non-empty str"):
            task_object_path(bad)


class TestAssetShape:
    """Structural facts the pour tasks rely on, checkable without MuJoCo."""

    @pytest.mark.parametrize(
        ("name", "joint_name", "joint_type"),
        [("hinged_carton", "cap_hinge", "hinge"), ("sliding_carton", "cap_slide", "slide")],
    )
    def test_cartons_articulate_their_cap(self, name, joint_name, joint_type):
        root = ET.parse(task_object_path(name)).getroot()
        assert root.tag == "mujoco"
        joints = root.findall(".//joint")
        assert [(j.get("name"), j.get("type")) for j in joints] == [(joint_name, joint_type)]
        # A range is what makes "opened" a bounded, scoreable quantity.
        assert joints[0].get("range") is not None

    def test_tray_is_rigid(self):
        root = ET.parse(task_object_path("open_tray")).getroot()
        assert root.tag == "mujoco"
        assert root.findall(".//joint") == []
        assert root.find(".//body[@name='tray']") is not None
