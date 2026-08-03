"""The Isaac backend speaks the URDF's joint names, not USD's mangled forms.

Isaac Sim's URDF importer transcodes any joint name that is not a valid USD
prim identifier: the ``robotstudio_so101`` URDF names its joints literally
``"1"``..``"6"``, and since a USD identifier cannot start with a digit they
import as ``tn__1_``..``tn__6_``. Before #1900 that mangled form leaked
through every public surface keyed by joint name - ``robot_joint_names``,
``get_observation`` keys, ``send_action`` resolution - so the same URDF
yielded different joint vocabularies on Isaac vs MuJoCo, and a cross-backend
consumer (e.g. a cuRobo planner built from the same URDF) mismatched the sim.

These tests need no Isaac Kit install:

  * the mapping logic (:mod:`strands_robots.simulation.isaac.joint_names`)
    is pure stdlib and tested directly;
  * the loader boundary is exercised end-to-end through ``add_robot`` with a
    fake ``isaacsim`` module tree (the pattern of
    ``tests/simulation/isaac/test_backend_parity.py``), pinning that the
    translation happens where the names enter the backend, once, and that
    every public surface downstream agrees.

The importer round-trip on a real Kit is covered by the GPU-gated
``tests_integ/simulation/test_isaac_urdf_joint_names_gpu.py``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from strands_robots.simulation.isaac.joint_names import demangle_usd_joint_names, urdf_joint_names
from strands_robots.simulation.isaac.simulation import IsaacSimulation

_NUMERIC_URDF = """\
<robot name="so101_numeric">
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <link name="l3"/>
  <link name="l4"/>
  <link name="l5"/>
  <link name="l6"/>
  <joint name="mount" type="fixed">
    <parent link="base"/>
    <child link="l1"/>
  </joint>
  <joint name="1" type="revolute">
    <parent link="l1"/>
    <child link="l2"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
  </joint>
  <joint name="2" type="revolute">
    <parent link="l2"/>
    <child link="l3"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
  </joint>
  <joint name="3" type="revolute">
    <parent link="l3"/>
    <child link="l4"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
  </joint>
  <joint name="4" type="revolute">
    <parent link="l4"/>
    <child link="l5"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
  </joint>
  <joint name="5" type="continuous">
    <parent link="l5"/>
    <child link="l6"/>
  </joint>
  <joint name="6" type="prismatic">
    <parent link="l6"/>
    <child link="base"/>
    <limit lower="0" upper="0.04" effort="10" velocity="1"/>
  </joint>
</robot>
"""

_MANGLED = [f"tn__{i}_" for i in ("1", "2", "3", "4", "5", "6")]
_URDF_NAMES = ["1", "2", "3", "4", "5", "6"]


@pytest.fixture
def numeric_urdf(tmp_path: Path) -> Path:
    path = tmp_path / "so101_numeric.urdf"
    path.write_text(_NUMERIC_URDF, encoding="utf-8")
    return path


class TestUrdfJointNames:
    """:func:`urdf_joint_names` - the kit-free URDF side of the map."""

    def test_movable_joints_in_file_order(self, numeric_urdf: Path) -> None:
        assert urdf_joint_names(str(numeric_urdf)) == _URDF_NAMES

    def test_fixed_joints_are_excluded(self, numeric_urdf: Path) -> None:
        assert "mount" not in urdf_joint_names(str(numeric_urdf))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, OSError)):
            urdf_joint_names(str(tmp_path / "nope.urdf"))

    def test_malformed_xml_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.urdf"
        bad.write_text("<robot><joint", encoding="utf-8")
        with pytest.raises(ValueError, match="malformed"):
            urdf_joint_names(str(bad))

    def test_non_robot_root_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "scene.urdf"
        bad.write_text("<mujoco/>", encoding="utf-8")
        with pytest.raises(ValueError, match="<robot>"):
            urdf_joint_names(str(bad))

    def test_nameless_joint_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "anon.urdf"
        bad.write_text(
            '<robot name="r"><link name="a"/><link name="b"/>'
            '<joint type="revolute"><parent link="a"/><child link="b"/></joint></robot>',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="without name"):
            urdf_joint_names(str(bad))


class TestDemangleUsdJointNames:
    """:func:`demangle_usd_joint_names` - the mapping logic, no kit needed."""

    def test_issue_case_numeric_names_translate(self) -> None:
        public, mapping = demangle_usd_joint_names(_MANGLED, _URDF_NAMES)
        assert public == _URDF_NAMES
        assert mapping == dict(zip(_MANGLED, _URDF_NAMES))

    def test_valid_names_pass_through_untouched(self) -> None:
        names = ["shoulder", "elbow", "gripper"]
        public, mapping = demangle_usd_joint_names(names, names)
        assert public == names
        assert mapping == {}

    def test_mixed_vocabulary_translates_only_the_mangled(self) -> None:
        public, mapping = demangle_usd_joint_names(
            ["shoulder", "tn__1_", "gripper"],
            ["shoulder", "1", "gripper"],
        )
        assert public == ["shoulder", "1", "gripper"]
        assert mapping == {"tn__1_": "1"}

    def test_dof_order_wins_over_urdf_file_order(self) -> None:
        # The articulation may enumerate DOFs in tree order, not file order;
        # positional reads/writes key off dof order, so public names must too.
        public, _ = demangle_usd_joint_names(["tn__2_", "tn__1_"], ["1", "2"])
        assert public == ["2", "1"]

    def test_legacy_underscore_substitution_translates(self) -> None:
        # Older importers apply TfMakeValidIdentifier: per-character "_"
        # substitution rather than bootstring transcoding.
        public, mapping = demangle_usd_joint_names(["joint_1"], ["joint 1"])
        assert public == ["joint 1"]
        assert mapping == {"joint_1": "joint 1"}

    def test_non_basic_characters_match_on_the_stem(self) -> None:
        # A name with characters outside the identifier alphabet transcodes
        # with a non-empty bootstring suffix; the match keys on the stem.
        public, mapping = demangle_usd_joint_names(["tn__wristroll_qA2f"], ["wrist-roll"])
        assert public == ["wrist-roll"]
        assert mapping == {"tn__wristroll_qA2f": "wrist-roll"}

    def test_ambiguous_decode_keeps_the_usd_name(self) -> None:
        # Both URDF names substitute to "joint_1" under the legacy mangle;
        # neither may claim it, so the USD name is kept (self-consistent).
        public, mapping = demangle_usd_joint_names(["joint_1"], ["joint 1", "joint-1"])
        assert public == ["joint_1"]
        assert mapping == {}

    def test_unknown_dof_name_is_kept(self) -> None:
        public, mapping = demangle_usd_joint_names(["mystery"], _URDF_NAMES)
        assert public == ["mystery"]
        assert mapping == {}

    def test_verbatim_claim_blocks_a_colliding_decode(self) -> None:
        # "tn__1_" would decode to "1", but "1" is already reported verbatim
        # by another DOF; a translation would key two DOFs by one name, so
        # the URDF joint is not a candidate and the USD name is kept.
        dof = ["1", "tn__1_"]
        public, mapping = demangle_usd_joint_names(dof, ["1"])
        assert public == dof
        assert mapping == {}

    def test_empty_inputs(self) -> None:
        assert demangle_usd_joint_names([], []) == ([], {})
        assert demangle_usd_joint_names([], ["1"]) == ([], {})


# ---------------------------------------------------------------------------
# Loader-boundary tests: add_robot(urdf_path=...) through a fake isaacsim tree.
# ---------------------------------------------------------------------------


class _FakeArticulationAction:
    """Stand-in for ``isaacsim.core.utils.types.ArticulationAction``."""

    def __init__(self, joint_positions=None, joint_indices=None):
        self.joint_positions = joint_positions
        self.joint_indices = joint_indices


class _FakeArticulation:
    """Articulation whose importer-reported DOF names are USD-mangled."""

    def __init__(self, prim_path: str, name: str):
        self.prim_path = prim_path
        self.name = name
        self.dof_names = list(_MANGLED)
        self.last_action = None

    def initialize(self) -> None:
        return None

    def get_joint_positions(self):
        return np.arange(len(self.dof_names), dtype=np.float32) * 0.1

    def apply_action(self, action) -> None:
        self.last_action = action

    def set_world_pose(self, position=None) -> None:
        return None


class _FakeURDFImporterConfig:
    """Accepts the attribute writes ``_load_urdf_robot`` probes for."""


class _FakeURDFImporter:
    """Isaac Sim 6.0-shaped importer: ``import_urdf()`` returns a USD path."""

    def __init__(self, config=None):
        self.config = config

    def import_urdf(self) -> str:
        return "/tmp/converted_robot.usd"


class _FakeWorld:
    def step(self, render: bool = False) -> None:  # noqa: ARG002 - signature parity
        return None


@pytest.fixture
def fake_isaacsim(monkeypatch):
    """Fake ``isaacsim`` module tree covering every import ``add_robot``'s URDF
    branch performs: articulation class, 6.0 URDF importer, stage reference,
    and the ``ArticulationAction`` that ``send_action`` builds."""
    names = (
        "isaacsim",
        "isaacsim.core",
        "isaacsim.core.api",
        "isaacsim.core.api.articulations",
        "isaacsim.core.utils",
        "isaacsim.core.utils.stage",
        "isaacsim.core.utils.types",
        "isaacsim.asset",
        "isaacsim.asset.importer",
        "isaacsim.asset.importer.urdf",
    )
    mods = {}
    for name in names:
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        mods[name] = mod
    mods["isaacsim.core.api.articulations"].Articulation = _FakeArticulation
    mods["isaacsim.core.utils.stage"].add_reference_to_stage = lambda usd_path, prim_path: None
    mods["isaacsim.core.utils.types"].ArticulationAction = _FakeArticulationAction
    mods["isaacsim.asset.importer.urdf"].URDFImporter = _FakeURDFImporter
    mods["isaacsim.asset.importer.urdf"].URDFImporterConfig = _FakeURDFImporterConfig
    # Wire submodule attributes so dotted imports resolve.
    mods["isaacsim"].core = mods["isaacsim.core"]
    mods["isaacsim"].asset = mods["isaacsim.asset"]
    mods["isaacsim.core"].api = mods["isaacsim.core.api"]
    mods["isaacsim.core"].utils = mods["isaacsim.core.utils"]
    mods["isaacsim.core.api"].articulations = mods["isaacsim.core.api.articulations"]
    mods["isaacsim.core.utils"].stage = mods["isaacsim.core.utils.stage"]
    mods["isaacsim.core.utils"].types = mods["isaacsim.core.utils.types"]
    mods["isaacsim.asset"].importer = mods["isaacsim.asset.importer"]
    mods["isaacsim.asset.importer"].urdf = mods["isaacsim.asset.importer.urdf"]
    return mods


def _sim_with_numeric_robot(numeric_urdf: Path) -> IsaacSimulation:
    sim = IsaacSimulation()
    sim._world = _FakeWorld()
    sim._world_created = True
    result = sim.add_robot(name="arm", urdf_path=str(numeric_urdf))
    assert result["status"] == "success", result
    return sim


class TestPublicApiSpeaksUrdfNames:
    """Acceptance criteria of #1900, at the backend's public surfaces."""

    def test_add_robot_reports_urdf_joint_names(self, fake_isaacsim, numeric_urdf: Path) -> None:
        sim = IsaacSimulation()
        sim._world = _FakeWorld()
        sim._world_created = True
        result = sim.add_robot(name="arm", urdf_path=str(numeric_urdf))
        assert result["status"] == "success", result
        payload = result["content"][0]["json"]
        assert payload["joint_names"] == _URDF_NAMES

    def test_robot_joint_names_match_the_urdf(self, fake_isaacsim, numeric_urdf: Path) -> None:
        sim = _sim_with_numeric_robot(numeric_urdf)
        assert sim.robot_joint_names("arm") == _URDF_NAMES

    def test_observation_keys_are_urdf_names(self, fake_isaacsim, numeric_urdf: Path) -> None:
        sim = _sim_with_numeric_robot(numeric_urdf)
        obs = sim.get_observation("arm")
        assert list(obs) == _URDF_NAMES
        assert obs["3"] == pytest.approx(0.2)

    def test_send_action_resolves_urdf_names(self, fake_isaacsim, numeric_urdf: Path) -> None:
        sim = _sim_with_numeric_robot(numeric_urdf)
        result = sim.send_action({"3": 0.5}, robot_name="arm")
        assert result["status"] == "success", result
        act = sim._robots["arm"].articulation.last_action
        assert list(np.asarray(act.joint_indices)) == [2]
        assert list(np.asarray(act.joint_positions)) == pytest.approx([0.5])

    def test_send_action_refuses_the_mangled_names(self, fake_isaacsim, numeric_urdf: Path) -> None:
        # The USD-internal form is not part of the public vocabulary: a
        # consumer keying actions by it (the pre-fix Isaac-only shape) gets
        # the structured unresolved-keys envelope, same as on MuJoCo.
        sim = _sim_with_numeric_robot(numeric_urdf)
        result = sim.send_action({"tn__3_": 0.5}, robot_name="arm")
        assert result["status"] == "error"
        assert "tn__3_" in result["content"][0]["text"]

    def test_usd_to_urdf_map_recorded_for_diagnostics(self, fake_isaacsim, numeric_urdf: Path) -> None:
        sim = _sim_with_numeric_robot(numeric_urdf)
        assert sim._robots["arm"].usd_to_urdf_joint_names == dict(zip(_MANGLED, _URDF_NAMES))

    def test_already_valid_urdf_names_are_unchanged(self, fake_isaacsim, tmp_path: Path, monkeypatch) -> None:
        # A URDF whose names are valid identifiers is never transcoded:
        # the importer reports them verbatim and no translation happens.
        urdf = tmp_path / "named.urdf"
        urdf.write_text(
            '<robot name="r"><link name="a"/><link name="b"/>'
            '<joint name="shoulder" type="revolute"><parent link="a"/><child link="b"/>'
            '<limit lower="-1" upper="1" effort="10" velocity="1"/></joint></robot>',
            encoding="utf-8",
        )
        monkeypatch.setattr(_FakeArticulation, "__init__", _valid_name_init)
        sim = IsaacSimulation()
        sim._world = _FakeWorld()
        sim._world_created = True
        result = sim.add_robot(name="arm", urdf_path=str(urdf))
        assert result["status"] == "success", result
        assert sim.robot_joint_names("arm") == ["shoulder"]
        assert sim._robots["arm"].usd_to_urdf_joint_names == {}


def _valid_name_init(self, prim_path: str, name: str) -> None:
    self.prim_path = prim_path
    self.name = name
    self.dof_names = ["shoulder"]
    self.last_action = None
