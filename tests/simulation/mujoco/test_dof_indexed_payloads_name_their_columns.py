"""A DOF-indexed physics payload names the joint that owns each entry.

``get_jacobian`` and ``get_mass_matrix`` report arrays indexed by MuJoCo's DOF
ordering - a Jacobian's columns, the mass matrix's diagonal - and used to report
only the width. A caller cannot reconstruct that ordering from the width for two
reasons that both arise in ordinary scenes:

* a free or ball joint owns several consecutive DOFs, so ``nv`` exceeds the
  joint count for every floating-base robot;
* ``nv`` spans the whole compiled model, so a scene holding two robots reports
  one width covering both and one robot's columns are an interior slice.

The third DOF-indexed query in the same mixin, ``inverse_dynamics``, already
answers this by naming every generalized force it reports, and that shape is the
one the other two now follow. Without it a caller who pairs a robot's joint
names with the leading columns reads a different robot's Jacobian and is told
nothing, which is what these tests pin.
"""

import inspect

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.physics import PhysicsMixin  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Two hinge joints and a site at the tip. Attached twice, this gives a scene
# whose nv (4) spans both robots while each robot reports two joint names.
ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link1" pos="0 0 0.1">
      <joint name="j1" type="hinge" axis="0 0 1" range="-2 2"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
      <body name="link2" pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
        <site name="tip" pos="0.2 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# A named free joint (six DOFs) plus one hinge: nv is 7 for two joint names.
FLOATING_XML = """
<mujoco model="floater">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <freejoint name="base_free"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1.0"/>
      <body name="arm" pos="0.1 0 0">
        <joint name="swing" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/>
        <site name="tip" pos="0.2 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# MJCF permits a bare <freejoint/>; its DOF columns have no joint name to report.
UNNAMED_JOINT_XML = """
<mujoco model="unnamed">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <freejoint/>
      <geom name="cube" type="box" size="0.1 0.1 0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _write(tmp_path, name, xml):
    path = tmp_path / name
    path.write_text(xml)
    return str(path)


def _json(result):
    """The result's json block, asserting the call succeeded first."""
    assert result["status"] == "success", result["content"][0]["text"]
    return result["content"][-1]["json"]


def _two_arm_sim(tmp_path):
    """A world holding two copies of ARM_XML, so nv spans both."""
    path = _write(tmp_path, "arm.xml", ARM_XML)
    sim = Simulation(tool_name="dof_two_arms")
    created = sim.create_world()
    assert created["status"] == "success", created["content"][0]["text"]
    for name, x in (("arm0", 0.0), ("arm1", 0.6)):
        added = sim.add_robot(name=name, urdf_path=path, position=[x, 0, 0])
        assert added["status"] == "success", added["content"][0]["text"]
    return sim


class TestAJacobianColumnIsAttributableToItsJoint:
    """The regression: the reported columns can be mapped back onto joints."""

    def test_the_second_robots_jacobian_is_not_read_as_the_firsts(self, tmp_path):
        """A two-robot scene reports one width; the payload must say whose columns.

        Fails before: the payload carried only ``nv``, so a caller pairing
        ``robot_joint_names("arm1")`` with the leading columns took arm0's.
        """
        sim = _two_arm_sim(tmp_path)
        try:
            payload = _json(sim.get_jacobian(site_name="arm1/tip"))
            jacp = np.array(payload["jacp"])
            nv = payload["nv"]
            nonzero = [i for i in range(jacp.shape[1]) if np.abs(jacp[:, i]).max() > 1e-12]
            asked_for = sim.robot_joint_names("arm1")
            assert nonzero, "premise: arm1's own joints must move its tip"

            names = payload.get("dof_joint_names")
            if names is None:
                pytest.fail(
                    f"get_jacobian reported a 3x{nv} Jacobian for 'arm1/tip' with only nv={nv}. "
                    f"Its non-zero columns are {nonzero}, which belong to arm1, while "
                    f"robot_joint_names('arm1') is {asked_for} - so a caller pairing those "
                    f"names with columns {list(range(len(asked_for)))} reads arm0's Jacobian "
                    "and is told nothing."
                )
            owners = {names[i] for i in nonzero}
            assert owners == {"arm1/j1", "arm1/j2"}, (
                f"the moving columns {nonzero} are named {sorted(owners)}, not arm1's joints"
            )

            # And the labels really do distinguish the two robots.
            other = _json(sim.get_jacobian(site_name="arm0/tip"))
            other_jacp = np.array(other["jacp"])
            other_nonzero = [i for i in range(other_jacp.shape[1]) if np.abs(other_jacp[:, i]).max() > 1e-12]
            assert {other["dof_joint_names"][i] for i in other_nonzero} == {"arm0/j1", "arm0/j2"}
            assert other_nonzero != nonzero, "premise: the two robots occupy different columns"
        finally:
            sim.destroy()

    def test_a_free_joints_columns_are_each_named(self, tmp_path):
        """nv exceeds the joint count for a floating base; every column is labelled."""
        sim = Simulation(tool_name="dof_floating")
        try:
            created = sim.create_world()
            assert created["status"] == "success", created["content"][0]["text"]
            added = sim.add_robot(name="fb", urdf_path=_write(tmp_path, "fb.xml", FLOATING_XML))
            assert added["status"] == "success", added["content"][0]["text"]

            payload = _json(sim.get_jacobian(site_name="fb/tip"))
            nv = payload["nv"]
            joint_names = sim.robot_joint_names("fb")
            assert nv > len(joint_names), (
                f"premise: a free joint must make nv ({nv}) exceed the joint count ({len(joint_names)})"
            )

            names = payload.get("dof_joint_names")
            if names is None:
                pytest.fail(
                    f"get_jacobian reported 3x{nv} columns for a robot with {len(joint_names)} "
                    f"joints ({joint_names}) and no way to tell which column is which: the free "
                    "joint owns six of them."
                )
            assert len(names) == nv, f"the label list must be nv ({nv}) long, got {len(names)}"
            assert names.count("fb/base_free") == 6, f"a free joint owns six columns, labelled {names}"
            assert names[6] == "fb/swing"

        finally:
            sim.destroy()

    def test_an_unnamed_joints_column_is_reported_as_unnamed(self, tmp_path):
        """A bare ``<freejoint/>`` has no name; the column is kept, not dropped.

        Dropping it would break the index alignment the labels exist to give.
        """
        sim = Simulation(tool_name="dof_unnamed")
        try:
            created = sim.create_world()
            assert created["status"] == "success", created["content"][0]["text"]
            added = sim.add_robot(name="u", urdf_path=_write(tmp_path, "u.xml", UNNAMED_JOINT_XML))
            assert added["status"] == "success", added["content"][0]["text"]

            payload = _json(sim.get_jacobian(body_name="u/base"))
            names = payload["dof_joint_names"]
            assert len(names) == payload["nv"], "the label list stays index-aligned with the columns"
            assert names.count(None) == 6, f"the unnamed free joint's six columns report None, got {names}"
        finally:
            sim.destroy()


class TestTheMassMatrixDiagonalIsAttributableToo:
    """The same rule on the other DOF-indexed payload in the mixin."""

    def test_the_diagonal_entries_are_named(self, tmp_path):
        sim = _two_arm_sim(tmp_path)
        try:
            payload = _json(sim.get_mass_matrix())
            nv = payload["shape"][0]
            diagonal = payload["diagonal"]
            assert len(diagonal) == nv, "premise: the diagonal is DOF-indexed"

            names = payload.get("dof_joint_names")
            if names is None:
                pytest.fail(
                    f"get_mass_matrix reported a {nv}-entry diagonal with only shape={payload['shape']}, "
                    "so a per-joint inertia cannot be read off it in a scene holding two robots."
                )
            assert len(names) == nv
            assert names == ["arm0/j1", "arm0/j2", "arm1/j1", "arm1/j2"]
        finally:
            sim.destroy()


class TestEveryReadOnlyQuerysDofIndexedPayloadIsLabelled:
    """Derived from the mixin, so a query added later is held to the same rule.

    The population is every read-only physics query - a public method named
    ``get_*`` / ``inverse_*`` / ``forward_*`` with no required parameter - plus
    ``get_jacobian``, whose target is optional in the signature but required in
    practice. Any json value that is DOF-indexed (a flat list of nv numbers, or
    a list of rows each nv wide) must be accompanied by ``dof_joint_names``.
    """

    @staticmethod
    def _read_only_queries():
        names = []
        for name, fn in inspect.getmembers(PhysicsMixin, inspect.isfunction):
            if name.startswith("_") or not name.startswith(("get_", "inverse_", "forward_")):
                continue
            params = list(inspect.signature(fn).parameters.values())[1:]  # drop self
            if any(p.default is inspect.Parameter.empty for p in params):
                continue
            names.append(name)
        return sorted(names)

    @staticmethod
    def _dof_indexed_keys(payload, nv):
        """Keys whose value is indexed by DOF."""
        found = []
        for key, value in payload.items():
            if not isinstance(value, list) or not value:
                continue
            if len(value) == nv and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                found.append(key)
            elif all(isinstance(row, list) and len(row) == nv for row in value):
                found.append(key)
        return found

    def test_the_scan_reaches_the_queries_it_grades(self):
        queries = self._read_only_queries()
        assert "get_mass_matrix" in queries and "inverse_dynamics" in queries, queries
        assert len(queries) >= 4, f"the scan must reach the mixin's read-only queries, found {queries}"

    def test_no_read_only_query_reports_dof_indexed_numbers_unlabelled(self, tmp_path):
        sim = _two_arm_sim(tmp_path)
        try:
            nv = sim.mj_model.nv
            assert nv == 4, f"premise: the fixture's nv must span both robots, got {nv}"

            checked, offenders = [], []
            calls = [(name, {}) for name in self._read_only_queries()]
            calls.append(("get_jacobian", {"site_name": "arm1/tip"}))
            for name, kwargs in calls:
                result = getattr(sim, name)(**kwargs)
                if result.get("status") != "success":
                    continue
                blocks = [b["json"] for b in result["content"] if isinstance(b, dict) and "json" in b]
                for payload in blocks:
                    keys = self._dof_indexed_keys(payload, nv)
                    if not keys:
                        continue
                    checked.append((name, tuple(keys)))
                    if "dof_joint_names" not in payload:
                        offenders.append(f"{name} reports DOF-indexed {keys} with keys {sorted(payload)}")

            assert checked, "premise: at least one query must report a DOF-indexed payload"
            assert len(checked) >= 2, f"expected the Jacobian and the mass diagonal, found {checked}"
            assert not offenders, "a DOF-indexed payload does not name its joints: " + "; ".join(offenders)
        finally:
            sim.destroy()


class TestNothingElseChanges:
    """These hold on both trees: the numbers and the existing keys are untouched."""

    def test_the_jacobian_numbers_are_unchanged(self, tmp_path):
        """The reported matrix still equals a direct mj_jacSite for the same pose."""
        sim = _two_arm_sim(tmp_path)
        try:
            payload = _json(sim.get_jacobian(site_name="arm1/tip"))
            model, data = sim.mj_model, sim._world._data
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "arm1/tip")
            mj.mj_jacSite(model, data, jacp, jacr, sid)
            assert np.allclose(np.array(payload["jacp"]), jacp)
            assert np.allclose(np.array(payload["jacr"]), jacr)
            assert payload["nv"] == model.nv
        finally:
            sim.destroy()

    def test_the_mass_matrix_keeps_its_existing_keys(self, tmp_path):
        sim = _two_arm_sim(tmp_path)
        try:
            payload = _json(sim.get_mass_matrix())
            assert {"shape", "rank", "condition_number", "diagonal", "total_mass"} <= set(payload)
            assert payload["shape"] == [4, 4]
            assert payload["rank"] > 0
            assert payload["total_mass"] > 0
        finally:
            sim.destroy()

    def test_inverse_dynamics_still_names_its_forces_by_joint(self, tmp_path):
        """The precedent this change follows, unchanged: a name-keyed mapping."""
        sim = _two_arm_sim(tmp_path)
        try:
            payload = _json(sim.inverse_dynamics())
            forces = payload["qfrc_inverse"]
            assert isinstance(forces, dict)
            assert set(forces) == {"arm0/j1", "arm0/j2", "arm1/j1", "arm1/j2"}
        finally:
            sim.destroy()

    def test_a_single_robot_scene_still_reports_its_own_width(self, tmp_path):
        """The case that always worked: nv is the robot's joint count."""
        sim = Simulation(tool_name="dof_single")
        try:
            created = sim.create_world()
            assert created["status"] == "success", created["content"][0]["text"]
            added = sim.add_robot(name="arm0", urdf_path=_write(tmp_path, "arm.xml", ARM_XML))
            assert added["status"] == "success", added["content"][0]["text"]
            payload = _json(sim.get_jacobian(site_name="arm0/tip"))
            assert payload["nv"] == 2
            assert len(payload["jacp"][0]) == 2
        finally:
            sim.destroy()
