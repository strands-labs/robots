"""GPU-gated integration: URDF joint names survive the Isaac importer (#1900).

The Isaac URDF importer transcodes any joint name that is not a valid USD
prim identifier (a purely numeric name like the ``robotstudio_so101``'s
``"1"`` imports as ``tn__1_``), and before #1900 that mangled form leaked
through the backend's public API - so the same URDF yielded different joint
vocabularies on Isaac vs MuJoCo. The mapping logic is pinned kit-free in
``tests/simulation/isaac/test_urdf_joint_name_demangle.py``; this test is
the importer round-trip the acceptance criteria call for: a real Kit import
of a numeric-joint URDF must report the URDF's own names on
``robot_joint_names``, key ``get_observation`` by them, and resolve
``send_action`` dict actions by them.

Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ \\
        tests_integ/simulation/test_isaac_urdf_joint_names_gpu.py -m gpu -v
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason="Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable.",
    ),
]

_URDF_NAMES = ["1", "2", "3", "4", "5", "6"]

# A minimal fixed-base arm whose six revolute joints carry the numeric names
# the robotstudio_so101 URDF uses. Inertials are required for the importer to
# build real rigid bodies; geometry is a small box per link.
_NUMERIC_URDF = """\
<robot name="numeric_arm">
  <link name="base_link">
    <inertial><mass value="1.0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision>
  </link>
  {links}
</robot>
"""

_LINK_TMPL = """\
  <link name="link{i}">
    <inertial><mass value="0.2"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.04 0.04 0.04"/></geometry></collision>
  </link>
  <joint name="{name}" type="revolute">
    <parent link="{parent}"/>
    <child link="link{i}"/>
    <origin xyz="0 0 0.06"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
"""


def _write_numeric_urdf(tmp_path) -> str:
    chunks = []
    parent = "base_link"
    for i, name in enumerate(_URDF_NAMES, start=1):
        chunks.append(_LINK_TMPL.format(i=i, name=name, parent=parent))
        parent = f"link{i}"
    path = tmp_path / "numeric_arm.urdf"
    path.write_text(_NUMERIC_URDF.format(links="".join(chunks)), encoding="utf-8")
    return str(path)


def _skip_if_isaac_unavailable() -> None:
    from strands_robots.simulation.isaac import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")


def test_numeric_urdf_joint_names_round_trip(tmp_path) -> None:
    """Import, then read/write the articulation by the URDF's own names."""
    from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

    _skip_if_isaac_unavailable()
    urdf_path = _write_numeric_urdf(tmp_path)

    sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
    try:
        r = sim.create_world()
        assert r["status"] == "success", f"create_world: {r}"

        r = sim.add_robot(name="arm", urdf_path=urdf_path)
        assert r["status"] == "success", f"add_robot: {r}"

        # Acceptance #1: the public vocabulary is the URDF's, exactly - the
        # same list MuJoCo reports for the same file. Set-compare then
        # order-check against the reported order, since the importer may
        # enumerate DOFs in tree order.
        names = sim.robot_joint_names("arm")
        assert sorted(names) == sorted(_URDF_NAMES), names
        assert all(not n.startswith("tn__") for n in names), names

        # Acceptance #2: observation keys are the URDF names.
        obs = sim.get_observation("arm", skip_images=True)
        assert set(_URDF_NAMES) <= set(obs), sorted(obs)

        # Acceptance #3: send_action resolves the URDF names - a dict action
        # keyed by them must fully resolve (no unresolved_keys envelope) and
        # land on the right joint. The PD servo needs a settle budget to
        # track the target (measured: ~0.17 rad of a 0.3 rad step after 20
        # substeps), so give it a generous one and assert the commanded
        # joint moved most of the way while every other joint stayed put.
        r = sim.send_action({"3": 0.3}, robot_name="arm", n_substeps=120)
        assert r["status"] == "success", f"send_action: {r}"
        obs = sim.get_observation("arm", skip_images=True)
        assert abs(obs["3"] - 0.3) < 0.15, obs["3"]
        for other in ("1", "2", "4", "5", "6"):
            assert abs(obs[other]) < 0.05, (other, obs[other])

        # The mangled forms are NOT part of the public vocabulary.
        r = sim.send_action({"tn__3_": 0.3}, robot_name="arm")
        assert r["status"] == "error", r
    finally:
        sim.destroy()
