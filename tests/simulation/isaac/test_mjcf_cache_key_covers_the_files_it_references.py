"""The USD cache key covers the closure the MJCF reads, not one directory of it.

``convert_mjcf_to_usd`` caches by content, and ``_asset_digest``'s own docstring
states the reason: keying on the named file alone would "hand back a stale USD
after any change that did not touch the one file named - which is most of them".

It walked ``dirname(mjcf_path)`` to cover that, which is the right idea and the
wrong file set. An MJCF reaches outside its own directory, and the shipped
registry's nested layouts do it by design (AGENTS.md > Registry conventions):

* ``lekiwi`` resolves to ``lekiwi/lekiwi.xml``, which ``<include>``s
  ``../so_arm100/so_arm100.xml`` - the entire arm's joints and geoms.
* ``asimov_v0`` resolves to ``xmls/asimov.xml``, which declares
  ``meshdir="../assets/meshes"`` - every mesh PhysX simulates.
* ``jvrc``, ``aliengo``, ``unitree_a1`` and ``reachy_mini`` share the shape.

So an asset re-download or an upstream update that changed ``so_arm100.xml``, or
any mesh under ``../assets/``, produced the SAME key while the entry directory
stayed byte-identical - and ``convert_mjcf_to_usd`` returned the USD built from
the old description under ``status: success``, with nothing anywhere to detect
it. The wrong ``so_arm100.xml`` is a wrong joint vocabulary, which is precisely
the joint-name parity this module exists to provide.

The collision holds in the other direction too, and that half is the one a
reader is least likely to expect: two robots whose entry directories are
identical but whose siblings differ shared one cache entry, so the second one
converted would be served the first one's USD.

These pins are filesystem-level and need no Isaac Sim: the digest is a pure
function of files on disk, so the defect and the fix are both observable by
writing two trees and comparing keys. The reviewer's own recipe -
``_asset_digest`` before and after touching a sibling - is
:meth:`TestAnIncludedSiblingChangesTheKey.test_editing_the_included_file_changes_the_key`.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.mjcf_assets import (  # noqa: E402 - after importorskip
    _asset_digest,
    _referenced_files,
)


def _lekiwi_layout(root: pathlib.Path, *, arm_body: str = "arm") -> pathlib.Path:
    """The ``lekiwi`` shape: an entry point including a sibling directory's file."""
    (root / "lekiwi").mkdir(parents=True)
    (root / "so_arm100").mkdir(parents=True)
    (root / "so_arm100" / "so_arm100.xml").write_text(
        f'<mujoco model="arm"><worldbody><body name="{arm_body}"/></worldbody></mujoco>',
        encoding="utf-8",
    )
    entry = root / "lekiwi" / "lekiwi.xml"
    entry.write_text(
        '<mujoco model="lekiwi"><include file="../so_arm100/so_arm100.xml"/></mujoco>',
        encoding="utf-8",
    )
    return entry


def _asimov_layout(root: pathlib.Path, *, mesh_bytes: bytes = b"solid a\nendsolid a\n") -> pathlib.Path:
    """The ``asimov_v0`` shape: ``meshdir`` pointing out of the entry directory."""
    (root / "xmls").mkdir(parents=True)
    (root / "assets" / "meshes").mkdir(parents=True)
    (root / "assets" / "meshes" / "link.stl").write_bytes(mesh_bytes)
    entry = root / "xmls" / "asimov.xml"
    entry.write_text(
        '<mujoco model="asimov">'
        '<compiler meshdir="../assets/meshes"/>'
        '<asset><mesh name="link" file="link.stl"/></asset>'
        "</mujoco>",
        encoding="utf-8",
    )
    return entry


class TestAnIncludedSiblingChangesTheKey:
    """``lekiwi``: the arm's whole description lives outside the digest root."""

    def test_the_included_file_is_part_of_the_closure(self, tmp_path: pathlib.Path) -> None:
        entry = _lekiwi_layout(tmp_path)

        referenced = _referenced_files(str(entry))

        assert str(tmp_path / "so_arm100" / "so_arm100.xml") in referenced

    def test_editing_the_included_file_changes_the_key(self, tmp_path: pathlib.Path) -> None:
        """The reviewer's recipe, and the headline defect: an upstream update to
        the arm left the key identical and served the old USD."""
        entry = _lekiwi_layout(tmp_path)
        before = _asset_digest(str(entry))

        (tmp_path / "so_arm100" / "so_arm100.xml").write_text(
            '<mujoco model="arm"><worldbody><body name="renamed_joint_vocabulary"/></worldbody></mujoco>',
            encoding="utf-8",
        )
        after = _asset_digest(str(entry))

        assert before != after, "a changed <include> target left the cache key identical"

    def test_two_trees_differing_only_outside_do_not_share_a_key(self, tmp_path: pathlib.Path) -> None:
        """The converse collision: identical entry directories, different arms."""
        left = _lekiwi_layout(tmp_path / "left", arm_body="arm")
        right = _lekiwi_layout(tmp_path / "right", arm_body="a_different_arm")

        assert _asset_digest(str(left)) != _asset_digest(str(right))

    def test_a_nested_include_is_followed(self, tmp_path: pathlib.Path) -> None:
        """An include resolves against the INCLUDING file, so the chain continues
        out of a second directory."""
        entry = _lekiwi_layout(tmp_path)
        (tmp_path / "deeper").mkdir()
        (tmp_path / "deeper" / "hand.xml").write_text('<mujoco model="hand"/>', encoding="utf-8")
        (tmp_path / "so_arm100" / "so_arm100.xml").write_text(
            '<mujoco model="arm"><include file="../deeper/hand.xml"/></mujoco>',
            encoding="utf-8",
        )
        before = _asset_digest(str(entry))

        (tmp_path / "deeper" / "hand.xml").write_text('<mujoco model="hand2"/>', encoding="utf-8")

        assert before != _asset_digest(str(entry))


class TestAMeshOutsideTheEntryDirectoryChangesTheKey:
    """``asimov_v0``: ``meshdir="../assets/meshes"``, so the simulated geometry is
    outside the digest root."""

    def test_the_mesh_is_part_of_the_closure(self, tmp_path: pathlib.Path) -> None:
        entry = _asimov_layout(tmp_path)

        referenced = _referenced_files(str(entry))

        assert str(tmp_path / "assets" / "meshes" / "link.stl") in referenced

    def test_editing_the_mesh_changes_the_key(self, tmp_path: pathlib.Path) -> None:
        entry = _asimov_layout(tmp_path)
        before = _asset_digest(str(entry))

        (tmp_path / "assets" / "meshes" / "link.stl").write_bytes(b"solid different\nendsolid different\n")

        assert before != _asset_digest(str(entry))

    def test_a_meshdir_declared_in_an_included_fragment_resolves_against_the_entry(
        self, tmp_path: pathlib.Path
    ) -> None:
        """``<compiler>`` is model-global: MuJoCo resolves a relative ``meshdir``
        against the MODEL file's directory even when an included fragment in a
        subdirectory declared it, which is the rule
        :func:`~strands_robots.simulation.isaac.loaders._parse_mjcf_mesh_assets`
        applies. Resolving it against the fragment instead would hash the wrong
        path, so the mesh would be missed exactly as before the fix."""
        (tmp_path / "frag").mkdir()
        (tmp_path / "meshes").mkdir()
        (tmp_path / "meshes" / "link.stl").write_bytes(b"one")
        (tmp_path / "frag" / "body.xml").write_text(
            '<mujoco><compiler meshdir="meshes"/><asset><mesh name="l" file="link.stl"/></asset></mujoco>',
            encoding="utf-8",
        )
        entry = tmp_path / "model.xml"
        entry.write_text('<mujoco><include file="frag/body.xml"/></mujoco>', encoding="utf-8")

        referenced = _referenced_files(str(entry))

        assert str(tmp_path / "meshes" / "link.stl") in referenced


class TestTheEntryDirectoryIsStillCovered:
    """The closure is a superset of the original walk, not a replacement.

    The walk catches files no parser models - a texture referenced from inside an
    STL, a file an importer picks up by convention - so dropping it in favour of
    the closure would trade one blind spot for another.
    """

    def test_a_sibling_inside_the_directory_still_changes_the_key(self, tmp_path: pathlib.Path) -> None:
        entry = tmp_path / "scene.xml"
        entry.write_text('<mujoco model="s"/>', encoding="utf-8")
        (tmp_path / "unreferenced.stl").write_bytes(b"a")
        before = _asset_digest(str(entry))

        (tmp_path / "unreferenced.stl").write_bytes(b"b")

        assert before != _asset_digest(str(entry))

    def test_two_entry_points_in_one_directory_differ(self, tmp_path: pathlib.Path) -> None:
        """Menagerie ships ``scene.xml`` beside the bare body, and they convert to
        different USD - the property the original digest already had."""
        (tmp_path / "scene.xml").write_text('<mujoco model="scene"/>', encoding="utf-8")
        (tmp_path / "robot.xml").write_text('<mujoco model="robot"/>', encoding="utf-8")

        assert _asset_digest(str(tmp_path / "scene.xml")) != _asset_digest(str(tmp_path / "robot.xml"))

    def test_the_key_is_stable_across_repeated_reads(self, tmp_path: pathlib.Path) -> None:
        """A digest that varied per call would miss every cache hit."""
        entry = _lekiwi_layout(tmp_path)

        assert _asset_digest(str(entry)) == _asset_digest(str(entry))


class TestABrokenReferenceIsNotAFailure:
    """This computes a cache key. Refusing to produce one would fail a conversion
    the vendor importer reports on perfectly well itself."""

    def test_a_missing_include_still_yields_a_key(self, tmp_path: pathlib.Path) -> None:
        entry = tmp_path / "model.xml"
        entry.write_text('<mujoco><include file="../nowhere/absent.xml"/></mujoco>', encoding="utf-8")

        assert isinstance(_asset_digest(str(entry)), str)

    def test_a_malformed_include_still_yields_a_key(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "frag").mkdir()
        (tmp_path / "frag" / "bad.xml").write_text("<mujoco><unclosed>", encoding="utf-8")
        entry = tmp_path / "model.xml"
        entry.write_text('<mujoco><include file="frag/bad.xml"/></mujoco>', encoding="utf-8")

        assert isinstance(_asset_digest(str(entry)), str)

    def test_a_cyclic_include_terminates(self, tmp_path: pathlib.Path) -> None:
        a = tmp_path / "a.xml"
        b = tmp_path / "b.xml"
        a.write_text('<mujoco><include file="b.xml"/></mujoco>', encoding="utf-8")
        b.write_text('<mujoco><include file="a.xml"/></mujoco>', encoding="utf-8")

        assert isinstance(_asset_digest(str(a)), str)

    def test_a_missing_reference_still_distinguishes_two_trees(self, tmp_path: pathlib.Path) -> None:
        """The path enters the manifest even when the bytes cannot, so two trees
        differing only in which absent file they name do not collide."""
        left = tmp_path / "left"
        right = tmp_path / "right"
        for directory, target in ((left, "absent_one.xml"), (right, "absent_two.xml")):
            directory.mkdir()
            (directory / "model.xml").write_text(f'<mujoco><include file="../{target}"/></mujoco>', encoding="utf-8")

        assert _asset_digest(str(left / "model.xml")) != _asset_digest(str(right / "model.xml"))
