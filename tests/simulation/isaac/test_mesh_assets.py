"""Unit pins for :mod:`strands_robots.simulation.isaac.mesh_assets` (#2459).

The OBJ/STL parsing + USD conversion is the CPU half of realizing LIBERO
objects as real meshes on the Isaac stage: it must be exercisable with no
Isaac Sim install (parsing is pure stdlib; conversion needs only the
``usd-core`` wheel), and it must fail loud - a missing or empty asset is an
error, never a silent default box, because a box standing in for a bowl is
exactly the eval-integrity defect the feature removes.
"""

from __future__ import annotations

import struct

import pytest

from strands_robots.simulation.isaac.mesh_assets import (
    MESH_EXTENSIONS,
    USD_EXTENSIONS,
    convert_mesh_to_usd,
    load_mesh_geometry,
    mesh_aabb,
)

_TETRA_OBJ = "v 0 0 0\nv 1 0 0\nv 0 2 0\nv 0 0 3\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"


def _binary_stl(tri_vertices: list[tuple[tuple[float, float, float], ...]]) -> bytes:
    """Assemble a well-formed binary STL from triangles of xyz tuples."""
    blob = b"\x00" * 80 + struct.pack("<I", len(tri_vertices))
    for tri in tri_vertices:
        rec = [0.0, 0.0, 0.0]
        for vert in tri:
            rec.extend(vert)
        blob += struct.pack("<12f", *rec) + struct.pack("<H", 0)
    return blob


def _binary_msh(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    *,
    nnormal: int = 0,
    ntexcoord: int = 0,
) -> bytes:
    """Assemble a legacy MuJoCo binary mesh: the format LIBERO's compiled
    scenes declare for the bowl/plate visual assets."""
    blob = struct.pack("<4i", len(vertices), nnormal, ntexcoord, len(faces))
    for v in vertices:
        blob += struct.pack("<3f", *v)
    blob += b"\x00" * (4 * (3 * nnormal + 2 * ntexcoord))
    for f in faces:
        blob += struct.pack("<3i", *f)
    return blob


class TestLoadMeshGeometry:
    def test_obj_vertices_and_faces(self, tmp_path):
        asset = tmp_path / "tetra.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")
        points, counts, indices = load_mesh_geometry(str(asset))
        assert len(points) == 4
        assert counts == [3, 3, 3, 3]
        assert max(indices) == 3 and min(indices) == 0

    def test_obj_slash_syntax_and_quads(self, tmp_path):
        # ``i/t/n`` face refs and n-gon faces both come from real exports;
        # only the vertex index is consumed and polygons are kept as n-gons.
        asset = tmp_path / "quad.obj"
        asset.write_text(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\nf 1/1/1 2/1/1 3/1/1 4/1/1\n",
            encoding="utf-8",
        )
        points, counts, indices = load_mesh_geometry(str(asset))
        assert len(points) == 4
        assert counts == [4]
        assert indices == [0, 1, 2, 3]

    def test_obj_negative_indices(self, tmp_path):
        asset = tmp_path / "neg.obj"
        asset.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n", encoding="utf-8")
        _points, counts, indices = load_mesh_geometry(str(asset))
        assert counts == [3]
        assert indices == [0, 1, 2]

    def test_obj_out_of_range_index_is_an_error(self, tmp_path):
        asset = tmp_path / "bad.obj"
        asset.write_text("v 0 0 0\nv 1 0 0\nf 1 2 3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="out of range"):
            load_mesh_geometry(str(asset))

    def test_binary_stl(self, tmp_path):
        asset = tmp_path / "part.stl"
        tri1 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        tri2 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        asset.write_bytes(_binary_stl([tri1, tri2]))
        points, counts, indices = load_mesh_geometry(str(asset))
        # Shared vertices are deduplicated: 6 raw corners -> 4 unique.
        assert len(points) == 4
        assert counts == [3, 3]
        assert len(indices) == 6

    def test_ascii_stl(self, tmp_path):
        asset = tmp_path / "part.stl"
        asset.write_text(
            "solid part\n"
            " facet normal 0 0 1\n"
            "  outer loop\n"
            "   vertex 0 0 0\n   vertex 1 0 0\n   vertex 0 1 0\n"
            "  endloop\n"
            " endfacet\n"
            "endsolid part\n",
            encoding="utf-8",
        )
        points, counts, _indices = load_mesh_geometry(str(asset))
        assert len(points) == 3
        assert counts == [3]

    def test_binary_msh(self, tmp_path):
        # The format LIBERO's robosuite-compiled MJCFs reference for the
        # bowl/plate VISUAL meshes - the exact objects #2459 names - so it
        # must parse, or those objects stay box proxies.
        asset = tmp_path / "bowl_vis.msh"
        verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)]
        faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        asset.write_bytes(_binary_msh(verts, faces, nnormal=4, ntexcoord=4))
        points, counts, indices = load_mesh_geometry(str(asset))
        assert points == verts
        assert counts == [3, 3, 3, 3]
        assert indices == [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]

    def test_msh_truncated_is_an_error(self, tmp_path):
        # Declared counts must reconcile with the byte length; a truncated
        # file must not parse as garbage geometry.
        asset = tmp_path / "torn.msh"
        blob = _binary_msh([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0, 1, 2)])
        asset.write_bytes(blob[:-4])
        with pytest.raises(ValueError, match="truncated"):
            load_mesh_geometry(str(asset))

    def test_msh_out_of_range_face_index_is_an_error(self, tmp_path):
        asset = tmp_path / "bad.msh"
        asset.write_bytes(_binary_msh([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0, 1, 3)]))
        with pytest.raises(ValueError, match="out of range"):
            load_mesh_geometry(str(asset))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mesh_geometry(str(tmp_path / "nope.obj"))

    def test_unsupported_extension_raises(self, tmp_path):
        asset = tmp_path / "part.dae"
        asset.write_bytes(b"\x00")
        with pytest.raises(ValueError, match="unsupported mesh format"):
            load_mesh_geometry(str(asset))

    def test_empty_mesh_raises(self, tmp_path):
        # An asset with no faces renders nothing; downstream would misread
        # the blank as a scene property, so the parse is where it fails.
        asset = tmp_path / "empty.obj"
        asset.write_text("v 0 0 0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no triangle geometry"):
            load_mesh_geometry(str(asset))


class TestMeshAabb:
    def test_full_extents_and_center(self, tmp_path):
        asset = tmp_path / "tetra.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")
        center, size = mesh_aabb(str(asset))
        assert center == (0.5, 1.0, 1.5)
        assert size == (1.0, 2.0, 3.0)

    def test_scale_is_applied(self, tmp_path):
        asset = tmp_path / "tetra.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")
        center, size = mesh_aabb(str(asset), scale=(2.0, 1.0, 0.5))
        assert center == (1.0, 1.0, 0.75)
        assert size == (2.0, 2.0, 1.5)


class TestConvertMeshToUsd:
    """Needs ``pxr`` (the ``usd-core`` wheel from the ``sim-isaac`` extra)."""

    @pytest.fixture(autouse=True)
    def _require_pxr(self):
        pytest.importorskip("pxr")

    def test_authors_a_referenceable_usd(self, tmp_path):
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]

        asset = tmp_path / "tetra.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")
        out = convert_mesh_to_usd(str(asset), cache_dir=str(tmp_path / "cache"))
        stage = Usd.Stage.Open(out)
        prim = stage.GetDefaultPrim()
        assert prim.IsValid() and prim.IsA(UsdGeom.Mesh)
        mesh = UsdGeom.Mesh(prim)
        assert len(mesh.GetPointsAttr().Get()) == 4
        assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3, 3, 3, 3]

    def test_conversion_is_cached_by_content(self, tmp_path):
        asset = tmp_path / "tetra.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")
        cache = str(tmp_path / "cache")
        first = convert_mesh_to_usd(str(asset), cache_dir=cache)
        second = convert_mesh_to_usd(str(asset), cache_dir=cache)
        assert first == second
        # Same bytes under a different name hit the same cache entry.
        copy = tmp_path / "renamed.obj"
        copy.write_text(_TETRA_OBJ, encoding="utf-8")
        assert convert_mesh_to_usd(str(copy), cache_dir=cache) == first

    def test_usd_input_is_passed_through(self, tmp_path):
        asset = tmp_path / "tetra.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")
        out = convert_mesh_to_usd(str(asset), cache_dir=str(tmp_path / "cache"))
        assert convert_mesh_to_usd(out, cache_dir=str(tmp_path / "other")) == out

    def test_missing_usd_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert_mesh_to_usd(str(tmp_path / "nope.usd"))

    def test_no_torn_cache_entry_on_failure(self, tmp_path):
        # A conversion that fails must leave nothing at the cache path a
        # later call would trust (the atomic-rename contract).
        asset = tmp_path / "bad.obj"
        asset.write_text("v 0 0 0\nf 1 2 3\n", encoding="utf-8")  # out-of-range face
        cache = tmp_path / "cache"
        with pytest.raises(ValueError):
            convert_mesh_to_usd(str(asset), cache_dir=str(cache))
        assert (
            not any(p.suffix == ".usda" and not p.name.startswith(".") for p in cache.glob("*")) or not cache.exists()
        )


def test_extension_vocabularies_are_disjoint():
    assert not set(MESH_EXTENSIONS) & set(USD_EXTENSIONS)
