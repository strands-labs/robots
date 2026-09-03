"""Behavior tests for ``MuJoCoSimEngine._ensure_meshes``.

``_ensure_meshes`` checks whether the mesh files (``.stl``/``.obj``) referenced
by a model XML are present on disk and, if any are missing, triggers a one-time
Menagerie auto-download. Its contract (see the method docstring) is:

* return ``None`` when every referenced mesh is already present (or downloads
  cleanly), so ``add_robot`` proceeds;
* return a standard ``{"status": "error", ...}`` dict when the auto-download
  fails, so the caller can propagate a clear message to the agent instead of
  letting MuJoCo raise a cryptic 'mesh not found'.

These paths were previously unexercised. The download-failure branch is the
one callers MUST propagate, so it is pinned here explicitly.
"""

from __future__ import annotations

import inspect
import os
import struct

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_ensure_meshes = MuJoCoSimEngine._ensure_meshes


def _binary_stl(faces=((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))) -> bytes:
    """A loadable binary STL (a unit tetrahedron by default).

    MuJoCo compiles a mesh it can hull, so a fixture that claims its meshes are
    present has to hand it real geometry - a stub file is refused for reasons
    that have nothing to do with where the reference resolved.
    """
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    out = b"\0" * 80 + struct.pack("<I", len(faces))
    for face in faces:
        out += struct.pack("<3f", 0.0, 0.0, 0.0)  # normal; MuJoCo recomputes it
        for index in face:
            out += struct.pack("<3f", *vertices[index])
        out += b"\0\0"  # attribute byte count
    return out


def _write(path, content):
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_no_mesh_references_returns_none(tmp_path):
    """A model that references no mesh files needs no download."""
    model = _write(
        tmp_path / "robot.xml",
        '<mujoco><worldbody><geom type="box" size="1 1 1"/></worldbody></mujoco>',
    )
    assert _ensure_meshes(model, "robot") is None


def test_all_meshes_present_returns_none(tmp_path, monkeypatch):
    """When every referenced mesh exists on disk, no download is attempted."""
    (tmp_path / "arm.stl").write_bytes(b"solid\n")
    model = _write(
        tmp_path / "robot.xml",
        '<mujoco><asset><mesh file="arm.stl"/></asset></mujoco>',
    )

    def _boom(*a, **k):
        raise AssertionError("download must not be called when meshes are present")

    monkeypatch.setattr("strands_robots.assets.download.download_robots", _boom)
    assert _ensure_meshes(model, "robot") is None


def test_meshdir_is_honored_when_resolving_mesh_paths(tmp_path, monkeypatch):
    """A ``meshdir`` attribute is joined onto the mesh path before existence check."""
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "arm.stl").write_bytes(b"solid\n")
    model = _write(
        tmp_path / "robot.xml",
        '<mujoco><compiler meshdir="meshes"/><asset><mesh file="arm.stl"/></asset></mujoco>',
    )

    def _boom(*a, **k):
        raise AssertionError("mesh resolves under meshdir; no download expected")

    monkeypatch.setattr("strands_robots.assets.download.download_robots", _boom)
    assert _ensure_meshes(model, "robot") is None


def test_missing_mesh_triggers_successful_download(tmp_path, monkeypatch):
    """A missing mesh triggers auto-download; a clean download yields ``None``."""
    model = _write(
        tmp_path / "robot.xml",
        '<mujoco><asset><mesh file="absent.stl"/></asset></mujoco>',
    )
    calls = {}

    def _ok(names, force):
        calls["names"] = names
        calls["force"] = force

    monkeypatch.setattr("strands_robots.assets.resolve_robot_name", lambda n: n)
    monkeypatch.setattr("strands_robots.assets.download.download_robots", _ok)

    assert _ensure_meshes(model, "so100") is None
    assert calls == {"names": ["so100"], "force": True}


def test_missing_mesh_download_failure_returns_error_dict(tmp_path, monkeypatch):
    """When auto-download fails, an error dict (not ``None``) is returned."""
    model = _write(
        tmp_path / "robot.xml",
        '<mujoco><asset><mesh file="absent.stl"/></asset></mujoco>',
    )

    def _fail(names, force):
        raise OSError("network down")

    monkeypatch.setattr("strands_robots.assets.resolve_robot_name", lambda n: n)
    monkeypatch.setattr("strands_robots.assets.download.download_robots", _fail)

    result = _ensure_meshes(model, "so100")
    assert isinstance(result, dict)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "so100" in text
    assert "network down" in text


def test_missing_mesh_in_included_file_is_detected(tmp_path, monkeypatch):
    """Mesh refs inside an ``<include>``d file are checked, not just the top XML."""
    _write(
        tmp_path / "parts.xml",
        '<mujoco><asset><mesh file="absent.stl"/></asset></mujoco>',
    )
    model = _write(
        tmp_path / "robot.xml",
        '<mujoco><include file="parts.xml"/></mujoco>',
    )
    seen = {}

    def _ok(names, force):
        seen["called"] = True

    monkeypatch.setattr("strands_robots.assets.resolve_robot_name", lambda n: n)
    monkeypatch.setattr("strands_robots.assets.download.download_robots", _ok)

    assert _ensure_meshes(model, "robot") is None
    assert seen.get("called") is True


def test_unreadable_model_path_is_tolerated(tmp_path):
    """A model path that cannot be opened is skipped, not raised on.

    Both the top-level include scan and the per-file mesh scan swallow read
    errors so a transient/odd path never crashes ``add_robot``; with nothing
    readable there is nothing to download, so ``None`` is returned.
    """
    missing_path = str(tmp_path / "does_not_exist.xml")
    assert _ensure_meshes(missing_path, "robot") is None


class TestAReferenceIsResolvedTheWayMuJoCoResolvesIt:
    """``_ensure_meshes`` must agree with MuJoCo about where a mesh lives.

    MuJoCo resolves a ``<mesh file=...>`` against the MAIN model file's
    directory plus the model's mesh subdirectory. Two details are easy to get
    wrong and both make a present mesh look absent:

    * the subdirectory can be declared as ``assetdir`` (which sets the mesh and
      texture directories together) rather than ``meshdir``, and
    * ``<compiler>`` is model-global, so the fragment declaring the directory
      need not be the fragment declaring the mesh, and neither is resolved
      against the declaring fragment's own directory.

    A false "missing" is not cosmetic: it spends a full ``force=True``
    re-download on every ``add_robot``, and ``add_robot`` returns that
    download's error dict when the download cannot run - refusing a robot whose
    meshes are all on disk.
    """

    @staticmethod
    def _no_download(monkeypatch):
        """Fail the test if any download is attempted."""

        def _boom(*a, **k):
            raise AssertionError("meshes are on disk; no download expected")

        monkeypatch.setattr("strands_robots.assets.download.download_robots", _boom)

    def test_assetdir_is_honored_like_meshdir(self, tmp_path, monkeypatch):
        """``assetdir`` names the mesh directory too, so it must be joined on."""
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "arm.stl").write_bytes(b"solid\n")
        model = _write(
            tmp_path / "robot.xml",
            '<mujoco><compiler assetdir="assets"/><asset><mesh file="arm.stl"/></asset></mujoco>',
        )
        self._no_download(monkeypatch)
        assert _ensure_meshes(model, "robot") is None

    def test_meshdir_overrides_assetdir(self, tmp_path, monkeypatch):
        """When both appear, ``meshdir`` wins - as it does in MuJoCo."""
        (tmp_path / "meshes").mkdir()
        (tmp_path / "meshes" / "arm.stl").write_bytes(b"solid\n")
        model = _write(
            tmp_path / "robot.xml",
            '<mujoco><compiler assetdir="nowhere" meshdir="meshes"/><asset><mesh file="arm.stl"/></asset></mujoco>',
        )
        self._no_download(monkeypatch)
        assert _ensure_meshes(model, "robot") is None

    def test_a_compiler_in_the_top_file_applies_to_an_included_mesh(self, tmp_path, monkeypatch):
        """``<compiler>`` is model-global: it reaches a mesh declared elsewhere."""
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "arm.stl").write_bytes(b"solid\n")
        _write(tmp_path / "parts.xml", '<mujoco><asset><mesh file="arm.stl"/></asset></mujoco>')
        model = _write(
            tmp_path / "robot.xml",
            '<mujoco><compiler meshdir="assets"/><include file="parts.xml"/></mujoco>',
        )
        self._no_download(monkeypatch)
        assert _ensure_meshes(model, "robot") is None

    def test_a_compiler_in_an_included_file_applies_to_a_top_level_mesh(self, tmp_path, monkeypatch):
        """The same, in the other direction."""
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "arm.stl").write_bytes(b"solid\n")
        _write(tmp_path / "parts.xml", '<mujoco><compiler meshdir="assets"/></mujoco>')
        model = _write(
            tmp_path / "robot.xml",
            '<mujoco><include file="parts.xml"/><asset><mesh file="arm.stl"/></asset></mujoco>',
        )
        self._no_download(monkeypatch)
        assert _ensure_meshes(model, "robot") is None

    def test_an_included_fragments_mesh_resolves_against_the_main_model_dir(self, tmp_path, monkeypatch):
        """A fragment in a subdirectory still resolves against the model root.

        This is the shipped ``skydio_x2`` / ``stretch3`` layout: the fragment
        that declares the meshes lives beside the model file's directory while
        the meshes sit under the model's own asset directory.
        """
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "arm.stl").write_bytes(b"solid\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        _write(
            sub / "parts.xml",
            '<mujoco><compiler assetdir="assets"/><asset><mesh file="arm.stl"/></asset></mujoco>',
        )
        model = _write(tmp_path / "robot.xml", '<mujoco><include file="sub/parts.xml"/></mujoco>')
        self._no_download(monkeypatch)
        assert _ensure_meshes(model, "robot") is None

    def test_an_included_fragments_mesh_may_also_sit_beside_that_fragment(self, tmp_path, monkeypatch):
        """MuJoCo also accepts the reference relative to the fragment's directory.

        This is the shipped ``lekiwi`` layout, and it is why both candidate
        locations are checked rather than only the model root.
        """
        sub = tmp_path / "sub"
        (sub / "meshes").mkdir(parents=True)
        (sub / "meshes" / "arm.stl").write_bytes(b"solid\n")
        _write(sub / "parts.xml", '<mujoco><asset><mesh file="meshes/arm.stl"/></asset></mujoco>')
        model = _write(tmp_path / "robot.xml", '<mujoco><include file="sub/parts.xml"/></mujoco>')
        self._no_download(monkeypatch)
        assert _ensure_meshes(model, "robot") is None

    def test_a_genuinely_absent_mesh_is_still_reported(self, tmp_path, monkeypatch):
        """Widening where we look must not stop us noticing a real absence."""
        (tmp_path / "assets").mkdir()
        _write(
            tmp_path / "robot.xml",
            '<mujoco><compiler assetdir="assets"/><asset><mesh file="arm.stl"/></asset></mujoco>',
        )
        called = {}
        monkeypatch.setattr("strands_robots.assets.resolve_robot_name", lambda n: n)
        monkeypatch.setattr(
            "strands_robots.assets.download.download_robots",
            lambda names, force: called.update(names=names, force=force),
        )
        assert _ensure_meshes(str(tmp_path / "robot.xml"), "robot") is None
        assert called == {"names": ["robot"], "force": True}


class TestTheDownloadPathUsesTheSameRule:
    """``_needs_download`` decides the same question and must not disagree.

    Both paths ask "are this model's meshes on disk?". They read the rule - and
    the scan that applies it - from one place, so a model cannot be judged
    present by one and absent by the other.

    The shape that separates two owners is a model whose meshes only an
    ``<include>``d fragment declares. Shipped Menagerie assets are built this
    way (``ability_hand``'s scene declares none of its 13 meshes; its included
    hand fragment declares all of them plus the ``meshdir``), and a scan that
    reads the main file alone finds no reference to check there at all - which
    is indistinguishable from a model whose meshes are all present.
    """

    @staticmethod
    def _include_shaped_model(tmp_path, *, meshes_present: bool):
        """Write a model whose only mesh - and its ``meshdir`` - an include declares.

        Returns the main model path and the registry entry naming it.
        """
        root = tmp_path / "robots"
        (root / "parts").mkdir(parents=True)
        (root / "assets").mkdir()
        _write(
            root / "parts" / "arm.xml",
            '<mujoco><compiler meshdir="assets"/><asset><mesh file="arm.stl"/></asset>'
            '<worldbody><body><geom type="mesh" mesh="arm"/></body></worldbody></mujoco>',
        )
        model = _write(root / "robot.xml", '<mujoco><include file="parts/arm.xml"/></mujoco>')
        if meshes_present:
            (root / "assets" / "arm.stl").write_bytes(_binary_stl())
        return model, {"asset": {"model_xml": "robot.xml", "dir": "robots"}}

    @pytest.mark.parametrize("meshes_present", [True, False])
    def test_the_two_owners_agree_about_a_mesh_only_an_include_declares(self, tmp_path, monkeypatch, meshes_present):
        """Both answers track MuJoCo's own verdict on the same tree.

        MuJoCo is the tie-breaker: it is the reader whose opinion the two
        owners exist to predict, so the case is not built on either one's
        reading of the model.
        """
        import mujoco

        import strands_robots.assets.download as dl_mod

        model, info = self._include_shaped_model(tmp_path, meshes_present=meshes_present)
        monkeypatch.setattr(dl_mod, "get_search_paths", lambda: [tmp_path])

        try:
            mujoco.MjModel.from_xml_path(model)
            mujoco_loads = True
        except ValueError:
            mujoco_loads = False
        assert mujoco_loads is meshes_present, "the fixture does not pose the case it claims"

        fetches: list[list[str]] = []
        monkeypatch.setattr("strands_robots.assets.resolve_robot_name", lambda n: n)
        monkeypatch.setattr(dl_mod, "download_robots", lambda names, force: fetches.append(names))
        _ensure_meshes(model, "robot")

        assert bool(fetches) is (not meshes_present), "add_robot's check disagreed with MuJoCo"
        assert dl_mod._needs_download("robot", info) is (not meshes_present), (
            "the download path disagreed with MuJoCo, so the fetch add_robot asks for is a no-op"
        )

    def test_force_reaches_a_model_whose_meshes_an_include_declares(self, tmp_path, monkeypatch):
        """``force`` decides the nothing-missing case, whichever fragment declares it."""
        import strands_robots.assets.download as dl_mod

        _model, info = self._include_shaped_model(tmp_path, meshes_present=True)
        monkeypatch.setattr(dl_mod, "get_search_paths", lambda: [tmp_path])

        assert dl_mod._needs_download("robot", info, force=False) is False
        assert dl_mod._needs_download("robot", info, force=True) is True

    def test_needs_download_honors_assetdir(self, tmp_path, monkeypatch):
        import strands_robots.assets.download as dl_mod
        from strands_robots.assets.download import _needs_download

        (tmp_path / "robots" / "assets").mkdir(parents=True)
        (tmp_path / "robots" / "assets" / "arm.stl").write_bytes(b"solid\n")
        (tmp_path / "robots" / "robot.xml").write_text(
            '<mujoco><compiler assetdir="assets"/><asset><mesh file="arm.stl"/></asset></mujoco>',
            encoding="utf-8",
        )
        info = {"asset": {"model_xml": "robot.xml", "dir": "robots"}}
        monkeypatch.setattr(dl_mod, "get_search_paths", lambda: [tmp_path])
        assert _needs_download("robot", info) is False

    def test_both_paths_read_the_scan_from_one_place(self):
        """The scan has a single owner, so the two callers cannot drift."""
        import strands_robots.assets.download as dl_mod
        from strands_robots.simulation.mujoco import simulation as sim_mod

        assert callable(dl_mod._mjcf_missing_meshes)
        for source in (
            inspect.getsource(sim_mod.MuJoCoSimEngine._ensure_meshes),
            inspect.getsource(dl_mod._needs_download),
        ):
            assert "_mjcf_missing_meshes" in source
            # ...and no second copy of the rules that owner applies: neither the
            # subdir attributes, nor which fragments make up the model, nor
            # which extensions name a mesh.
            assert 'meshdir="([^"]*)"' not in source
            assert "<include\\s+file=" not in source
            assert "(?:stl|STL" not in source


class TestTheSubdirRuleItself:
    """Unit coverage for the shared rule, including the precedence."""

    def test_meshdir_wins_over_assetdir_across_fragments(self):
        from strands_robots.assets.download import _mjcf_mesh_subdir

        assert _mjcf_mesh_subdir('<compiler assetdir="a"/>', '<compiler meshdir="m"/>') == "m"
        assert _mjcf_mesh_subdir('<compiler meshdir="m"/>', '<compiler assetdir="a"/>') == "m"

    def test_assetdir_is_used_when_no_meshdir_is_declared(self):
        from strands_robots.assets.download import _mjcf_mesh_subdir

        assert _mjcf_mesh_subdir('<compiler assetdir="a"/>') == "a"

    def test_nothing_declared_yields_the_model_dir_itself(self):
        from strands_robots.assets.download import _mjcf_mesh_subdir

        assert _mjcf_mesh_subdir("<mujoco/>") == ""
        assert _mjcf_mesh_subdir() == ""

    def test_the_declaring_fragments_dir_is_not_a_candidate(self):
        """MuJoCo rejects <fragment dir>/<subdir>/<file>, so we must not accept it."""
        from strands_robots.assets.download import _mjcf_mesh_candidates

        candidates = _mjcf_mesh_candidates("arm.stl", "/model", "assets", "sub")
        assert candidates == ["/model/assets/arm.stl", "/model/assets/sub/arm.stl"]
        assert "/model/sub/assets/arm.stl" not in candidates
