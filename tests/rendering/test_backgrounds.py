# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Background renderer contracts (panorama path: zero ML deps)."""

from pathlib import Path

import numpy as np
import pytest

from strands_robots.rendering import (
    GSPLAT_SCENES,
    CameraParams,
    PanoramaBackground,
    download_gsplat_scene,
    gsplat_scene_names,
    gsplat_skybox_align_for,
    gsplat_skybox_scene_names,
)


def _cam(width: int = 64, height: int = 48) -> CameraParams:
    fy = 0.5 * height / np.tan(np.deg2rad(45.0) / 2.0)
    K = np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])
    return CameraParams(K=K, T_world_cam=np.eye(4), width=width, height=height, znear=0.01, zfar=100.0)


def test_procedural_panorama_renders_at_camera_resolution() -> None:
    cam = _cam()
    rgb, depth = PanoramaBackground().render(cam)
    assert rgb.shape == (48, 64, 3)
    assert rgb.dtype == np.uint8
    assert depth.shape == (48, 64)
    # Panorama sits at infinity: every pixel reports the far plane, so any
    # finite foreground wins the compositor's depth test.
    assert np.all(depth == np.float32(cam.zfar))


def test_procedural_panorama_is_deterministic() -> None:
    cam = _cam(32, 24)
    a, _ = PanoramaBackground().render(cam)
    b, _ = PanoramaBackground().render(cam)
    assert np.array_equal(a, b)


def test_panorama_missing_image_path_falls_back_to_procedural(tmp_path) -> None:
    bg = PanoramaBackground(image_path=tmp_path / "nope.jpg")
    rgb, _ = bg.render(_cam(16, 12))
    assert rgb.shape == (12, 16, 3)


def test_download_gsplat_scene_rejects_unknown_name(tmp_path) -> None:
    with pytest.raises(KeyError, match="Unknown scene"):
        download_gsplat_scene("not-a-scene", cache_dir=tmp_path)


def test_download_gsplat_scene_returns_cached_file_without_downloading(tmp_path, monkeypatch) -> None:
    # A present, non-empty cache file short-circuits the network fetch: the
    # helper must return the cached path and never call urlretrieve. (The .spz
    # scene derives slug "tabletop" + ".spz" from its source URL.)
    cached = tmp_path / "tabletop.spz"
    cached.write_bytes(b"already-on-disk")

    def _must_not_download(*args, **kwargs):
        raise AssertionError("urlretrieve must not run on a cache hit")

    monkeypatch.setattr("urllib.request.urlretrieve", _must_not_download)

    dest = download_gsplat_scene("tabletop (indoor room)", cache_dir=tmp_path)
    assert dest == cached
    assert dest.read_bytes() == b"already-on-disk"


def test_download_gsplat_scene_fetches_via_atomic_part_rename(tmp_path, monkeypatch) -> None:
    # Cache miss downloads to a temporary ``.part`` sidecar and only then
    # atomically renames it to the final path, so a killed download never
    # leaves a truncated file at the cache location. A .spz source URL caches
    # under a ``.spz`` extension.
    from strands_robots.rendering import backgrounds

    calls: list[tuple[str, str]] = []

    def _fake_urlretrieve(url, filename):
        calls.append((url, str(filename)))
        Path(filename).write_bytes(b"SPLAT-BYTES")

    monkeypatch.setattr("urllib.request.urlretrieve", _fake_urlretrieve)

    name = "tabletop (indoor room)"
    dest = download_gsplat_scene(name, cache_dir=tmp_path)

    assert dest == tmp_path / "tabletop.spz"
    assert dest.read_bytes() == b"SPLAT-BYTES"
    assert len(calls) == 1
    fetched_url, part_path = calls[0]
    assert fetched_url == backgrounds.GSPLAT_SCENES[name]
    # Downloaded to the temp sidecar, not straight to the final path.
    assert part_path.endswith(".spz.part")
    # The sidecar was renamed away, leaving only the final cached file.
    assert not (tmp_path / "tabletop.spz.part").exists()


def test_download_gsplat_scene_maps_ply_url_to_ply_extension(tmp_path, monkeypatch) -> None:
    # A .ply source URL must cache under a ``.ply`` extension so the loader
    # dispatches to the PLY reader (not the SPZ reader). "bonsai" ships as .ply.
    monkeypatch.setattr(
        "urllib.request.urlretrieve",
        lambda url, filename: Path(filename).write_bytes(b"ply-bytes"),
    )
    dest = download_gsplat_scene("bonsai (indoor tabletop)", cache_dir=tmp_path)
    assert dest == tmp_path / "bonsai.ply"
    assert dest.read_bytes() == b"ply-bytes"


def test_scene_preset_registries_are_consistent() -> None:
    names = gsplat_scene_names()
    assert set(names) == set(GSPLAT_SCENES)
    # Every curated skybox scene is a real preset.
    assert set(gsplat_skybox_scene_names()) <= set(names)
    # Curated alignment comes back as a mutable copy (callers **kwargs it).
    align = gsplat_skybox_align_for("tabletop")
    assert align["up_axis"] == (0.0, 1.0, 0.0)
    align["up_axis"] = None
    assert gsplat_skybox_align_for("tabletop")["up_axis"] == (0.0, 1.0, 0.0)
    # Uncurated scenes get an empty dict (best-effort auto alignment).
    assert gsplat_skybox_align_for("someone_uploaded.ply") == {}


def _write_solid_panorama(path, color, width: int = 256, height: int = 128) -> None:
    """Write a solid-color equirectangular image the loader can read."""
    from PIL import Image

    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[:] = np.asarray(color, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_panorama_loads_equirectangular_image_from_disk(tmp_path) -> None:
    # A real image on disk takes the load path (not the procedural fallback):
    # a solid-color equirectangular panorama must render to that exact color
    # at every pixel, whatever direction each ray points.
    magenta = (255, 0, 255)
    img = tmp_path / "sky.png"
    _write_solid_panorama(img, magenta)
    rgb, _ = PanoramaBackground(image_path=img).render(_cam())
    # Bilinear resampling can floor a uniform texel by 1 LSB; allow that.
    assert np.all(np.abs(rgb.astype(int) - np.array(magenta, dtype=int)) <= 1)


def test_panorama_loads_image_once_and_caches(tmp_path) -> None:
    # The panorama is read from disk on first render and cached: editing the
    # file afterwards must not change subsequent renders (the loader is not
    # re-hit). Proves the cache-return path, observable as a stable render.
    img = tmp_path / "sky.png"
    _write_solid_panorama(img, (10, 200, 30))
    bg = PanoramaBackground(image_path=img)
    first, _ = bg.render(_cam())
    _write_solid_panorama(img, (200, 10, 30))  # overwrite on disk after load
    second, _ = bg.render(_cam())
    assert np.array_equal(first, second)
    assert np.all(np.abs(first.astype(int) - np.array((10, 200, 30), dtype=int)) <= 1)


def test_panorama_yaw_rotation_changes_the_background() -> None:
    # rotation_deg spins the panorama around world +Z, so a non-zero yaw must
    # remap texels to different pixels than the unrotated render. The
    # procedural panorama has azimuthal variation (the window-light lobe), so
    # a 90-degree yaw is observable.
    cam = _cam()
    unrotated, _ = PanoramaBackground(rotation_deg=0.0).render(cam)
    rotated, _ = PanoramaBackground(rotation_deg=90.0).render(cam)
    assert not np.array_equal(unrotated, rotated)


def test_gsplat_rasterizer_available_is_a_nonraising_capability_probe() -> None:
    # The probe must never raise -- callers rely on it to decide up front
    # whether to fall back to PanoramaBackground. It always returns
    # (ok: bool, reason: str) with a non-empty reason.
    from strands_robots.rendering.backgrounds import gsplat_rasterizer_available

    ok, reason = gsplat_rasterizer_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str) and reason


# --------------------------------------------------------------------------- #
# SPZ (Niantic Gaussian SPlat) reader -- pure numpy + torch, no gsplat/CUDA.
# --------------------------------------------------------------------------- #
# The .spz reader is a hand-rolled binary parser (gzip container, 16-byte
# header, struct-of-arrays body, and a bit-packed "smallest-three" quaternion
# codec for version 3). It is the format the curated MuJoCo-GS-Web scenes ship
# in, so a decode regression silently corrupts every splat backdrop. These
# tests pin the wire format by round-tripping known inputs through the parser.

_INV_SQRT2 = 1.0 / np.sqrt(2.0)


def _encode_spz_rotation_v3(q_wxyz: tuple[float, float, float, float]) -> list[int]:
    """Encode a WXYZ quaternion into 4 bytes using the SPZ v3 smallest-three
    codec (the inverse of ``_decode_spz_rotations(..., smallest_three=True)``).

    The largest-magnitude component is dropped (reconstructed on decode from
    unit norm); its index goes in the top 2 bits. The other three are stored
    LSB-first in decode order (component index 3, 2, 1, 0, skipping the
    largest) as a 9-bit magnitude plus a sign bit.
    """
    w, x, y, z = q_wxyz
    xyzw = np.array([x, y, z, w], dtype=float)
    xyzw /= np.linalg.norm(xyzw)
    i_largest = int(np.argmax(np.abs(xyzw)))
    if xyzw[i_largest] < 0:  # canonical form: largest component positive
        xyzw = -xyzw
    comp = 0
    shift = 0
    for i in (3, 2, 1, 0):
        if i == i_largest:
            continue
        val = xyzw[i]
        negbit = 1 if val < 0 else 0
        mag = min(int(round(abs(val) / _INV_SQRT2 * 511)), 511)
        comp |= ((mag & 511) | (negbit << 9)) << shift
        shift += 10
    comp |= (i_largest & 3) << 30
    return [comp & 0xFF, (comp >> 8) & 0xFF, (comp >> 16) & 0xFF, (comp >> 24) & 0xFF]


def _pack_positions(means: np.ndarray, frac_bits: int) -> bytes:
    """Encode (N,3) float means as 9 bytes/point (24-bit LE signed fixed point)."""
    fixed = (np.round(means * (1 << frac_bits)).astype(np.int64)) & 0xFFFFFF
    out = np.zeros((means.shape[0], 3, 3), np.uint8)
    out[:, :, 0] = fixed & 0xFF
    out[:, :, 1] = (fixed >> 8) & 0xFF
    out[:, :, 2] = (fixed >> 16) & 0xFF
    return out.reshape(means.shape[0], 9).tobytes()


def _build_spz(
    tmp_path,
    means: np.ndarray,
    alpha: np.ndarray,
    col: np.ndarray,
    scl: np.ndarray,
    rot: np.ndarray,
    *,
    version: int,
    frac_bits: int = 12,
    sh_degree: int = 0,
):
    """Write a minimal gzip-compressed .spz file and return its path."""
    import gzip
    import struct

    from strands_robots.rendering.backgrounds import _SPZ_MAGIC

    n = means.shape[0]
    header = struct.pack("<iii", _SPZ_MAGIC, version, n) + struct.pack("<BBBB", sh_degree, frac_bits, 0, 0)
    body = (
        _pack_positions(means, frac_bits)
        + alpha.astype(np.uint8).tobytes()
        + col.astype(np.uint8).reshape(n, 3).tobytes()
        + scl.astype(np.uint8).reshape(n, 3).tobytes()
        + rot.astype(np.uint8).tobytes()
    )
    path = tmp_path / f"scene_v{version}.spz"
    path.write_bytes(gzip.compress(header + body))
    return path


class TestSpzGaussianSplatReader:
    """The .spz reader must decode Niantic Gaussian-splat scenes (versions 2
    and 3) into the canonical splat dict, and reject files it cannot parse."""

    def test_decode_rotations_v3_roundtrips_quaternions(self) -> None:
        from strands_robots.rendering.backgrounds import _decode_spz_rotations

        quats = [(1.0, 0.0, 0.0, 0.0), (0.8, 0.1, 0.2, 0.3), (0.0, 1.0, 0.0, 0.0), (0.3, -0.4, 0.5, 0.7)]
        rot = np.array([_encode_spz_rotation_v3(q) for q in quats], np.uint8)
        decoded = _decode_spz_rotations(rot, smallest_three=True)
        assert decoded.shape == (4, 4)
        for q, d in zip(quats, decoded):
            expected = np.array(q) / np.linalg.norm(q)
            if float(np.dot(expected, d)) < 0:  # a quaternion and its negation are the same rotation
                d = -d
            # 9-bit magnitude quantisation caps the error at ~1/sqrt(2)/511.
            assert np.max(np.abs(expected - d)) < 3e-3
            assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-4

    def test_decode_rotations_v2_reconstructs_w_from_xyz(self) -> None:
        from strands_robots.rendering.backgrounds import _decode_spz_rotations

        # v2 stores xyz mapped through byte/127.5 - 1; w is sqrt(1 - |xyz|^2).
        rot = np.array([[128, 128, 128], [200, 60, 130]], np.uint8)
        decoded = _decode_spz_rotations(rot, smallest_three=False)
        assert decoded.shape == (2, 4)
        for byte_row, d in zip(rot, decoded):
            xyz = byte_row[:3].astype(np.float32) / 127.5 - 1.0
            w = np.sqrt(max(0.0, 1.0 - float((xyz * xyz).sum())))
            expected = np.array([w, xyz[0], xyz[1], xyz[2]], np.float32)  # WXYZ order
            assert np.allclose(d, expected, atol=1e-4)

    def test_load_v3_parses_every_field(self, tmp_path) -> None:
        pytest.importorskip("torch")
        from strands_robots.rendering.backgrounds import _load_spz_splats

        means = np.array([[0.5, -0.25, 1.0], [0.0, 0.125, -0.5]], np.float32)
        alpha = np.array([255, 128], np.uint8)
        col = np.array([[128, 128, 128], [255, 0, 64]], np.uint8)
        scl = np.array([[160, 160, 160], [80, 100, 120]], np.uint8)
        rot = np.array(
            [_encode_spz_rotation_v3((1.0, 0.0, 0.0, 0.0)), _encode_spz_rotation_v3((0.8, 0.1, 0.2, 0.3))], np.uint8
        )
        path = _build_spz(tmp_path, means, alpha, col, scl, rot, version=3, frac_bits=12)

        splats = _load_spz_splats(path, device="cpu")
        assert set(splats) == {"means", "scales", "quats", "opacities", "colors"}
        m = splats["means"].numpy()
        assert m.shape == (2, 3) and m.dtype == np.float32
        # 24-bit fixed point at frac_bits=12 recovers the means to sub-mm.
        assert np.allclose(m, means, atol=1e-3)
        # scale = exp(byte/16 - 10); opacity = byte/255.
        assert np.allclose(splats["scales"].numpy(), np.exp(scl.astype(np.float32) / 16.0 - 10.0), atol=1e-5)
        assert np.allclose(splats["opacities"].numpy(), alpha.astype(np.float32) / 255.0, atol=1e-4)
        # DC color decodes to [0,1] linear RGB; the neutral 128 byte maps near 0.5.
        colors = splats["colors"].numpy()
        assert colors.shape == (2, 3)
        assert np.all((colors >= 0.0) & (colors <= 1.0))
        assert np.allclose(colors[0], 0.5, atol=0.02)
        # v3 rot decodes to a unit WXYZ quaternion; row 0 is (near-)identity.
        quats = splats["quats"].numpy()
        assert quats.shape == (2, 4)
        assert np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-3)
        assert np.allclose(np.abs(quats[0]), np.array([1.0, 0.0, 0.0, 0.0]), atol=3e-3)

    def test_load_v2_takes_the_three_byte_rotation_path(self, tmp_path) -> None:
        pytest.importorskip("torch")
        from strands_robots.rendering.backgrounds import _load_spz_splats

        means = np.array([[0.0, 0.0, 0.0]], np.float32)
        alpha = np.array([200], np.uint8)
        col = np.array([[10, 20, 30]], np.uint8)
        scl = np.array([[128, 128, 128]], np.uint8)
        rot = np.array([[128, 128, 128]], np.uint8)  # v2: 3 bytes/point
        path = _build_spz(tmp_path, means, alpha, col, scl, rot, version=2)

        splats = _load_spz_splats(path, device="cpu")
        assert splats["means"].shape == (1, 3)
        assert splats["quats"].shape == (1, 4)  # w reconstructed -> 4 comps
        assert abs(float(splats["opacities"][0]) - 200.0 / 255.0) < 1e-4

    def test_load_rejects_bad_magic(self, tmp_path) -> None:
        pytest.importorskip("torch")
        import gzip
        import struct

        from strands_robots.rendering.backgrounds import _load_spz_splats

        path = tmp_path / "bad_magic.spz"
        path.write_bytes(gzip.compress(struct.pack("<iii", 0xDEAD, 3, 0) + struct.pack("<BBBB", 0, 12, 0, 0)))
        with pytest.raises(ValueError, match="bad SPZ magic"):
            _load_spz_splats(path, device="cpu")

    def test_load_rejects_unsupported_version(self, tmp_path) -> None:
        pytest.importorskip("torch")
        import gzip
        import struct

        from strands_robots.rendering.backgrounds import _SPZ_MAGIC, _load_spz_splats

        path = tmp_path / "bad_version.spz"
        path.write_bytes(gzip.compress(struct.pack("<iii", _SPZ_MAGIC, 9, 0) + struct.pack("<BBBB", 0, 12, 0, 0)))
        with pytest.raises(ValueError, match="unsupported SPZ version"):
            _load_spz_splats(path, device="cpu")


def _build_ply(path, *, means, scales_log, quats, opacity_logit, f_dc):
    """Write a minimal 3DGS ``.ply`` with the fields ``_load_ply_splats`` reads.

    The 3D Gaussian-Splatting PLY layout stores scales in log-space
    (``scale_0..2``), opacity as a logit (``opacity``), rotations as raw WXYZ
    quaternions (``rot_0..3``), and DC color as SH coefficients (``f_dc_0..2``);
    the loader converts each back to the render-ready representation.
    """
    from plyfile import PlyData, PlyElement

    n = len(means)
    fields = [
        ("x", means[:, 0]),
        ("y", means[:, 1]),
        ("z", means[:, 2]),
        ("scale_0", scales_log[:, 0]),
        ("scale_1", scales_log[:, 1]),
        ("scale_2", scales_log[:, 2]),
        ("rot_0", quats[:, 0]),
        ("rot_1", quats[:, 1]),
        ("rot_2", quats[:, 2]),
        ("rot_3", quats[:, 3]),
        ("opacity", opacity_logit),
        ("f_dc_0", f_dc[:, 0]),
        ("f_dc_1", f_dc[:, 1]),
        ("f_dc_2", f_dc[:, 2]),
    ]
    dtype = [(name, "f4") for name, _ in fields]
    verts = np.empty(n, dtype=dtype)
    for name, col in fields:
        verts[name] = np.asarray(col, np.float32)
    PlyData([PlyElement.describe(verts, "vertex")]).write(str(path))


class TestPlyGaussianSplatReader:
    """Pin the 3DGS ``.ply`` decode contract in :func:`_load_ply_splats`.

    The wire layout stores raw training parameters (log-space scales, logit
    opacity, SH DC color); the loader must apply the inverse activations so the
    rasterizer receives linear scales, [0,1] opacity, and [0,1] RGB.
    """

    def test_load_decodes_every_field(self, tmp_path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("plyfile")
        from strands_robots.rendering.backgrounds import _load_ply_splats

        means = np.array([[0.5, -0.25, 1.0], [0.0, 0.125, -0.5]], np.float32)
        scales_log = np.array([[-2.0, -1.0, 0.0], [1.0, 0.5, -0.5]], np.float32)
        # rot_0..3 are stored/loaded verbatim (no re-normalization in the loader).
        quats = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]], np.float32)
        opacity_logit = np.array([0.0, 2.0], np.float32)
        f_dc = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]], np.float32)
        path = tmp_path / "splat.ply"
        _build_ply(
            path,
            means=means,
            scales_log=scales_log,
            quats=quats,
            opacity_logit=opacity_logit,
            f_dc=f_dc,
        )

        splats = _load_ply_splats(path, device="cpu")

        assert set(splats) == {"means", "scales", "quats", "opacities", "colors"}
        # means pass through unchanged.
        assert np.allclose(splats["means"].numpy(), means, atol=1e-6)
        # scales are exp() of the stored log-space values.
        assert np.allclose(splats["scales"].numpy(), np.exp(scales_log), atol=1e-6)
        # rotations are the raw stored quaternions (loader does not normalize).
        assert np.allclose(splats["quats"].numpy(), quats, atol=1e-6)
        # opacity is a sigmoid of the stored logit: 0 -> 0.5, 2 -> ~0.881.
        assert np.allclose(
            splats["opacities"].numpy(),
            1.0 / (1.0 + np.exp(-opacity_logit)),
            atol=1e-6,
        )
        # DC color decodes via 0.5 + SH_C0 * f_dc, clipped to [0, 1].
        sh_c0 = 0.28209479177387814
        colors = splats["colors"].numpy()
        assert colors.shape == (2, 3)
        assert np.all((colors >= 0.0) & (colors <= 1.0))
        assert np.allclose(colors, np.clip(0.5 + sh_c0 * f_dc, 0.0, 1.0), atol=1e-6)

    def test_load_clips_saturated_dc_color_into_unit_range(self, tmp_path) -> None:
        # Large-magnitude SH DC terms would push 0.5 + SH_C0 * f_dc outside
        # [0, 1]; the loader must clip so the rasterizer never sees invalid RGB.
        pytest.importorskip("torch")
        pytest.importorskip("plyfile")
        from strands_robots.rendering.backgrounds import _load_ply_splats

        means = np.zeros((2, 3), np.float32)
        scales_log = np.zeros((2, 3), np.float32)
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (2, 1))
        opacity_logit = np.zeros(2, np.float32)
        f_dc = np.array([[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]], np.float32)
        path = tmp_path / "saturated.ply"
        _build_ply(
            path,
            means=means,
            scales_log=scales_log,
            quats=quats,
            opacity_logit=opacity_logit,
            f_dc=f_dc,
        )

        colors = _load_ply_splats(path, device="cpu")["colors"].numpy()
        assert np.allclose(colors[0], 1.0)
        assert np.allclose(colors[1], 0.0)


# --------------------------------------------------------------------------- #
# GsplatBackground config normalization + splat clipping (no gsplat/CUDA)
# --------------------------------------------------------------------------- #
#
# Constructing a GsplatBackground and clipping its gaussians are pure
# numpy/torch bookkeeping -- the heavy gsplat rasterizer is only touched lazily
# on the first ``render()``. These contracts (mode resolution, background-fill
# defaults, and transform-aware sub-floor/opacity clipping) are pinned here
# without needing the ``sim-gs`` extra.


class TestGsplatBackgroundConfig:
    """The GsplatBackground constructor normalizes its render-mode config."""

    def test_defaults_to_no_alignment_black_fill_and_lazy_splats(self) -> None:
        from strands_robots.rendering.backgrounds import GsplatBackground

        bg = GsplatBackground(ply_path="/nonexistent/scene.ply")

        # No skybox/backdrop alignment unless explicitly asked for.
        assert bg._skybox is False
        assert bg._auto_backdrop is False
        # Identity world_from_gs and no splats loaded yet (lazy on first render).
        assert np.allclose(bg._transform, np.eye(4))
        assert bg._explicit_transform is False
        assert bg._splats is None
        # Non-skybox default fill is black (voids read as black, matching the
        # dark scene background convention).
        assert bg._bg_fill.tolist() == [0.0, 0.0, 0.0]

    def test_skybox_mode_defaults_to_neutral_grey_void_fill(self) -> None:
        from strands_robots.rendering.backgrounds import GsplatBackground

        bg = GsplatBackground(ply_path="scene.ply", skybox=True)

        assert bg._skybox is True
        # Unobserved zenith/edges read as a light-grey ceiling/sky, not black.
        assert bg._bg_fill.tolist() == [188.0, 188.0, 192.0]

    def test_explicit_bg_fill_overrides_the_mode_default(self) -> None:
        from strands_robots.rendering.backgrounds import GsplatBackground

        bg = GsplatBackground(ply_path="scene.ply", skybox=True, bg_fill=(10, 20, 30))

        assert bg._bg_fill.tolist() == [10.0, 20.0, 30.0]

    def test_explicit_transform_disables_skybox_and_backdrop_fits(self) -> None:
        from strands_robots.rendering.backgrounds import GsplatBackground

        transform = np.eye(4)
        transform[0, 3] = 1.5
        bg = GsplatBackground(
            ply_path="scene.ply",
            transform=transform,
            skybox=True,
            auto_backdrop=True,
        )

        # A caller-supplied world_from_gs wins: no auto-fit clobbers it.
        assert bg._explicit_transform is True
        assert bg._skybox is False
        assert bg._auto_backdrop is False
        assert np.allclose(bg._transform, transform)

    def test_skybox_alignment_and_clip_parameters_are_captured(self) -> None:
        from strands_robots.rendering.backgrounds import GsplatBackground

        bg = GsplatBackground(
            ply_path="scene.ply",
            skybox=True,
            up_sign=-1.0,
            yaw_deg=30,
            radius=2.0,
            floor_z=-0.5,
            clip_below=0.1,
            min_opacity=0.3,
            floor_pct=1.5,
            metric=True,
            up_axis=(0, 0, 1),
            major_axis=(1, 0, 0),
            own_floor=True,
        )

        assert bg._up_sign == -1.0
        assert bg._yaw_deg == 30.0
        assert bg._radius == 2.0
        assert bg._floor_z == -0.5
        assert bg._clip_below == 0.1
        assert bg._min_opacity == 0.3
        assert bg._floor_pct == 1.5
        assert bg._metric is True
        assert bg._up_axis == (0, 0, 1)
        assert bg._major_axis == (1, 0, 0)
        # own_floor tells the compositor to hide the MuJoCo grid ground.
        assert bg.own_floor is True

    def test_backdrop_center_and_radius_are_captured(self) -> None:
        from strands_robots.rendering.backgrounds import GsplatBackground

        bg = GsplatBackground(
            ply_path="scene.ply",
            auto_backdrop=True,
            backdrop_center=(1.0, 2.0, 3.0),
            backdrop_radius=5.0,
        )

        assert bg._auto_backdrop is True
        assert bg._backdrop_center.tolist() == [1.0, 2.0, 3.0]
        assert bg._backdrop_radius == 5.0


class TestGsplatBackgroundClipSplats:
    """_clip_splats drops sub-floor gaussians and low-opacity floaters."""

    @staticmethod
    def _splats(means, opacities):
        import torch

        n = len(means)
        return {
            "means": torch.tensor(means, dtype=torch.float32),
            "opacities": torch.tensor(opacities, dtype=torch.float32).reshape(-1, 1),
            "colors": torch.zeros(n, 3),
            "scales": torch.ones(n, 3),
            "quats": torch.zeros(n, 4),
        }

    def test_clips_below_floor_and_low_opacity(self) -> None:
        pytest.importorskip("torch")
        from strands_robots.rendering.backgrounds import GsplatBackground

        bg = GsplatBackground(ply_path="scene.ply")
        bg._transform = np.eye(4)
        # z:   -1.0 (below floor), 0.5, 2.0, 3.0 ; opacity: 0.9, 0.1, 0.9, 0.9
        bg._splats = self._splats(
            [[0, 0, -1.0], [0, 0, 0.5], [0, 0, 2.0], [0, 0, 3.0]],
            [0.9, 0.1, 0.9, 0.9],
        )

        kept, total = bg._clip_splats(clip_below=0.0, min_opacity=0.25)

        # Kept iff world-z >= 0 AND opacity >= 0.25 -> only the z=2 and z=3 ones.
        assert total == 4
        assert kept == 2
        assert bg._splats["means"].tolist() == [[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        # Every parallel field is filtered to the same survivors.
        for key in ("means", "opacities", "colors", "scales", "quats"):
            assert bg._splats[key].shape[0] == 2

    def test_zero_min_opacity_disables_the_opacity_filter(self) -> None:
        pytest.importorskip("torch")
        from strands_robots.rendering.backgrounds import GsplatBackground

        bg = GsplatBackground(ply_path="scene.ply")
        bg._transform = np.eye(4)
        bg._splats = self._splats([[0, 0, -1.0], [0, 0, 1.0]], [0.01, 0.01])

        kept, total = bg._clip_splats(clip_below=0.0, min_opacity=0.0)

        # Faint floaters survive; only the sub-floor gaussian is dropped.
        assert (kept, total) == (1, 2)
        assert bg._splats["means"].tolist() == [[0.0, 0.0, 1.0]]

    def test_clip_threshold_is_applied_in_world_frame_after_transform(self) -> None:
        pytest.importorskip("torch")
        from strands_robots.rendering.backgrounds import GsplatBackground

        bg = GsplatBackground(ply_path="scene.ply")
        # world_from_gs lifts every gaussian by +5 in z, so gaussians that sit
        # below the floor in their own frame clear it in world coordinates.
        transform = np.eye(4)
        transform[2, 3] = 5.0
        bg._transform = transform
        bg._splats = self._splats([[0, 0, -1.0], [0, 0, -4.0]], [0.9, 0.9])

        kept, total = bg._clip_splats(clip_below=0.0, min_opacity=0.0)

        # gs-z -1 -> world 4.0 (kept); gs-z -4 -> world 1.0 (kept): the transform
        # is honored, so both clear the world-frame floor.
        assert (kept, total) == (2, 2)


class TestBakeGsplatPanorama:
    """bake_gsplat_panorama reprojects six outward cube faces into an
    equirectangular panorama using the same spherical convention
    ``PanoramaBackground`` samples with, so a baked backdrop reads correctly
    per-camera. The gaussian-splat render is the only CUDA-bound step; the
    cube-to-equirect reprojection and image write are pure numpy + Pillow and
    are pinned here without gsplat/CUDA by doubling the render boundary.
    """

    @staticmethod
    def _face_color(fwd) -> np.ndarray:
        """Map an outward face direction to a distinct RGB so a panorama pixel's
        colour identifies which world direction it sampled."""
        return ((np.asarray(fwd, dtype=float) + 1.0) * 0.5 * 255.0).astype(np.uint8)

    def test_reprojects_cube_faces_into_equirectangular_directions(self, tmp_path, monkeypatch) -> None:
        torch = pytest.importorskip("torch")
        from PIL import Image

        from strands_robots.rendering import backgrounds as bg

        # A room-like cloud so _upright_view_transform has a well-defined thin
        # (up) axis; it does not change the doubled render but exercises the
        # real pre-render setup (load + upright transform) in bake.
        rng = np.random.default_rng(0)
        means = np.column_stack(
            [rng.uniform(-4, 4, 2000), rng.uniform(-2, 2, 2000), rng.uniform(-0.1, 0.1, 2000)]
        ).astype(np.float32)
        face_color = self._face_color

        def fake_load(self) -> None:
            self._splats = {"means": torch.from_numpy(means)}

        def fake_render(self, cam):
            # bake builds T_world_cam columns [right, up, -fwd], so the outward
            # face direction is -Z of the camera rotation. Paint the whole face
            # that direction's colour to trace where each pixel is sampled from.
            fwd = -np.asarray(cam.T_world_cam, dtype=float)[:3, 2]
            rgb = np.empty((cam.height, cam.width, 3), np.uint8)
            rgb[:] = face_color(fwd)
            return rgb, np.zeros((cam.height, cam.width), np.float32)

        monkeypatch.setattr(bg.GsplatBackground, "_load", fake_load)
        monkeypatch.setattr(bg.GsplatBackground, "render", fake_render)

        # PNG keeps the reprojection lossless so exact direction->colour
        # assertions hold (production bakes a .jpg; the format is incidental).
        out = bg.bake_gsplat_panorama(
            tmp_path / "scene.ply",
            out_path=tmp_path / "pano.png",
            face_size=32,
            equi_w=64,
            equi_h=32,
            device="cpu",
        )

        assert out.exists() and out.stat().st_size > 0
        pano = np.array(Image.open(out))
        assert pano.shape == (32, 64, 3)
        assert pano.dtype == np.uint8

        # Equirect convention (matches PanoramaBackground): column W/2 -> theta 0
        # and row H/2 -> phi 0 -> +X; column 0 -> theta -pi -> -X; row 0 -> phi
        # +pi/2 -> +Z. Each sampled pixel must carry that face's colour.
        np.testing.assert_array_equal(pano[16, 32], self._face_color([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(pano[16, 0], self._face_color([-1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(pano[0, 32], self._face_color([0.0, 0.0, 1.0]))

    def test_returns_cached_panorama_without_rerendering(self, tmp_path, monkeypatch) -> None:
        # A non-empty output short-circuits: bake returns the cached path and
        # never touches the (GPU) splat load/render path.
        from strands_robots.rendering import backgrounds as bg

        out = tmp_path / "pano.jpg"
        out.write_bytes(b"cached-panorama")

        def boom(self) -> None:
            raise AssertionError("bake must not load splats when a cached panorama exists")

        monkeypatch.setattr(bg.GsplatBackground, "_load", boom)

        result = bg.bake_gsplat_panorama(tmp_path / "scene.ply", out_path=out, device="cpu")

        assert result == out
        assert out.read_bytes() == b"cached-panorama"

    # ----- the cache must answer only the request it was baked for ----- #
    #
    # Baking is six splat renders, so a warm output short-circuits the pass.
    # The short-circuit is keyed on the output *path*, so the default path has
    # to identify the panorama it holds: ``equi_w`` / ``equi_h`` / ``face_size``
    # all change the pixels, and a path that ignores them makes the second
    # caller's geometry a no-op.

    def _stub_render_boundary(self, monkeypatch, tmp_path):
        """Double the CUDA-bound splat boundary; return the per-face render log."""
        torch = pytest.importorskip("torch")

        from strands_robots.rendering import backgrounds as bg

        rng = np.random.default_rng(0)
        means = np.column_stack([rng.uniform(-4, 4, 512), rng.uniform(-2, 2, 512), rng.uniform(-0.1, 0.1, 512)]).astype(
            np.float32
        )
        rendered: list[tuple[int, int]] = []

        def fake_load(inner_self) -> None:
            inner_self._splats = {"means": torch.from_numpy(means)}

        def fake_render(inner_self, cam):
            rendered.append((cam.width, cam.height))
            return (
                np.full((cam.height, cam.width, 3), 128, np.uint8),
                np.zeros((cam.height, cam.width), np.float32),
            )

        monkeypatch.setattr(bg.GsplatBackground, "_load", fake_load)
        monkeypatch.setattr(bg.GsplatBackground, "render", fake_render)
        ply = tmp_path / "scene.ply"
        ply.write_bytes(b"placeholder")
        return bg, ply, rendered

    @staticmethod
    def _panorama_size(path) -> tuple[int, int]:
        """Return the (width, height) of a baked panorama, closing the file."""
        from PIL import Image

        with Image.open(path) as img:
            return img.size

    def test_a_new_equirect_resolution_is_rendered_not_served_from_the_cache(self, tmp_path, monkeypatch) -> None:
        # The headline defect: bake small, then ask for a bigger panorama. The
        # second request used to return the first call's small image with zero
        # renders, so the caller silently got a backdrop at a resolution it had
        # not asked for (and then sampled that as if it were the big one).
        bg, ply, rendered = self._stub_render_boundary(monkeypatch, tmp_path)

        first = bg.bake_gsplat_panorama(ply, face_size=16, equi_w=64, equi_h=32, device="cpu")
        first_size = self._panorama_size(first)
        assert first_size == (64, 32)

        rendered.clear()
        second = bg.bake_gsplat_panorama(ply, face_size=16, equi_w=128, equi_h=64, device="cpu")
        second_size = self._panorama_size(second)
        first_size_after = self._panorama_size(first)

        assert second_size == (128, 64), "the requested equirect size must be honored"
        assert rendered, "a new resolution must re-render rather than reuse the cached panorama"
        assert second != first, "distinct requests must not collide on one cache path"
        # The first panorama is still intact for whoever asked for that size.
        assert first_size_after == (64, 32)

    def test_a_new_face_size_is_rendered_not_served_from_the_cache(self, tmp_path, monkeypatch) -> None:
        # face_size changes the sharpness of the cube faces the panorama is
        # reprojected from, not the output dimensions -- so a cache keyed on the
        # output size alone would still wrongly serve the coarser bake.
        bg, ply, rendered = self._stub_render_boundary(monkeypatch, tmp_path)

        bg.bake_gsplat_panorama(ply, face_size=16, equi_w=64, equi_h=32, device="cpu")

        rendered.clear()
        bg.bake_gsplat_panorama(ply, face_size=64, equi_w=64, equi_h=32, device="cpu")

        assert rendered, "a new face_size must re-render rather than reuse the cached panorama"
        assert {size for size in rendered} == {(64, 64)}, "the requested face_size must be honored"

    def test_repeating_one_request_reuses_the_cached_panorama(self, tmp_path, monkeypatch) -> None:
        # Control: the cache still exists. Asking twice for the same geometry
        # renders once, so the expensive path is not re-run per call.
        bg, ply, rendered = self._stub_render_boundary(monkeypatch, tmp_path)

        first = bg.bake_gsplat_panorama(ply, face_size=16, equi_w=64, equi_h=32, device="cpu")
        assert rendered, "the first bake must render"

        rendered.clear()
        second = bg.bake_gsplat_panorama(ply, face_size=16, equi_w=64, equi_h=32, device="cpu")

        assert second == first
        assert not rendered, "an identical request must be served from the cache"


# --------------------------------------------------------------------------- #
# GsplatBackground._load -- splat load + alignment wiring (no gsplat/CUDA)
# --------------------------------------------------------------------------- #
#
# _load() gates on the ``sim-gs`` extra (torch + gsplat) and then, for a
# .spz/.ply scene, decodes the splats and -- in skybox / auto-backdrop mode --
# fits the ``world_from_gs`` transform and drops out-of-bounds gaussians. The
# gsplat *rasterizer* is CUDA-bound and only touched later in ``render()``, so
# the decode + alignment wiring is exercised here on CPU with the optional-dep
# gate stubbed out. A room-like ``.spz`` (wide X, deep Y, thin Z) gives the PCA
# up-axis a well-defined direction so the fit is meaningful.


def _room_spz(tmp_path, n: int = 1500, seed: int = 0):
    """Write a room-like scene as a minimal v2 ``.spz`` and return (path, means)."""
    rng = np.random.default_rng(seed)
    means = np.column_stack([rng.uniform(-4.0, 4.0, n), rng.uniform(-2.0, 2.0, n), rng.uniform(-0.1, 0.1, n)]).astype(
        np.float32
    )
    alpha = np.full(n, 255, np.uint8)
    col = np.full((n, 3), 128, np.uint8)
    scl = np.full((n, 3), 128, np.uint8)
    rot = np.full((n, 3), 128, np.uint8)  # v2 stores 3 bytes/point
    path = _build_spz(tmp_path, means, alpha, col, scl, rot, version=2)
    return path, means


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine ``world_from_gs`` to ``(N, 3)`` points."""
    return pts @ T[:3, :3].T + T[:3, 3]


class TestGsplatBackgroundLoad:
    """_load decodes a scene and, per mode, fits the world_from_gs transform."""

    def test_skybox_mode_seats_the_scene_floor_and_keeps_permissive_splats(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        from strands_robots.rendering import backgrounds as bg

        # Stub the sim-gs optional-dep gate: we exercise the CPU decode + fit,
        # not the CUDA rasterizer.
        monkeypatch.setattr(bg, "require_optional", lambda *a, **k: None)
        path, means = _room_spz(tmp_path)

        floor_z = -0.3
        b = bg.GsplatBackground(
            ply_path=path,
            device="cpu",
            skybox=True,
            up_sign=1.0,
            floor_z=floor_z,
            floor_pct=2.0,
            radius=2.5,
            # A permissive world-z floor keeps every gaussian, so the clip
            # branch runs but drops nothing.
            clip_below=-100.0,
            min_opacity=0.0,
            up_axis=(0.0, 0.0, 1.0),
            major_axis=(1.0, 0.0, 0.0),
        )
        b._load()

        # A skybox transform was fit (not left at identity).
        assert not np.allclose(b._transform, np.eye(4))
        # The fit seats the floor_pct percentile of world-z at floor_z -- this
        # is what puts the photoreal ground under the arm.
        world = _apply(b._transform, means)
        assert np.isclose(np.percentile(world[:, 2], 2.0), floor_z, atol=1e-2)
        # ...and scales the horizontal 95th-percentile extent to ``radius``.
        horiz = np.linalg.norm(world[:, :2] - world[:, :2].mean(axis=0), axis=1)
        assert np.isclose(np.percentile(horiz, 95), 2.5, rtol=0.05)
        # The permissive clip retained every decoded gaussian.
        assert b._splats is not None
        assert b._splats["means"].detach().cpu().numpy().shape[0] == means.shape[0]

    def test_skybox_clip_drops_gaussians_below_the_world_floor(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        from strands_robots.rendering import backgrounds as bg

        monkeypatch.setattr(bg, "require_optional", lambda *a, **k: None)
        path, means = _room_spz(tmp_path)

        # A world-z clip at 0 with the floor seated at -0.3 drops the sub-floor
        # slab, so fewer gaussians survive than were decoded.
        b = bg.GsplatBackground(
            ply_path=path,
            device="cpu",
            skybox=True,
            up_sign=1.0,
            floor_z=-0.3,
            clip_below=0.0,
            min_opacity=0.0,
            up_axis=(0.0, 0.0, 1.0),
            major_axis=(1.0, 0.0, 0.0),
        )
        b._load()

        assert b._splats is not None
        kept = b._splats["means"].detach().cpu().numpy().shape[0]
        assert 0 <= kept < means.shape[0]

    def test_auto_backdrop_mode_centres_scene_on_backdrop_center(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        from strands_robots.rendering import backgrounds as bg

        monkeypatch.setattr(bg, "require_optional", lambda *a, **k: None)
        path, means = _room_spz(tmp_path)

        center = (0.05, 0.05, 0.25)
        b = bg.GsplatBackground(
            ply_path=path,
            device="cpu",
            auto_backdrop=True,
            backdrop_center=center,
            backdrop_radius=3.0,
        )
        b._load()

        # The backdrop fit maps the scene centroid exactly onto backdrop_center.
        assert not np.allclose(b._transform, np.eye(4))
        assert np.allclose(_apply(b._transform, means.mean(axis=0)), center, atol=5e-3)
        # No clip is applied in backdrop mode -- every gaussian is retained.
        assert b._splats is not None
        assert b._splats["means"].detach().cpu().numpy().shape[0] == means.shape[0]

    def test_missing_scene_file_raises_filenotfound(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        from strands_robots.rendering import backgrounds as bg

        monkeypatch.setattr(bg, "require_optional", lambda *a, **k: None)
        b = bg.GsplatBackground(ply_path=tmp_path / "does_not_exist.spz", device="cpu")
        with pytest.raises(FileNotFoundError, match="Gaussian Splat not found"):
            b._load()
