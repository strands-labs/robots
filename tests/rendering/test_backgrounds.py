# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Background renderer contracts (panorama path: zero ML deps)."""

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
