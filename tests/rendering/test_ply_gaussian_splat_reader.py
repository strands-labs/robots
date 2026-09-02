# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``.ply`` reader must decode the standard 3DGS layout into the splat dict.

``.ply`` is the primary documented input for :class:`GsplatBackground` -- every
downloadable scene preset in ``GSPLAT_SCENES`` ships as one, and the INRIA
layout it uses is what Nerfstudio, Polycam and Marble export. Its ``.spz``
sibling reader is pinned field by field; this reader had no cells at all, and
its two decisions are both silent when wrong:

* geometry is stored **pre-activation** -- ``scale_*`` in log space and
  ``opacity`` as a logit -- so a reader that forwards the stored numbers hands
  the rasterizer sub-millimetre gaussians at ~50% alpha and renders a haze;
* higher-order spherical harmonics are flattened **channel-major**, so
  ``f_rest_{c * (K - 1) + j}`` is channel ``c``, coefficient ``j``. Reading that
  block coefficient-major transposes colour against direction, which shows up
  only as view-dependent colour that moves the wrong way.

The reader's output shape is also a contract with its consumer:
:meth:`GsplatBackground.render` derives ``sh_degree`` from
``colors.shape[1]``, and treats a 2-D ``colors`` as baked RGB with no
coefficients to evaluate. Both branches are pinned here against the degree the
file declared.

``plyfile`` ships only in the GPU-only ``sim-gs`` extra, which ``[all]`` does
not pull, so a cell gated on importing it does not run where the suite is
installed -- which is how a decode this specific came to be graded only where
someone had already installed a CUDA rasterizer. The seam stood in here is the
narrowest one that exists: the tests write a real ``binary_little_endian`` 3DGS
PLY -- including the ``nx ny nz`` normals INRIA writes and this reader ignores,
so a positional read of the property block would not survive -- and the
stand-in does the single thing ``plyfile`` does for this caller, which is to
hand back the vertex element as a numpy structured array with one field per
declared property.

A double is only worth as much as its faithfulness, so that is a cell too:
:func:`test_the_stand_in_reads_what_plyfile_reads` parses the same bytes both
ways and compares the arrays column by column. It is the one cell here that
needs the extra, and it is the reason the rest do not.
"""

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# SH band 0 basis function; the DC term bakes to RGB as ``0.5 + SH_C0 * dc``
# (graphdeco-inria/gaussian-splatting).
SH_C0 = 0.28209479177387814


class _VertexElement:
    """Stands in for ``plyfile.PlyElement``: carries the structured array."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class _PlyData:
    """Stands in for ``plyfile.PlyData`` over ``binary_little_endian`` floats."""

    @staticmethod
    def read(path: str) -> dict[str, _VertexElement]:
        raw = Path(path).read_bytes()
        head, body = raw.split(b"end_header\n", 1)
        lines = head.decode("ascii").splitlines()
        count = next(int(line.split()[2]) for line in lines if line.startswith("element vertex "))
        names = [line.split()[2] for line in lines if line.startswith("property float ")]
        data = np.frombuffer(body, np.dtype([(name, "<f4") for name in names]), count=count)
        return {"vertex": _VertexElement(data)}


@pytest.fixture
def ply_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``plyfile`` importable, for this test only.

    Both the direct ``from plyfile import PlyData`` and ``require_optional``'s
    memo are covered, so the stand-in cannot outlive the test through the
    cache.
    """
    from strands_robots import utils as sr_utils

    module = types.ModuleType("plyfile")
    module.PlyData = _PlyData  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "plyfile", module)
    monkeypatch.setitem(sr_utils._lazy_modules, "plyfile", module)


@pytest.fixture
def gsplat_stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy ``_load``'s ``gsplat`` requirement without the CUDA extra.

    ``_load`` requires the module and never calls into it -- rasterization is
    ``render``'s job -- so an empty module is the whole of what it reads.
    """
    from strands_robots import utils as sr_utils

    module = types.ModuleType("gsplat")
    monkeypatch.setitem(sys.modules, "gsplat", module)
    monkeypatch.setitem(sr_utils._lazy_modules, "gsplat", module)


def _write_3dgs_ply(
    path: Path,
    *,
    means: np.ndarray,
    log_scales: np.ndarray,
    logit_opacity: np.ndarray,
    dc: np.ndarray,
    quats: np.ndarray,
    rest: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write an INRIA-layout 3DGS PLY, properties in the order INRIA emits them.

    Args:
        path: Destination ``.ply``.
        means: ``(N, 3)`` positions.
        log_scales: ``(N, 3)`` log-space scales, as stored.
        logit_opacity: ``(N,)`` opacity logits, as stored.
        dc: ``(N, 3)`` SH DC coefficients.
        quats: ``(N, 4)`` rotations in the stored ``w x y z`` order.
        rest: Optional ``f_rest_<i>`` columns, higher-order SH.

    Returns:
        *path*, written.
    """
    count = len(means)
    fields: dict[str, np.ndarray] = {
        "x": means[:, 0],
        "y": means[:, 1],
        "z": means[:, 2],
        # Normals: INRIA writes them as zeros and this reader never reads them.
        "nx": np.zeros(count),
        "ny": np.zeros(count),
        "nz": np.zeros(count),
        "f_dc_0": dc[:, 0],
        "f_dc_1": dc[:, 1],
        "f_dc_2": dc[:, 2],
        **(rest or {}),
        "opacity": logit_opacity,
        "scale_0": log_scales[:, 0],
        "scale_1": log_scales[:, 1],
        "scale_2": log_scales[:, 2],
        "rot_0": quats[:, 0],
        "rot_1": quats[:, 1],
        "rot_2": quats[:, 2],
        "rot_3": quats[:, 3],
    }
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n"
    header += "".join(f"property float {name}\n" for name in fields)
    header += "end_header\n"
    record = np.zeros(count, dtype=[(name, "<f4") for name in fields])
    for name, column in fields.items():
        record[name] = column
    path.write_bytes(header.encode("ascii") + record.tobytes())
    return path


def _dc_only_scene(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """A two-gaussian DC-only asset plus the stored (pre-activation) columns."""
    means = np.array([[0.5, 0.25, -0.75], [-1.0, 2.0, 3.0]], np.float32)
    log_scales = np.array([[0.0, np.log(2.0), -1.0], [1.0, 0.0, 0.5]], np.float32)
    logit_opacity = np.array([0.0, 2.0], np.float32)
    # Row 0 saturates the RGB bake at both ends; row 1 lands inside it.
    dc = np.array([[0.0, 2.0, -3.0], [0.1, 0.2, 0.3]], np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]], np.float32)
    path = _write_3dgs_ply(
        tmp_path / "scene.ply",
        means=means,
        log_scales=log_scales,
        logit_opacity=logit_opacity,
        dc=dc,
        quats=quats,
    )
    return path, {
        "means": means,
        "log_scales": log_scales,
        "logit_opacity": logit_opacity,
        "dc": dc,
        "quats": quats,
    }


def _sh_scene(tmp_path: Path, degree: int) -> tuple[Path, np.ndarray, int]:
    """A two-gaussian asset with higher-order SH, one value per (channel, coeff).

    Encoding the pair in the value means a block read the other way round is
    visible as a transpose rather than merely as a wrong number.

    Args:
        tmp_path: Directory to write into.
        degree: SH degree; the rest block then holds ``(degree + 1) ** 2 - 1``
            coefficients per channel.

    Returns:
        ``(path, dc, n_rest)``.
    """
    n_rest = (degree + 1) ** 2 - 1
    rest = {
        f"f_rest_{channel * n_rest + coeff}": np.full(2, 100.0 * channel + coeff, np.float32)
        for channel in range(3)
        for coeff in range(n_rest)
    }
    dc = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]], np.float32)
    path = _write_3dgs_ply(
        tmp_path / f"scene_deg{degree}.ply",
        means=np.zeros((2, 3), np.float32),
        log_scales=np.zeros((2, 3), np.float32),
        logit_opacity=np.zeros(2, np.float32),
        dc=dc,
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (2, 1)),
        rest=rest,
    )
    return path, dc, n_rest


def test_a_dc_only_asset_bakes_the_dc_term_to_rgb_and_activates_the_geometry(tmp_path: Path, ply_reader: None) -> None:
    pytest.importorskip("torch")
    from strands_robots.rendering.backgrounds import _load_ply_splats

    path, stored = _dc_only_scene(tmp_path)

    splats = _load_ply_splats(path, device="cpu")

    assert set(splats) == {"means", "scales", "quats", "opacities", "colors"}
    assert np.allclose(splats["means"].numpy(), stored["means"])
    # ``rot_0..rot_3`` is stored w-first, which is the order gsplat wants.
    assert np.allclose(splats["quats"].numpy(), stored["quats"])
    # Both activations: exp for the log-space scale, sigmoid for the logit.
    assert np.allclose(splats["scales"].numpy(), np.exp(stored["log_scales"]), atol=1e-6)
    expected_alpha = 1.0 / (1.0 + np.exp(-stored["logit_opacity"]))
    assert np.allclose(splats["opacities"].numpy(), expected_alpha, atol=1e-6)
    # No ``antialiased`` key: that flag is an SPZ header bit, and its absence is
    # what leaves ``GsplatBackground`` in the classic rasterize mode.
    assert "antialiased" not in splats

    colors = splats["colors"].numpy()
    # 2-D colours are the fast path: baked RGB, nothing left to evaluate.
    assert colors.shape == (2, 3)
    assert np.allclose(colors, np.clip(0.5 + SH_C0 * stored["dc"], 0.0, 1.0), atol=1e-6)
    # The clip is load-bearing, not defensive: this asset reaches both bounds.
    assert (colors[0, 1], colors[0, 2]) == (1.0, 0.0)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_a_higher_order_asset_returns_raw_sh_with_the_dc_coefficient_first(
    tmp_path: Path, ply_reader: None, degree: int
) -> None:
    pytest.importorskip("torch")
    from strands_robots.rendering.backgrounds import _load_ply_splats

    path, dc, n_rest = _sh_scene(tmp_path, degree)

    colors = _load_ply_splats(path, device="cpu")["colors"].numpy()

    assert colors.shape == (2, n_rest + 1, 3)
    # DC first and unbaked: the rasterizer's SH evaluation applies the basis.
    assert np.allclose(colors[:, 0, :], dc)
    expected = np.array([[100.0 * channel + coeff for channel in range(3)] for coeff in range(n_rest)])
    assert np.allclose(colors[0, 1:, :], expected)
    # ``render`` reads the degree back out of the coefficient count.
    assert int(np.sqrt(colors.shape[1])) - 1 == degree


@pytest.mark.parametrize(
    ("n_rest_fields", "match"),
    [
        (4, "not divisible by 3"),
        (6, "does not match any degree"),
    ],
)
def test_a_rest_block_the_layout_cannot_explain_is_refused(
    tmp_path: Path, ply_reader: None, n_rest_fields: int, match: str
) -> None:
    pytest.importorskip("torch")
    from strands_robots.rendering.backgrounds import _load_ply_splats

    path = _write_3dgs_ply(
        tmp_path / "scene.ply",
        means=np.zeros((1, 3), np.float32),
        log_scales=np.zeros((1, 3), np.float32),
        logit_opacity=np.zeros(1, np.float32),
        dc=np.zeros((1, 3), np.float32),
        quats=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        rest={f"f_rest_{i}": np.zeros(1, np.float32) for i in range(n_rest_fields)},
    )

    with pytest.raises(ValueError, match=match):
        _load_ply_splats(path, device="cpu")


def test_a_ply_scene_reaches_the_ply_reader_and_stays_in_the_classic_mode(
    tmp_path: Path, ply_reader: None, gsplat_stand_in: None
) -> None:
    pytest.importorskip("torch")
    from strands_robots.rendering import GsplatBackground

    path, stored = _dc_only_scene(tmp_path)
    background = GsplatBackground(path, device="cpu")

    background._load()

    # Suffix dispatch: anything that is not ``.spz`` is read as a PLY.
    assert background._splats is not None
    assert np.allclose(background._splats["means"].numpy(), stored["means"])
    # ``_rasterize_mode`` follows the SPZ antialiased header bit, which a PLY
    # does not carry; the AA opacity compensation must not be applied to an
    # asset that was not trained with it.
    assert background._rasterize_mode == "classic"


def test_a_cuda_device_without_a_cuda_capable_torch_is_refused_before_the_scene_is_read(
    tmp_path: Path, ply_reader: None, gsplat_stand_in: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from strands_robots.rendering import GsplatBackground

    path, _ = _dc_only_scene(tmp_path)
    background = GsplatBackground(path, device="cuda")

    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        background._load()

    # Refused up front, where the caller chose the device -- not after the
    # decode, and not deep inside the first frame's rasterization.
    assert background._splats is None


def test_an_unreadable_scene_is_refused_where_the_caller_named_it(tmp_path: Path) -> None:
    from strands_robots.rendering import GsplatBackground

    path, _ = _dc_only_scene(tmp_path)
    path.chmod(0o000)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("the file mode does not bind for this user")
        with pytest.raises(PermissionError, match="not readable"):
            GsplatBackground(path, device="cpu")
    finally:
        path.chmod(0o644)


@pytest.mark.parametrize("degree", [0, 2])
def test_the_stand_in_reads_what_plyfile_reads(tmp_path: Path, degree: int) -> None:
    """Whatever ``plyfile`` parses out of these bytes, the stand-in parses too.

    This is what lets the cells above stand for the real dependency: a double
    that answered a different array would make every one of them agree with
    nothing. Covers both property blocks, since the higher-order one is the
    only place the field *count* varies.
    """
    plyfile = pytest.importorskip("plyfile")

    path = _dc_only_scene(tmp_path)[0] if degree == 0 else _sh_scene(tmp_path, degree)[0]
    real = plyfile.PlyData.read(str(path))["vertex"].data
    stood_in = _PlyData.read(str(path))["vertex"].data

    assert real.dtype.names == stood_in.dtype.names
    assert len(real) == len(stood_in)
    for name in real.dtype.names:
        assert np.array_equal(real[name], stood_in[name]), name
