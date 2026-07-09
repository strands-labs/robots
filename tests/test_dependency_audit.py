"""Supply-chain audit regression tests for declared dependencies.

Pins the guard added after a dependency-confusion finding: the ``mimicgen``
distribution name on PyPI is not NVlabs MimicGen (which has never published to
PyPI), so it must never appear as a PyPI-sourced dependency. These tests verify
both the live ``pyproject.toml`` and that the reusable audit in
``scripts/audit_deps.py`` actually catches a re-introduction.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_AUDIT_PATH = _REPO_ROOT / "scripts" / "audit_deps.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_deps", _AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["audit_deps"] = module
    spec.loader.exec_module(module)
    return module


audit_deps = _load_audit_module()


def test_pyproject_has_no_denylisted_pypi_dependency():
    """The live pyproject must not pin any denylisted name from PyPI."""
    findings = audit_deps.audit(_PYPROJECT, check_pypi=False)
    assert findings == [], f"dependency audit reported: {findings}"


def test_mimicgen_is_not_a_pypi_dependency():
    """mimicgen must not be a PyPI-sourced dependency (confusion vector)."""
    deps = audit_deps.collect_pypi_dependencies(_PYPROJECT)
    assert "mimicgen" not in deps


def test_mimicgen_stays_denylisted():
    """The guard's denylist must retain mimicgen so re-adds are blocked."""
    assert "mimicgen" in audit_deps.DENYLIST


def test_audit_flags_reintroduced_mimicgen(tmp_path):
    """Re-adding mimicgen==1.0.0 must make the denylist audit fail."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'name = "x"\n'
        'version = "0"\n'
        'dependencies = ["numpy>=1.24"]\n'
        "[project.optional-dependencies]\n"
        'vera-sim = ["mimicgen==1.0.0", "mujoco>=3.5.0"]\n',
        encoding="utf-8",
    )
    findings = audit_deps.audit(pyproject, check_pypi=False)
    assert any("mimicgen" in f.lower() for f in findings)


def test_git_and_self_reference_deps_are_excluded(tmp_path):
    """Git-sourced and self-referencing extras are not treated as PyPI deps."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'name = "x"\n'
        'version = "0"\n'
        'dependencies = ["numpy>=1.24"]\n'
        "[project.optional-dependencies]\n"
        'vera = ["vera @ git+https://github.com/sizhe-li/VERA.git"]\n'
        'all = ["x[vera]"]\n',
        encoding="utf-8",
    )
    deps = audit_deps.collect_pypi_dependencies(pyproject)
    assert deps == {"numpy": "numpy>=1.24"}


def test_denylist_names_are_canonicalized(tmp_path):
    """A denylisted name is caught regardless of case / dash-underscore form."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0"\ndependencies = ["MimicGen==1.0.0"]\n',
        encoding="utf-8",
    )
    findings = audit_deps.audit(pyproject, check_pypi=False)
    assert findings, "canonicalized denylist match should fire"


def test_pyproject_has_no_direct_reference_dependency():
    """The live pyproject must declare no PEP 508 direct-reference dependency.

    A ``name @ <url>`` requirement (git/URL/file) makes the PyPI upload endpoint
    reject the distribution even though the wheel builds and passes ``twine
    check`` -- this is what failed the v0.4.1 publish. Git-only dependencies must
    be documented as a manual install, never declared as a dependency or extra.
    """
    findings = audit_deps.check_direct_references(_PYPROJECT)
    assert findings == [], f"direct-reference dependency found: {findings}"


def test_audit_flags_direct_reference_dependency(tmp_path):
    """A re-introduced ``git+`` dependency must make the audit fail."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'name = "x"\n'
        'version = "0"\n'
        'dependencies = ["numpy>=1.24"]\n'
        "[project.optional-dependencies]\n"
        'vera = ["vera @ git+https://github.com/sizhe-li/VERA.git"]\n',
        encoding="utf-8",
    )
    findings = audit_deps.audit(pyproject, check_pypi=False)
    assert any("DIRECT REFERENCE" in f and "vera" in f for f in findings), findings


def test_direct_reference_check_ignores_extras_specifiers_and_markers(tmp_path):
    """Self-referencing extras, version specifiers and markers are not flagged."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'name = "x"\n'
        'version = "0"\n'
        'dependencies = ["numpy>=1.24", "torch>=2.0; platform_machine == \'aarch64\'"]\n'
        "[project.optional-dependencies]\n"
        'all = ["x[dev]"]\n'
        'dev = ["pytest>=8"]\n',
        encoding="utf-8",
    )
    assert audit_deps.check_direct_references(pyproject) == []


# ---------------------------------------------------------------------------
# lerobot 0.6 floor + torch/torchvision override-removal invariant.
#
# strands_robots used to carry per-platform torch/torchvision/torchcodec
# overrides in the ``[lerobot]`` extra and ``[tool.uv].override-dependencies``
# to compensate for lerobot 0.5.1's deficient dependency markers (its torch<2.11
# cap that skipped the NVIDIA Thor/Jetson sm_110 cuBLAS fix, and its torchcodec
# marker that excluded linux aarch64, leaving Thor/Jetson with no video decoder).
# lerobot 0.6 fixed those markers upstream: torch>=2.7,<2.12 with a
# ``torchcodec>=0.11,<0.12`` aarch64 marker that pulls the ABI-matched torch 2.11
# on every platform. Requiring lerobot >= 0.6.0 is therefore what lets those
# overrides be dropped: the codec/decoder stack now resolves ABI-consistently
# (torch 2.11 + torchcodec 0.11.x + torchvision 0.26) on linux x86_64/aarch64 and
# macOS arm64 with no strands override.
#
# These two invariants are coupled: reverting the lerobot floor below 0.6 WITHOUT
# restoring the overrides would silently break the video decoder on aarch64/macOS,
# and re-adding a torch<2.11 override would conflict with lerobot 0.6's torch 2.11
# resolution. This guard fails if either half regresses.
import tomllib  # noqa: E402

from packaging.requirements import Requirement  # noqa: E402
from packaging.version import Version  # noqa: E402


def _lerobot_extra_requirement() -> Requirement:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    extra = data["project"]["optional-dependencies"]["lerobot"]
    for spec in extra:
        req = Requirement(spec)
        if req.name == "lerobot":
            return req
    raise AssertionError("no `lerobot` requirement found in the [lerobot] extra")


def test_lerobot_extra_requires_at_least_0_6() -> None:
    """The ``[lerobot]`` extra must floor lerobot at >= 0.6.0.

    The 0.5.1-era torch/torchcodec overrides were removed because lerobot 0.6's
    own markers resolve the decoder stack correctly; that only holds for
    lerobot >= 0.6, so the floor must not regress below it.
    """
    req = _lerobot_extra_requirement()
    assert Version("0.6.0") in req.specifier, f"lerobot floor must admit 0.6.0, got {req.specifier}"
    assert Version("0.5.9") not in req.specifier, (
        f"lerobot floor must exclude 0.5.x (the overrides that compensated for "
        f"lerobot 0.5.1's decoder markers were removed), got {req.specifier}"
    )


def test_no_torch_or_torchvision_uv_override() -> None:
    """No ``torch``/``torchvision`` pin may live in ``[tool.uv].override-dependencies``.

    lerobot 0.6 resolves torch 2.11 + torchvision 0.26 (the ABI-matched pair, and
    the torch build that fixes the Thor sm_110 cuBLAS bug) on every platform
    unaided. A strands ``torch``/``torchvision`` override -- in particular a
    ``torch<2.11`` cap like the 0.5.1-era one -- would conflict with that
    resolution, so it must stay removed. (The diffusers security-floor override
    is unrelated and intentionally retained.)
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    overrides = data.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    offenders = [o for o in overrides if Requirement(o).name in ("torch", "torchvision")]
    assert not offenders, (
        "torch/torchvision uv overrides must stay removed (they compensated for "
        f"lerobot 0.5.1 and conflict with lerobot 0.6's torch 2.11): {offenders}"
    )
