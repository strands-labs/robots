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
