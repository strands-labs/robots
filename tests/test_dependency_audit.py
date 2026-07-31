"""Supply-chain audit regression tests for declared dependencies.

Pins the guard added after a dependency-confusion finding: the ``mimicgen``
distribution name on PyPI is not NVlabs MimicGen (which has never published to
PyPI), so it must never appear as a PyPI-sourced dependency. These tests verify
both the live ``pyproject.toml`` and that the reusable audit in
``scripts/audit_deps.py`` actually catches a re-introduction.
"""

from __future__ import annotations

import ast
import importlib.util
import re
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

import pytest  # noqa: E402
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


# ---------------------------------------------------------------------------
# ruff minor-cap invariant.
#
# ruff is a <1.0 tool, so per the dependency-bound convention it must be capped
# at the minor (`>=X.Y,<X.(Y+1)`), the same way lerobot is (`>=0.5.0,<0.6.0`).
# It regressed to a `<1.0.0` (major) cap, which admitted ruff 0.16.0. That
# release made python-code-block formatting inside Markdown a stable default,
# so `ruff format --check strands_robots tests tests_integ` began reformatting
# pre-existing docs and turned CI red with no source change. Capping the minor
# makes the CI lint toolchain deterministic: a formatter minor bump is adopted
# deliberately (by raising the cap), never silently.


def _ruff_requirements() -> list[Requirement]:
    """Every declared ``ruff`` requirement across the pyproject lint surfaces.

    Covers the ``dev`` extra and the ``[tool.hatch.envs.default]`` deps -- the
    two places the lint/format toolchain is pinned. Both feed CI, so both must
    stay minor-capped and in lockstep.
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    specs: list[str] = list(data["project"]["optional-dependencies"]["dev"])
    specs += list(data["tool"]["hatch"]["envs"]["default"]["dependencies"])
    reqs = [Requirement(s) for s in specs if Requirement(s).name == "ruff"]
    assert reqs, "no `ruff` requirement found in pyproject"
    return reqs


def test_ruff_is_minor_capped() -> None:
    """ruff must be minor-capped so a formatter minor bump cannot silently break CI.

    ruff 0.16.0 turned Markdown code-block formatting into a default, which
    reformatted pre-existing docs under ``ruff format --check`` and reddened CI
    with no code change. A ``<1.0.0`` (major) cap let that minor slip in; the
    convention for a ``<1.0`` dep is to cap the minor.
    """
    for req in _ruff_requirements():
        assert Version("0.16.0") not in req.specifier, (
            f"ruff must be minor-capped below 0.16.0 (its Markdown-format default "
            f"reddened CI with no source change); got {req.specifier}"
        )
        assert Version("0.15.12") in req.specifier, (
            f"ruff bound must still admit the validated 0.15.x line; got {req.specifier}"
        )


def test_ruff_bound_is_consistent_across_pyproject() -> None:
    """The ruff bound must be identical in every declaration so they never drift."""
    specs = {str(req.specifier) for req in _ruff_requirements()}
    assert len(specs) == 1, f"ruff bound must be identical across pyproject, got {specs}"


# ---------------------------------------------------------------------------
# Phantom `==` version-pin guard + the vera-sim / lerobot-0.6 fork invariant.
#
# `robomimic==0.5.0` was pinned in the [vera-sim] extra, but robomimic's highest
# PyPI release is 0.3.0 -- v0.5.0 exists only as an ARISE-Initiative GitHub tag.
# The pin was thus unresolvable forever (it wedged `uv lock`, freezing uv.lock at
# a months-old lerobot 0.5.1 resolution and hiding vla_jepa/molmoact2/lerobot.rl)
# AND a dependency-confusion vector (whoever publishes robomimic 0.5.0 to PyPI
# gets installed). A phantom `==` version differs from a nonexistent NAME, so the
# name-existence audit missed it; check_pinned_versions_exist closes that gap.
#
# Separately, [vera-sim] pins gymnasium==0.29.1, mutually exclusive with
# lerobot>=0.6.0 (gymnasium>=1.1.1). uv resolves all extras jointly, so absent a
# fork declaration that pin drags the WHOLE resolution below lerobot 0.6. The
# [tool.uv].conflicts entries fork vera-sim away from the lerobot-0.6 extras.


def test_pinned_version_check_flags_phantom_version():
    """A `name==X` pin whose version is absent from PyPI must be flagged."""
    findings = audit_deps.check_pinned_versions_exist(
        {"robomimic": "robomimic==0.5.0"},
        version_fetcher=lambda _name: {"0.1.0", "0.2.0", "0.3.0"},
    )
    assert any("PHANTOM VERSION" in f and "0.5.0" in f for f in findings), findings


def test_pinned_version_check_passes_for_existing_version():
    """An exact pin to a real release must not be flagged."""
    findings = audit_deps.check_pinned_versions_exist(
        {"robosuite": "robosuite==1.4.1"},
        version_fetcher=lambda _name: {"1.4.0", "1.4.1"},
    )
    assert findings == [], findings


def test_pinned_version_check_ignores_ranges_and_inconclusive():
    """Range specifiers and inconclusive (None) fetches must never fire."""
    # A range pin is not an exact `==`, so it is not a phantom-version candidate.
    assert (
        audit_deps.check_pinned_versions_exist(
            {"lerobot": "lerobot[feetech,dataset]>=0.6.0,<0.7.0"},
            version_fetcher=lambda _name: {"0.6.0"},
        )
        == []
    )
    # A None fetch (network flakiness) must be treated as inconclusive, not a
    # failure, so the audit never reddens the build on a transient error.
    assert (
        audit_deps.check_pinned_versions_exist(
            {"robomimic": "robomimic==0.5.0"},
            version_fetcher=lambda _name: None,
        )
        == []
    )


def test_vera_sim_has_no_phantom_robomimic_pin():
    """The live [vera-sim] extra must not pin robomimic (a phantom `==` version).

    robomimic must be a source install (documented in the extra), never a PyPI
    pin, so the unresolvable/confusion-prone `robomimic==0.5.0` cannot return.
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    vera_sim = data["project"]["optional-dependencies"]["vera-sim"]
    offenders = [spec for spec in vera_sim if Requirement(spec).name == "robomimic"]
    assert offenders == [], f"robomimic must not be a PyPI pin in [vera-sim]: {offenders}"


def test_vera_sim_is_forked_away_from_lerobot06_extras():
    """[tool.uv].conflicts must fork [vera-sim] from the lerobot-0.6 extras.

    [vera-sim]'s gymnasium==0.29.1 is mutually exclusive with lerobot>=0.6.0
    (gymnasium>=1.1.1). Without a conflict declaration uv resolves all extras
    jointly and that single pin drags the whole lock below lerobot 0.6 (the
    regression that froze uv.lock at lerobot 0.5.1). Each lerobot-0.6 extra must
    be declared as conflicting with vera-sim so uv forks the resolution instead.
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    conflicts = data.get("tool", {}).get("uv", {}).get("conflicts", [])
    forked = set()
    for pair in conflicts:
        extras = {member.get("extra") for member in pair}
        if "vera-sim" in extras:
            forked |= extras - {"vera-sim"}
    for extra in ("lerobot", "lerobot-async", "molmoact2", "all"):
        assert extra in forked, (
            f"[tool.uv].conflicts must fork vera-sim from the '{extra}' extra so "
            f"its gymnasium 0.29 pin cannot drag the lock below lerobot 0.6; "
            f"forked pairs found: {sorted(forked)}"
        )


# ---------------------------------------------------------------------------
# The IK solver stack must be declared by the extra that ships `move_to`.
#
# `move_to` is a Cartesian transport primitive in the MuJoCo backend's
# agent-callable action enum, and it solves inverse kinematics through
# `strands_robots.simulation.ik.MinkIKBridge`, i.e. through `mink` +
# `qpsolvers`. Neither was declared by any extra reachable from `[all]`, so on
# a `pip install "strands-robots[all]"` the action returned
# `IK bridge unavailable: ... No module named 'mink'` -- and the only reason CI
# did not see it was that the dev environment installed `mink` by hand. These
# guards pin the two halves of that: the extra declares the solver, and the dev
# env does not compensate for an undeclared dependency (which is what let the
# gap stay invisible while every IK test passed).
_SELF_NAME = "strands-robots"
_IK_SOLVER_PACKAGES = ("mink", "qpsolvers")


def _extra_requirements(extra: str) -> dict[str, set[str]]:
    """Distributions an extra pulls in, each mapped to the extras requested on it.

    Follows ``strands-robots[...]`` self-references, so a composite extra such as
    ``[all]`` reports its full closure.

    Args:
        extra: Name of the extra in ``[project.optional-dependencies]``.

    Returns:
        Mapping of lower-cased distribution name to the union of extras
        requested on that distribution (an empty set when it is required
        without any).
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    seen: set[str] = set()
    requested: dict[str, set[str]] = {}
    pending = [extra]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        assert current in extras, f"unknown extra {current!r}"
        for spec in extras[current]:
            req = Requirement(spec)
            if req.name.lower() == _SELF_NAME:
                pending.extend(req.extras)
                continue
            requested.setdefault(req.name.lower(), set()).update(req.extras)
    return requested


def _extra_closure(extra: str) -> set[str]:
    """Canonical names an extra pulls in, following ``strands-robots[...]`` self-references.

    Args:
        extra: Name of the extra in ``[project.optional-dependencies]``.

    Returns:
        The set of distribution names reachable from *extra*, lower-cased.
    """
    return set(_extra_requirements(extra))


def test_sim_mujoco_extra_declares_the_ik_solver_stack() -> None:
    """``[sim-mujoco]`` must declare the solver its ``move_to`` primitive needs.

    The extra ships the MuJoCo backend, whose action enum includes ``move_to``;
    that action is a dead end unless the same extra also brings the IK solver.
    """
    closure = _extra_closure("sim-mujoco")
    missing = [pkg for pkg in _IK_SOLVER_PACKAGES if pkg not in closure]
    assert not missing, (
        f"[sim-mujoco] ships the move_to primitive but does not declare {missing}; "
        f"move_to then returns 'IK bridge unavailable'. Declared: {sorted(closure)}"
    )


def test_all_extra_can_run_the_move_to_primitive() -> None:
    """``[all]`` must be able to honor every action the backends it installs advertise."""
    closure = _extra_closure("all")
    missing = [pkg for pkg in _IK_SOLVER_PACKAGES if pkg not in closure]
    assert not missing, (
        f"pip install 'strands-robots[all]' advertises move_to but does not install {missing}, so the action cannot run"
    )


def test_dev_env_does_not_install_undeclared_ik_dependencies() -> None:
    """The dev env must not hand-install the IK stack the extras are meant to declare.

    A test environment that adds a dependency no extra provides makes the IK
    paths green in CI while they are unreachable for users; the solver has to
    arrive through ``features = ["all"]`` like every other dependency.
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    env = data["tool"]["hatch"]["envs"]["default"]
    assert env["features"] == ["all"], "the dev env must install the package via features"
    offenders = [spec for spec in env.get("dependencies", []) if Requirement(spec).name.lower() in _IK_SOLVER_PACKAGES]
    assert not offenders, (
        f"the dev env pins {offenders} directly; declare them in the extra that "
        f"needs them instead so a user install matches CI"
    )


def test_ik_install_hints_name_only_declared_extras() -> None:
    """Every IK install hint must be a command of extras, not extras plus loose packages.

    A hint reading ``uv pip install 'strands-robots[sim-mujoco]' mink`` documents
    a dependency the extra does not declare. Each hint's install command must
    consist solely of ``strands-robots[...]`` specs whose closure provides the
    solver, so following it is sufficient.
    """
    from strands_robots.policies.cosmos3 import sim_ik as cosmos3_sim_ik
    from strands_robots.policies.vera import sim_ik as vera_sim_ik
    from strands_robots.simulation import ik as shared_ik

    hints = {
        "shared install": shared_ik._DEFAULT_INSTALL_HINT,
        "shared no-backend": shared_ik._DEFAULT_NO_BACKEND_MSG,
        "vera install": vera_sim_ik._install_hint(),
        "vera no-backend": vera_sim_ik._NO_BACKEND_MSG,
        "cosmos3 install": cosmos3_sim_ik._install_hint(),
        "cosmos3 no-backend": cosmos3_sim_ik._NO_BACKEND_MSG,
    }
    for label, hint in hints.items():
        commands = [line.split("uv pip install", 1)[1] for line in hint.splitlines() if "uv pip install" in line]
        assert commands, f"{label} hint has no install command: {hint!r}"
        for command in commands:
            specs = [token.strip("'\".") for token in command.split() if token.strip("'\". ")]
            assert specs, f"{label} hint has an empty install command"
            for spec in specs:
                req = Requirement(spec)
                assert req.name.lower() == _SELF_NAME, (
                    f"{label} hint tells the user to install {spec!r} alongside the "
                    f"extra, i.e. a dependency no extra declares: {command.strip()!r}"
                )
                for name in req.extras:
                    closure = _extra_closure(name)
                    missing = [pkg for pkg in _IK_SOLVER_PACKAGES if pkg not in closure]
                    assert not missing, f"{label} hint points at [{name}], which does not provide {missing}"


# ---------------------------------------------------------------------------
# Every extra a reader is told to install must be an extra that exists.
#
# History: docs/policies/vera.md led its install section with
# ``pip install 'strands-robots[vera]'`` -- an extra that pyproject.toml explains
# at length can never exist, because VERA ships only as a git repository and PyPI
# rejects metadata carrying a VCS reference. Two further sites named ``[isaac]``
# and ``[sim-libero]`` for what are really ``sim-isaac`` and ``benchmark-libero``.
#
# The failure mode is silent in the worst direction: pip does NOT fail on an
# unknown extra. ``pip install 'strands-robots[vera]'`` exits 0, prints one
# ``WARNING: strands-robots does not provide the extra 'vera'``, and installs the
# base package with none of the dependencies the reader was promised. The reader
# sees a successful install, then hits an ImportError somewhere unrelated-looking
# with nothing tying it back to the install step. That makes an extra name in an
# install hint load-bearing, and a typo in one is not cosmetic.
#
# Two guards, one per surface that hands a name to a user: written instructions,
# and the runtime ``require_optional(extra=...)`` messages.
# ---------------------------------------------------------------------------

# A qualified mention -- ``strands-robots[NAME]``. Only the qualified form is
# swept: a bare ``[wbc]`` in prose is ambiguous, because lerobot's own extras
# (``[smolvla]``, ``[pi]``, ``[dataset]``) are written exactly the same way, so
# requiring the distribution name keeps the sweep free of false positives.
_EXTRA_MENTION_RE = re.compile(r"strands[-_]robots\[([^\]\s]+)\]")

# A token that can be an extra name at all. Anything else caught by the mention
# regex is a template hole rather than an instruction -- ``[{extra}]`` in the
# message formatters, ``[<extra>]`` in a docstring, ``[...]`` as prose ellipsis --
# and naming a hole is not naming a missing extra.
_EXTRA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Trees that can carry an install instruction. CHANGELOG.md and changelog.d/ are
# deliberately absent, not overlooked: the log is a historical record, so an entry
# that fixes an extra name has to be free to quote the broken name, and an entry
# written while an extra existed must not be rewritten when it is later renamed.
_EXTRA_SCAN_ROOTS = ("strands_robots", "tests", "tests_integ", "examples", "docs", "scripts")
_EXTRA_SCAN_FILES = ("README.md", "pyproject.toml")

# Skipped by suffix rather than selected by it, so a mention in a file type nobody
# thought of -- a Dockerfile, a notebook, a compose file -- is still swept.
_BINARY_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".svg",
        ".pdf",
        ".mp4",
        ".webm",
        ".mov",
        ".zip",
        ".gz",
        ".tar",
        ".whl",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".onnx",
        ".safetensors",
        ".usd",
        ".usda",
        ".usdc",
        ".stl",
        ".obj",
        ".dae",
        ".bin",
        ".so",
        ".dylib",
        ".pyc",
        ".woff",
        ".woff2",
        ".ttf",
    }
)

_EXTRA_MENTION_ALLOWED = {
    # This test states the rule, so it quotes the broken names the rule forbids.
    "tests/test_dependency_audit.py",
    # Exercises the require_optional message formatter with a deliberately
    # synthetic extra ("my-extra"). It asserts the formatting, not the name.
    "tests/test_utils.py",
}


def _declared_extras() -> set[str]:
    """Normalized names in ``[project.optional-dependencies]``."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return {_normalize_extra(name) for name in data["project"]["optional-dependencies"]}


def _normalize_extra(name: str) -> str:
    """Normalize an extra the way PEP 685 requires a resolver to compare them.

    ``strands-robots[Sim_Isaac]`` really does install ``sim-isaac``, so comparing
    raw spelling would report a working command as broken.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _iter_scanned_files() -> list[Path]:
    files = [_REPO_ROOT / name for name in _EXTRA_SCAN_FILES]
    for root in _EXTRA_SCAN_ROOTS:
        files.extend(
            path
            for path in (_REPO_ROOT / root).rglob("*")
            if path.is_file() and path.suffix.lower() not in _BINARY_SUFFIXES and "__pycache__" not in path.parts
        )
    return [path for path in files if path.exists()]


def test_written_install_hints_name_only_declared_extras() -> None:
    """Every ``strands-robots[...]`` in the tree must name a declared extra.

    An undeclared name here is an instruction that installs nothing and still
    exits 0, so the reader is stranded by a command that reported success.
    """
    extras = _declared_extras()
    offenders: list[str] = []
    mentions = 0
    for path in _iter_scanned_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in _EXTRA_MENTION_ALLOWED:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for match in _EXTRA_MENTION_RE.finditer(line):
                for raw in (part.strip() for part in match.group(1).split(",")):
                    if not _EXTRA_NAME_RE.match(raw):
                        continue
                    mentions += 1
                    if _normalize_extra(raw) not in extras:
                        offenders.append(f"{rel}:{lineno} names [{raw}] -- {line.strip()[:100]}")
    # The sweep is only meaningful if it is actually reading the tree.
    assert mentions > 100, f"the extras sweep matched only {mentions} mentions; the scan roots have drifted"
    assert not offenders, (
        "these sites tell a reader to install an extra that does not exist; pip exits 0 on an "
        "unknown extra and installs none of the dependencies, so the failure surfaces later and "
        f"misattributed. Declared extras: {sorted(extras)}\n" + "\n".join(offenders)
    )


def test_require_optional_call_sites_name_declared_extras() -> None:
    """``require_optional(extra=...)`` must name a declared extra.

    The value is interpolated straight into the ImportError a user sees
    (``pip install strands-robots[<extra>]``), so an undeclared name here hands
    out a no-op install command at the exact moment the user needs a working one.
    """
    extras = _declared_extras()
    offenders: list[str] = []
    checked = 0
    for path in sorted((_REPO_ROOT / "strands_robots").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in {"require_optional", "require_optionals"}:
                continue
            for keyword in node.keywords:
                if keyword.arg != "extra" or not isinstance(keyword.value, ast.Constant):
                    continue
                value = keyword.value.value
                if not isinstance(value, str):
                    continue
                checked += 1
                if _normalize_extra(value.strip()) not in extras:
                    rel = path.relative_to(_REPO_ROOT).as_posix()
                    offenders.append(f"{rel}:{node.lineno} passes extra={value!r}")
    assert checked > 20, f"only {checked} literal extra= call sites found; the audit has stopped seeing them"
    assert not offenders, (
        "these require_optional call sites name an extra that does not exist, so the ImportError "
        "they raise tells the user to run an install that silently does nothing. Declared extras: "
        f"{sorted(extras)}\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# Declaring `qpsolvers` is not the same as declaring a QP backend.
#
# `qpsolvers` is a solver-agnostic front end: it ships no solver of its own, and
# each backend arrives through one of its extras (`qpsolvers[daqp]`, `[quadprog]`,
# ...). With none installed, `qpsolvers.available_solvers` is empty,
# `mink.solve_ik` cannot run, and `move_to` returns
#     IK bridge unavailable: No qpsolvers backend is installed; the mink IK
#     bridge needs one (e.g. 'daqp' or 'quadprog'). Install the sim extra:
#     uv pip install 'strands-robots[sim-mujoco]'.
# -- advising the extra that shipped the primitive. So an extra that solves IK
# has to declare a backend, or its own remedy cannot fix it.
#
# The `_IK_SOLVER_PACKAGES` guards above check distribution NAMES, which a bare
# `qpsolvers` satisfies; these check that a backend comes with it.
_QP_FRONTEND = "qpsolvers"

#: Extras whose code path calls `mink.solve_ik`: `[sim-mujoco]` ships `move_to`,
#: `[cosmos3-sim]` ships the Cosmos 3 -> MuJoCo bridge (both via MinkIKBridge).
_MINK_IK_EXTRAS = ("sim-mujoco", "cosmos3-sim")


def _declared_qp_backends(extra: str) -> set[str]:
    """qpsolvers backend extras that *extra* requests, following self-references."""
    return _extra_requirements(extra).get(_QP_FRONTEND, set())


@pytest.mark.parametrize("extra", _MINK_IK_EXTRAS)
def test_mink_ik_extra_declares_a_qp_backend(extra: str) -> None:
    """Every extra that solves IK through mink must declare a QP backend.

    Relying on `mink`'s own `qpsolvers[daqp]` pin leaves the guarantee resting on
    a transitive of a third-party package: if mink ever drops or renames it, the
    IK primitives break for anyone who installed exactly what this project asked
    them to.
    """
    backends = _declared_qp_backends(extra)
    assert backends, (
        f"[{extra}] solves IK via mink but declares {_QP_FRONTEND!r} with no "
        f"backend extra, so resolving 'strands-robots[{extra}]' need not install "
        f"any QP solver and mink.solve_ik cannot run. Declare one, e.g. "
        f"'{_QP_FRONTEND}[daqp]>=4.0.0'."
    )


def test_all_extra_declares_a_qp_backend() -> None:
    """`pip install 'strands-robots[all]'` must be able to complete an IK solve.

    `[all]` advertises `move_to` by installing the MuJoCo backend, so the QP
    backend that action needs has to be part of the same closure.
    """
    assert _declared_qp_backends("all"), (
        "pip install 'strands-robots[all]' advertises move_to but declares no "
        f"{_QP_FRONTEND} backend, so the action can return 'IK bridge unavailable'"
    )


def test_declared_qp_backends_are_real_qpsolvers_extras() -> None:
    """A declared backend must be an extra `qpsolvers` actually publishes.

    An unknown extra is not an install error - pip warns and installs nothing -
    so a typo such as `qpsolvers[dapq]` would resolve "successfully" and leave
    the solver missing exactly as before.
    """
    import importlib.metadata

    provided = importlib.metadata.metadata(_QP_FRONTEND).get_all("Provides-Extra") or []
    published = {name.lower() for name in provided}
    assert published, f"could not read {_QP_FRONTEND} extras from installed metadata"
    declared = {backend for extra in _MINK_IK_EXTRAS for backend in _declared_qp_backends(extra)}
    unknown = sorted(b for b in declared if b.lower() not in published)
    assert not unknown, (
        f"declared {_QP_FRONTEND} backend(s) {unknown} are not published extras of "
        f"{_QP_FRONTEND}; pip installs nothing for an unknown extra. Published: {sorted(published)}"
    )
