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
from typing import Any

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
# on every platform. Requiring lerobot >= 0.6 is therefore what lets those
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


def test_lerobot_extra_requires_at_least_0_6_1() -> None:
    """The ``[lerobot]`` extra must floor lerobot at >= 0.6.1.

    Two coupled reasons, either of which alone requires the floor:

    * The 0.5.1-era torch/torchcodec overrides were removed because lerobot
      0.6's own markers resolve the decoder stack correctly; that only holds
      for lerobot >= 0.6.
    * Bucket streaming (``stream_dataset(repo_type="bucket")``) needs a
      ``StreamingLeRobotDataset`` that accepts ``repo_type``, which 0.6.0 does
      not and 0.6.1 does - so 0.6.0 must be *excluded*, not merely admitted.
    """
    req = _lerobot_extra_requirement()
    # The declared lower BOUND, not membership of one version: a later raise
    # (say >=0.6.2) must not fail a guard whose requirement it still satisfies.
    lower = min(Version(s.version) for s in req.specifier if s.operator == ">=")
    assert lower >= Version("0.6.1"), f"lerobot floor must be >= 0.6.1, got {req.specifier}"
    assert Version("0.6.0") not in req.specifier, (
        f"lerobot floor must exclude 0.6.0 (its StreamingLeRobotDataset takes no "
        f"repo_type, so bucket streaming cannot be served), got {req.specifier}"
    )
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
# rejects metadata carrying a VCS reference. A further site named ``[isaac]``
# for what is really ``sim-isaac``.
#
# The failure mode is silent in the worst direction: pip does NOT fail on an
# unknown extra, and on a current pip it no longer even warns. Measured on pip
# 26.0.1 against a throwaway project whose only extra is ``real``, both
# ``pip install --dry-run --no-deps '.[real]'`` and the same command for
# ``'.[nope]'`` answer ``Would install extra-probe-0.0.1`` and exit 0, with no
# mention of ``nope`` anywhere in the output. The
# ``WARNING: ... does not provide the extra ...`` line older pip printed is gone,
# so the only remaining signal is the ImportError the reader meets later,
# somewhere unrelated-looking, with nothing tying it back to the install step.
# That makes an extra name in an install hint load-bearing, and a typo in one is
# not cosmetic.
#
# Three guards, one per surface that hands a name to a user: a written install
# command, a "which extra do I need" table column, and the runtime
# ``require_optional(extra=...)`` message.
# ---------------------------------------------------------------------------

# A qualified mention -- ``strands-robots[NAME]``. Only the qualified form is
# swept here: a bare ``[wbc]`` in prose is ambiguous, because lerobot's own extras
# (``[smolvla]``, ``[pi]``, ``[dataset]``) are written exactly the same way, so
# requiring the distribution name keeps the sweep free of false positives.
#
# That ambiguity is a property of prose, and it does not survive a table column
# whose header says what the column holds. A cell under ``| Install extra |`` in
# this project's own docs names one of this project's extras by construction, so
# the bare spelling is unambiguous there and is swept by
# ``test_install_extra_table_columns_name_only_declared_extras`` below. The
# policy catalogue is exactly such a column, and it is where the ``[vera]`` row in
# the history above survived the page fix.
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


# Headers that mean "the extra you install". A column merely containing the word
# is not one: ``Extra outputs`` in the Cosmos 3 page lists result keys, and a
# header whose prose mentions an extra in passing is prose.
_INSTALL_EXTRA_HEADERS = frozenset({"extra", "extras", "install extra", "install extras"})


def _install_extra_cells(text: str) -> list[tuple[int, str]]:
    """Return ``(line number, extra name)`` for each install-extra table cell.

    Args:
        text: Markdown source.

    Returns:
        One entry per backticked name in a column whose header is an
        install-extra header, in either cell form in use across the tree - a bare
        name (``sim-mujoco``) or the bracket a reader types (``[sim-mujoco]``).
        Reading only the bare form would leave the installation and architecture
        matrices, half the columns, ungraded. Fenced regions are skipped: a table
        inside a fence is sample output rather than an instruction.
    """
    found: list[tuple[int, str]] = []
    in_fence = False
    columns: list[int] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line.startswith("|"):
            columns = []
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        header = [index for index, cell in enumerate(cells) if cell.lower() in _INSTALL_EXTRA_HEADERS]
        if header:
            columns = header
            continue
        for index in columns:
            if index >= len(cells):
                continue
            for token in re.findall(r"`([^`]+)`", cells[index]):
                name = token[1:-1] if token.startswith("[") and token.endswith("]") else token
                if _EXTRA_NAME_RE.match(name):
                    found.append((number, name))
    return found


def test_install_extra_table_columns_name_only_declared_extras() -> None:
    """A cell under ``| Install extra |`` must name an extra that exists.

    This is the third surface: a reader choosing a provider or a feature scans
    the catalogue column rather than a prose command, and copies the name out of
    it. The qualified-mention sweep above cannot see that spelling, because the
    cell carries the bare name - the distribution is named by the header.
    """
    extras = _declared_extras()
    offenders: list[str] = []
    columns = 0
    cells = 0
    for path in _iter_scanned_files():
        if path.suffix.lower() != ".md":
            continue
        found = _install_extra_cells(path.read_text(encoding="utf-8", errors="ignore"))
        if not found:
            continue
        columns += 1
        cells += len(found)
        rel = path.relative_to(_REPO_ROOT).as_posix()
        offenders.extend(
            f"{rel}:{number} names [{name}]" for number, name in found if _normalize_extra(name) not in extras
        )
    # A scan that has stopped matching must fail rather than report a clean tree.
    assert columns >= 3, f"only {columns} file(s) with an install-extra column were read"
    assert cells >= 20, f"only {cells} extra name(s) were read out of install-extra columns"
    assert not offenders, (
        "these table cells tell a reader to install an extra that does not exist. pip exits 0 on "
        "an unknown extra without a warning, so the reader installs the base package and none of "
        f"the dependencies the row promised. Declared extras: {sorted(extras)}\n"
        + "\n".join(offenders)
        + "\nDeclare the extra, or say in the cell that there is none."
    )


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        pytest.param(
            "| Extra | Pulls in |\n|---|---|\n| `nope` | things |\n",
            ["nope"],
            id="bare-cell-form",
        ),
        pytest.param(
            "| Extra | Pulls in |\n|---|---|\n| `[nope]` | things |\n",
            ["nope"],
            id="bracket-cell-form",
        ),
        pytest.param(
            "| Provider | Class | Install extra |\n|---|---|---|\n| `p` | `C` | `nope` |\n",
            ["nope"],
            id="third-column",
        ),
        pytest.param(
            "| Extra outputs | Meaning |\n|---|---|\n| `last_rollout` | a path |\n",
            [],
            id="not-an-install-column",
        ),
        pytest.param(
            "```\n| Extra | Pulls in |\n|---|---|\n| `nope` | things |\n```\n",
            [],
            id="fenced-table-is-sample-output",
        ),
        pytest.param(
            "| Extra | Pulls in |\n|---|---|\n| _(none - git-only)_ | nothing |\n",
            [],
            id="a-cell-that-names-no-extra",
        ),
    ],
)
def test_install_extra_cell_reader_reads_install_columns_only(markdown: str, expected: list[str]) -> None:
    """The cell reader picks up install-extra cells in both forms, and nothing else.

    Graded on constructed markdown because the tree is clean once the catalogue is
    fixed, so the corpus can no longer exercise a rejection.
    """
    assert [name for _, name in _install_extra_cells(markdown)] == expected


def test_the_install_extra_cell_rule_can_both_accept_and_reject() -> None:
    """The rule is not constantly one answer."""
    extras = _declared_extras()
    outcomes = {
        _normalize_extra(name) in extras
        for _, name in _install_extra_cells("| Extra | Pulls in |\n|---|---|\n| `nope` | x |\n| `lerobot` | y |\n")
    }
    assert outcomes == {True, False}


def test_the_policy_catalogue_names_no_vera_extra() -> None:
    """VERA is the provider with no extra, so its row must not name one.

    ``pyproject.toml`` says at length that a ``vera`` extra can never exist - it
    would need a direct reference to the upstream git repository, which
    ``test_pyproject_has_no_direct_reference_dependency`` separately forbids.
    ``vera-sim`` exists and is a different thing: the gymnasium / robosuite
    evaluation stack, declared in conflict with the lerobot extras, so it is not
    the provider's install either.
    """
    extras = _declared_extras()
    assert "vera" not in extras
    assert "vera-sim" in extras
    overview = (_REPO_ROOT / "docs" / "policies" / "overview.md").read_text(encoding="utf-8")
    row = next(line for line in overview.splitlines() if line.startswith("| [`vera`]"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert "`vera`" not in cells[2], f"the install-extra cell names an extra: {cells[2]!r}"
    assert "none" in cells[2].lower(), f"the cell should say there is none: {cells[2]!r}"


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


# ---------------------------------------------------------------------------
# Environment audit: a downgraded `coverage` next to numba+robosuite (#1803).
#
# Installing Isaac Sim via the pip wheels (`isaacsim[all,extscache]`) silently
# downgrades `coverage` to the 7.4.4 that `isaacsim-kernel` pins. numba's
# tracer probe then fails, and the first *visible* symptom lands far from the
# cause: robosuite's OSC controller import dies with
#     AttributeError: module 'coverage.types' has no attribute 'Tracer'
# from whichever import happens to reach numba first. This guard turns the red
# herring into a named error at test-collection time instead: if this
# environment has numba and robosuite installed alongside a `coverage` old
# enough to trip the clash, fail loudly with the remedy.
#
# The floor was measured in #1805: coverage 7.6.0 still names the protocol
# `TracerCore`; `Tracer` exists only from 7.6.1 onward.
_COVERAGE_CLASH_FLOOR = Version("7.6.1")


def test_environment_coverage_is_compatible_with_numba_robosuite() -> None:
    """coverage<7.6.1 + numba + robosuite is a known-broken combination (#1803)."""
    import importlib.metadata

    versions: dict[str, str] = {}
    for dist in ("coverage", "numba", "robosuite"):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip(f"{dist} not installed; the numba/coverage clash cannot occur here")

    installed = Version(versions["coverage"])
    assert installed >= _COVERAGE_CLASH_FLOOR, (
        f"coverage=={versions['coverage']} is installed alongside "
        f"numba=={versions['numba']} and robosuite=={versions['robosuite']}: numba's "
        "tracer probe fails on this coverage (AttributeError: module "
        "'coverage.types' has no attribute 'Tracer'), which breaks robosuite's OSC "
        "controller import with a red-herring error far from the cause. This is "
        "the collateral of a pip-installed Isaac Sim (isaacsim-kernel pins "
        "coverage==7.4.4). Remedy: pip install "
        f"'coverage>={_COVERAGE_CLASH_FLOOR}' "
        "(the resulting pip conflict warning against isaacsim-kernel is cosmetic - "
        "coverage is kit test tooling, not a runtime dependency)."
    )


# ---------------------------------------------------------------------------
# Security floors for transitive packages.
#
# ``cbor2``, ``ujson``, ``twisted`` and ``pyopenssl`` each arrive only through
# another dependency (cbor2/ujson under autobahn, twisted under roslibpy,
# pyopenssl under twisted), so nothing in ``[project]`` declares a version for
# them. All four resolve above a HIGH advisory, which is why
# ``dependency-review-action`` (``fail-on-severity: high``) is green -- but the
# version was the resolver's choice, not a stated requirement, and cbor2 sits
# exactly ON its patch floor with no margin at all. Any input that moves one of
# them down re-introduces a HIGH advisory into the dependency graph GitHub
# scans: a transitive cap added upstream, a new marker branch, or another fork
# of the resolution under ``[tool.uv] conflicts`` is each sufficient, and none
# of them looks like a security change in review.
#
# ``[tool.uv] constraint-dependencies`` is the mechanism, and it is a constraint
# rather than an override for a measured reason. A constraint bounds a version
# and fails the resolution loudly when something genuinely requires less; an
# override *replaces* the conflicting requirement and resolves in silence.
# Measured on this manifest with ``gymnasium>=1.1.1``, which the ``[vera-sim]``
# extra contradicts by pinning ``gymnasium==0.29.1``:
#
#   as a constraint -> `uv lock` exits 1: "Because strands-robots[vera-sim]
#                      depends on gymnasium==0.29.1 and gymnasium>=1.1.1 ...
#                      requirements are unsatisfiable"
#   as an override  -> `uv lock` exits 0: "Updated gymnasium v0.29.1 -> v1.3.0"
#
# For a security floor the loud failure is the wanted behaviour: it says a
# dependency asked for a vulnerable version, which is the thing worth knowing.
# The floors below cost nothing to state -- adding them changed no resolved
# version (264 packages, zero differences).
#
# Each floor is the first version clearing every HIGH/CRITICAL advisory for that
# package, deliberately not the version currently resolved: the floor states the
# requirement, so it stays correct as the resolution moves above it.

_UV_LOCK = _REPO_ROOT / "uv.lock"

#: Transitive package -> the HIGH advisory its floor clears. These are the
#: packages a floor is *required* for; the constraint table may hold more.
_SECURITY_FLOORS: dict[str, str] = {
    "cbor2": "GHSA-3c37-wwvx-h642",
    "twisted": "GHSA-grgv-6hw6-v9g4",
    "ujson": "GHSA-c38f-wx89-p2xg",
    "pyopenssl": "GHSA-5pwr-322w-8jr4",
}


def _uv_requirement_list(key: str) -> list[str]:
    """A requirement list from ``[tool.uv]`` - the constraints or the overrides."""
    table = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8")).get("tool", {}).get("uv", {})
    raw = table.get(key, [])
    assert isinstance(raw, list), f"[tool.uv] {key} must be a list, got {type(raw).__name__}"
    return [str(spec) for spec in raw]


def _declared_constraints() -> dict[str, Requirement]:
    """``[tool.uv] constraint-dependencies`` keyed by distribution name."""
    return {Requirement(spec).name: Requirement(spec) for spec in _uv_requirement_list("constraint-dependencies")}


def _locked_versions() -> dict[str, list[Version]]:
    """Every version ``uv.lock`` resolves, keyed by name.

    A name maps to more than one version when ``[tool.uv] conflicts`` forks the
    resolution, so a floor has to hold for *each* fork, not just the first.
    """
    lock = tomllib.loads(_UV_LOCK.read_text(encoding="utf-8"))
    out: dict[str, list[Version]] = {}
    for package in lock.get("package", []):
        if "version" in package:
            out.setdefault(package["name"], []).append(Version(package["version"]))
    return out


def _unsatisfied_floors(constraints: dict[str, Requirement], locked: dict[str, list[Version]]) -> list[str]:
    """Names whose locked version falls below a declared floor.

    Pure so the check itself can be exercised against a synthetic lock; a
    package absent from the lock is not a violation (a constraint applies only
    where the package is actually resolved).
    """
    return sorted(
        name for name, req in constraints.items() for version in locked.get(name, []) if version not in req.specifier
    )


def _comment_block_above(entry_substring: str) -> str:
    """The contiguous ``#`` comment lines directly above a constraint entry."""
    lines = _PYPROJECT.read_text(encoding="utf-8").splitlines()
    hits = [i for i, line in enumerate(lines) if entry_substring in line and not line.lstrip().startswith("#")]
    assert len(hits) == 1, f"expected one entry line containing {entry_substring!r}, got {len(hits)}"
    index = hits[0] - 1
    collected: list[str] = []
    while index >= 0 and lines[index].lstrip().startswith("#"):
        collected.append(lines[index])
        index -= 1
    return "\n".join(reversed(collected))


def test_every_transitive_package_clearing_a_high_advisory_declares_a_floor() -> None:
    """Each of the four must carry a floor, so the version is stated not chosen.

    Without a floor the resolver is free to pick any release the graph allows,
    including one inside the advisory range. cbor2 makes the point: it resolves
    to 5.9.0, which *is* the first patched version for GHSA-3c37-wwvx-h642, so
    there is no margin whatsoever between the current resolution and a HIGH
    advisory re-entering the dependency graph.
    """
    declared = _declared_constraints()
    missing = sorted(set(_SECURITY_FLOORS) - set(declared))
    assert not missing, (
        f"transitive packages with no declared security floor: {missing}. Each "
        "resolves above a HIGH advisory only because the resolver happened to "
        "pick that version. Declare the floor in [tool.uv] constraint-dependencies."
    )


def test_every_declared_constraint_is_satisfied_by_the_locked_version() -> None:
    """A declared floor the lock does not meet is a floor in name only.

    The resolver enforces this when it runs, so a violation means the lock was
    not regenerated after the floor was raised - the same manifest/lock drift
    class the parity gate exists for, on the security-relevant field.
    """
    constraints = _declared_constraints()
    assert constraints, "no [tool.uv] constraint-dependencies declared, so this guard checks nothing"
    locked = _locked_versions()
    violations = _unsatisfied_floors(constraints, locked)
    assert not violations, (
        "uv.lock resolves these below their declared floor: "
        + ", ".join(f"{n} -> {[str(v) for v in locked[n]]} violates {constraints[n].specifier}" for n in violations)
        + ". Run `uv lock` and commit the result."
    )


def test_each_security_floor_names_the_advisory_it_clears() -> None:
    """A bare version pin is indistinguishable from an arbitrary one.

    The advisory id is what lets a later reader tell a security floor from a
    compatibility pin, and therefore what stops it being dropped as noise on
    the next dependency sweep.
    """
    for name, ghsa in sorted(_SECURITY_FLOORS.items()):
        comment = _comment_block_above(f'"{name}>=')
        assert ghsa in comment, f"the {name} floor does not name {ghsa} in the comment above it; got {comment!r}"


def test_the_security_floors_are_constraints_and_not_overrides() -> None:
    """These must bound the version, never replace the requirement.

    An override silently discards a conflicting requirement - measured on this
    manifest, ``gymnasium>=1.1.1`` as an override resolves cleanly while the
    ``[vera-sim]`` extra's ``gymnasium==0.29.1`` is dropped without a word. As a
    constraint the same floor fails the resolution and names the conflict. A
    security floor that hides "a dependency asked for a vulnerable version" has
    removed the signal it exists to raise.
    """
    overrides = {Requirement(spec).name for spec in _uv_requirement_list("override-dependencies")}
    misplaced = sorted(set(_SECURITY_FLOORS) & overrides)
    assert not misplaced, (
        f"security floors must live in constraint-dependencies, not "
        f"override-dependencies (an override masks a genuine conflict): {misplaced}"
    )


def test_the_floored_packages_are_transitive_rather_than_declared() -> None:
    """A floor in ``[tool.uv]`` is the right home only while these stay transitive.

    If one of them ever becomes a direct dependency, its version belongs in
    ``[project]`` beside the requirement that needs it, where the bound is
    visible to anyone reading the dependency list. This holds today and is
    asserted so the constraint table does not outlive its justification.
    """
    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    declared_names = {
        Requirement(spec).name
        for group in [project.get("dependencies", [])] + list(project.get("optional-dependencies", {}).values())
        for spec in group
        if not spec.startswith("strands-robots[")
    }
    direct = sorted(set(_SECURITY_FLOORS) & declared_names)
    assert not direct, (
        f"these carry a [tool.uv] constraint but are now declared directly, so the "
        f"bound belongs in [project] beside the requirement: {direct}"
    )


def test_the_floor_check_rejects_a_lock_below_a_declared_floor() -> None:
    """The satisfaction check must actually fire, not merely find nothing.

    Feeds a synthetic lock so a green suite means the floors hold, rather than
    meaning the comparison silently matched nothing.
    """
    constraints = {"cbor2": Requirement("cbor2>=5.9.0")}
    assert _unsatisfied_floors(constraints, {"cbor2": [Version("5.9.0")]}) == []
    assert _unsatisfied_floors(constraints, {"cbor2": [Version("5.8.0")]}) == ["cbor2"]
    # A fork that resolves one version below the floor is still a violation.
    assert _unsatisfied_floors(constraints, {"cbor2": [Version("5.9.0"), Version("5.8.0")]}) == ["cbor2"]
    # A package the lock does not resolve at all is not a violation.
    assert _unsatisfied_floors(constraints, {}) == []


# ---------------------------------------------------------------------------
# Cosmos 3 diffusers capability floor.
#
# The cosmos3-diffusers extra exists so Cosmos3Policy(backend="diffusers") can
# build a diffusers.Cosmos3OmniPipeline. Measured against the released wheels,
# that symbol (and CosmosActionCondition) first ships in diffusers 0.39.0 -
# 0.36.0, 0.37.1 and 0.38.0 carry neither - so a floor below 0.39 resolves to a
# diffusers the extra cannot use at all, and the backend's only remedy for that
# state is an ImportError hint the caller reaches after installing. The floor
# must therefore be the capability floor, not a nominal lower bound.
#
# The [tool.uv] override-dependencies diffusers pin cannot be left below that
# floor: a uv *override* REPLACES a requirement rather than intersecting with it,
# so it is the effective floor for the whole resolution. At >=0.38.0 it silently
# discarded the extra's floor and uv locked diffusers 0.38.0 - a release carrying
# no Cosmos3OmniPipeline at all.
# ---------------------------------------------------------------------------

_DIFFUSERS_WITHOUT_COSMOS3_OMNI = ("0.30.0", "0.36.0", "0.37.1", "0.38.0")
_FIRST_DIFFUSERS_WITH_COSMOS3_OMNI = "0.39.0"


def test_cosmos3_diffusers_floor_ships_the_omni_pipeline() -> None:
    """The extra's diffusers floor must exclude every release lacking the pipeline."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    extra = data["project"]["optional-dependencies"]["cosmos3-diffusers"]
    reqs = [Requirement(r) for r in extra]
    req = next((r for r in reqs if r.name == "diffusers"), None)
    assert req is not None, f"cosmos3-diffusers must declare diffusers, got {[r.name for r in reqs]}"
    for lacking in _DIFFUSERS_WITHOUT_COSMOS3_OMNI:
        assert Version(lacking) not in req.specifier, (
            f"diffusers {lacking} ships no Cosmos3OmniPipeline, so the cosmos3-diffusers "
            f"floor must exclude it; got {req.specifier}"
        )
    assert Version(_FIRST_DIFFUSERS_WITH_COSMOS3_OMNI) in req.specifier, (
        f"diffusers {_FIRST_DIFFUSERS_WITH_COSMOS3_OMNI} is the first release shipping "
        f"Cosmos3OmniPipeline and must stay installable; got {req.specifier}"
    )


def test_the_diffusers_uv_override_does_not_undercut_the_capability_floor() -> None:
    """The uv override must dominate both the CVE floor and the capability floor.

    A uv ``override-dependencies`` entry *replaces* every requirement for that
    package instead of intersecting with it, so it is the effective floor for the
    whole resolution: an override below the ``cosmos3-diffusers`` floor silently
    discards it. Measured - with the override at ``>=0.38.0`` and the extra at
    ``>=0.39``, ``uv lock`` pinned diffusers 0.38.0, which ships no
    ``Cosmos3OmniPipeline``. The override must therefore stay at or above the
    extra's floor while still excluding the releases the CVE fix predates.
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    overrides = data.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    diffusers_overrides = [Requirement(o) for o in overrides if Requirement(o).name == "diffusers"]
    assert diffusers_overrides, (
        f"the diffusers override must stay declared in [tool.uv].override-dependencies, got {overrides}"
    )
    for req in diffusers_overrides:
        assert Version("0.37.1") not in req.specifier, (
            f"the diffusers override must still exclude the unpatched 0.37.1; got {req}"
        )
        for lacking in _DIFFUSERS_WITHOUT_COSMOS3_OMNI:
            assert Version(lacking) not in req.specifier, (
                f"a uv override replaces the extra's requirement, so it must not admit "
                f"diffusers {lacking} (no Cosmos3OmniPipeline); got {req}"
            )
        assert Version(_FIRST_DIFFUSERS_WITH_COSMOS3_OMNI) in req.specifier, (
            f"the override must keep diffusers {_FIRST_DIFFUSERS_WITH_COSMOS3_OMNI} installable; got {req}"
        )


def test_the_lockfile_pins_a_diffusers_that_ships_the_omni_pipeline() -> None:
    """The committed lock must not pin a diffusers the extra cannot use.

    The floors above are declarations; this is the resolved fact a
    ``uv sync``/``uv run`` user actually gets.
    """
    lock = tomllib.loads((_REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = sorted({p["version"] for p in lock.get("package", []) if p["name"] == "diffusers" and p.get("version")})
    assert locked, "uv.lock must resolve diffusers (the cosmos3-diffusers extra declares it)"
    for version in locked:
        assert Version(version) >= Version(_FIRST_DIFFUSERS_WITH_COSMOS3_OMNI), (
            f"uv.lock pins diffusers {version}, which ships no Cosmos3OmniPipeline; "
            f"run `uv lock` after raising the floor"
        )


# ---------------------------------------------------------------------------
# Naming an extra that exists is not the same as naming one that supplies the
# module.
#
# The guard above pins that every `require_optional(extra=...)` names a declared
# extra. `[ros2]` is declared, so `require_optional("rclpy", extra="ros2")`
# passed it -- and still emitted a refusal whose every line was a dead end:
#
#     'rclpy' is required for the ROS 2 telemetry bridge (ros2_bridge=True)
#     Install with:
#       pip install 'strands-robots[ros2]'
#       pip install rclpy
#
# `pip install 'strands-robots[ros2]'` exits 0 having installed only the
# cyclonedds RMW binding, leaving rclpy exactly as missing -- verbatim the
# failure mode the comment above describes, an install that reported success and
# changed nothing. `pip install rclpy` fails outright ("No matching distribution
# found for rclpy"). So the operator who asked for `ros2_bridge=True` was handed
# two instructions and no way forward, while pyproject.toml, the `[ros2]` block
# in docs/ros2-integration.md, the `ros_telemetry` module docstring and the
# `use_ros` tool's own hint all already stated the remedy that works: source a
# system ROS 2 distro.
#
# pyproject.toml names the libraries this applies to verbatim -- "rclpy /
# rosidl_runtime_py ... are NOT distributed on PyPI ... and cannot be `pip
# install`ed" -- so the inventory below records that statement rather than making
# a new judgement. `sensor_msgs` is deliberately absent: its hint is the template
# `ros-<distro>-sensor-msgs`, and `ros-jazzy-sensor-msgs` does resolve on PyPI,
# so that hint is a hole for the reader to fill, not a dead end.
# ---------------------------------------------------------------------------

_SYSTEM_PROVIDED_MODULES = frozenset({"rclpy", "rosidl_runtime_py"})


def _unusable_remedy_reason(value: ast.expr) -> str | None:
    """Return why ``value`` cannot serve as a ``system_install=`` remedy.

    Only a literal is judged. The shipped form is a reference to a module-level
    constant, whose text is not knowable from the syntax tree, so a name, an
    attribute or an f-string is accepted here and left to the runtime tests in
    ``tests/test_utils.py``. A literal, though, is decidable, and three literals
    that satisfy the keyword's *presence* defeat what it is for -- each reason
    below names the message ``require_optional`` really renders for it.

    Args:
        value: The expression a call site passes as ``system_install=``.

    Returns:
        A reason naming the consequence, or ``None`` when the value is a usable
        remedy or is not statically decidable.
    """
    if not isinstance(value, ast.Constant):
        return None
    literal = value.value
    if literal is None:
        return "is None, which renders the pip block verbatim - the message passing nothing gives"
    if not isinstance(literal, str):
        return (
            f"is {type(literal).__name__} rather than a string, so str.join raises TypeError "
            f"and the documented ImportError never reaches the caller"
        )
    if not literal.strip():
        return "is blank, so the refusal names the module and carries no remedy at all"
    return None


def _require_optional_call_sites() -> list[tuple[str, int, str, dict[str, ast.expr]]]:
    """Every ``require_optional``/``require_optionals`` call site in the package.

    The keyword *values* are carried, not only their names: whether a remedy is
    usable is a property of the string it names, so a caller that reads only the
    names cannot tell a real hint from one that renders nothing.

    Returns:
        One ``(relative path, line number, module name, keyword-to-value)`` tuple
        per requested module -- the aggregate helper takes a list, so a single
        call can request several.
    """
    sites: list[tuple[str, int, str, dict[str, ast.expr]]] = []
    for path in sorted((_REPO_ROOT / "strands_robots").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in {"require_optional", "require_optionals"} or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                modules = [first.value]
            elif isinstance(first, ast.List | ast.Tuple):
                modules = [e.value for e in first.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            else:
                continue
            remedies = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            rel = path.relative_to(_REPO_ROOT).as_posix()
            sites.extend((rel, node.lineno, module, remedies) for module in modules)
    return sites


def test_require_optional_offers_no_pip_remedy_for_a_system_provided_module() -> None:
    """A module pip cannot supply must be refused with ``system_install=``.

    ``pip_install``/``extra`` both render a ``pip install`` line, and for these
    libraries every such line is an instruction the caller can follow to no
    effect. ``system_install`` replaces the block with the step that does supply
    the module, so passing it is what makes the refusal actionable.

    What makes a refusal actionable is the string, not the keyword: a
    ``system_install=`` carrying ``None`` renders the pip block byte-for-byte,
    which is the message this guard exists to forbid. So the value is graded too,
    by :func:`_unusable_remedy_reason`.
    """
    offenders: list[str] = []
    checked = 0
    for rel, lineno, module, remedies in _require_optional_call_sites():
        if module.split(".")[0] not in _SYSTEM_PROVIDED_MODULES:
            continue
        checked += 1
        pip_remedies = sorted(remedies.keys() & {"pip_install", "extra"})
        if pip_remedies:
            offenders.append(f"{rel}:{lineno} requests {module!r} but passes {', '.join(pip_remedies)}")
        elif "system_install" not in remedies:
            offenders.append(f"{rel}:{lineno} requests {module!r} with no system_install= remedy")
        elif (reason := _unusable_remedy_reason(remedies["system_install"])) is not None:
            offenders.append(f"{rel}:{lineno} requests {module!r} but its system_install= {reason}")
    assert checked, (
        f"no require_optional site requests any of {sorted(_SYSTEM_PROVIDED_MODULES)}; "
        f"the sweep has stopped seeing them and would pass vacuously"
    )
    assert not offenders, (
        "these sites refuse for want of a library that ships with a system ROS 2 install and is "
        "not on PyPI, yet hand the caller a pip command: an extra that installs something else "
        "and exits 0, or a distribution name that does not resolve. Pass "
        "system_install=ROS2_SYSTEM_INSTALL_HINT instead.\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# A ``system_install=`` remedy is graded by the string it carries.
#
# The sweep above once asked only whether the keyword was present. Three
# literals satisfy that and defeat what the keyword is for, one of them
# byte-for-byte reproducing the message the sweep exists to forbid. Both
# production sites pass a module-level constant, so nothing in the tree is an
# offender today and the corpus cannot exercise the rule -- the exemplars below
# grade the predicate directly instead, and the premises measure each refused
# literal against ``require_optional`` itself.
# ---------------------------------------------------------------------------

#: A module name no distribution supplies, so ``require_optional`` always
#: refuses it. Using an absent name rather than blocking a real one keeps these
#: probes hermetic: nothing is patched, and no import cache is disturbed.
_ABSENT_MODULE = "strands_robots_no_such_optional_dependency"

#: Call sources whose ``system_install=`` the rule must accept. The first two are
#: the shipped form (a reference to a constant, whose text the syntax tree cannot
#: know); the rest are values that do name a step a caller can take.
_ACCEPTED_REMEDIES = (
    'require_optional("rclpy", system_install=ROS2_SYSTEM_INSTALL_HINT)',
    'require_optional("rclpy", system_install=ros_telemetry.ROS2_SYSTEM_INSTALL_HINT)',
    'require_optional("rclpy", system_install="Source a distro: source /opt/ros/jazzy/setup.bash")',
    'require_optional("rclpy", system_install=f"Source {distro}/setup.bash")',
)

#: Call sources whose ``system_install=`` the rule must refuse, each with a
#: phrase its reason has to carry so a reader learns the consequence, not just
#: that the value was rejected. The phrases are chosen to distinguish the
#: branches from one another: ``"None"`` would not, because the non-string branch
#: reports ``NoneType`` and so satisfies it.
_REFUSED_REMEDIES = (
    ('require_optional("rclpy", system_install=None)', "pip block"),
    ('require_optional("rclpy", system_install="")', "blank"),
    ('require_optional("rclpy", system_install="   ")', "blank"),
    ('require_optional("rclpy", system_install=0)', "TypeError"),
    ('require_optional("rclpy", system_install=False)', "TypeError"),
)


def _remedy_node(source: str) -> ast.expr:
    """Return the ``system_install=`` value node of a single-call ``source``.

    Args:
        source: One Python expression statement calling ``require_optional``.

    Returns:
        The expression the call passes as ``system_install=``.
    """
    statement = ast.parse(source).body[0]
    assert isinstance(statement, ast.Expr), source
    call = statement.value
    assert isinstance(call, ast.Call), source
    return next(kw.value for kw in call.keywords if kw.arg == "system_install")


def _refusal_for(**overrides: Any) -> str:
    """Return the ``ImportError`` text ``require_optional`` renders for an absent module.

    Args:
        **overrides: Keyword arguments forwarded verbatim, so a probe may pass a
            value deliberately outside the declared ``str | None`` annotation.

    Returns:
        The refusal message.
    """
    from strands_robots import utils

    with pytest.raises(ImportError) as raised:
        utils.require_optional(_ABSENT_MODULE, purpose="a probe", **overrides)
    return str(raised.value)


@pytest.mark.parametrize("source", _ACCEPTED_REMEDIES)
def test_a_usable_system_install_remedy_is_accepted(source: str) -> None:
    """Over-reach control: grading the value must not refuse a real remedy."""
    assert _unusable_remedy_reason(_remedy_node(source)) is None, source


@pytest.mark.parametrize(("source", "expected_word"), _REFUSED_REMEDIES)
def test_a_system_install_remedy_that_supplies_nothing_is_refused(source: str, expected_word: str) -> None:
    """A literal that satisfies the keyword's presence but not its purpose."""
    reason = _unusable_remedy_reason(_remedy_node(source))
    assert reason is not None, source
    assert expected_word in reason, f"{source} -> {reason!r} does not name {expected_word!r}"


def test_the_remedy_rule_reaches_both_outcomes() -> None:
    """Non-vacuity: the exemplars are not all judged the same way."""
    verdicts = {
        _unusable_remedy_reason(_remedy_node(source)) is None
        for source in (*_ACCEPTED_REMEDIES, *(source for source, _ in _REFUSED_REMEDIES))
    }
    assert verdicts == {True, False}, f"the rule only ever answered {verdicts}"


def test_a_none_remedy_renders_the_pip_block_it_exists_to_replace() -> None:
    """Premise: ``system_install=None`` is byte-identical to omitting the keyword.

    This is why ``None`` is refused rather than merely discouraged. The keyword
    is present, so a presence check passes, and the message a caller reads is the
    dead-end pip instruction the guard above was written to forbid.
    """
    with_none = _refusal_for(system_install=None)
    assert with_none == _refusal_for(), "None must not be distinguishable from omitting the keyword"
    assert f"pip install {_ABSENT_MODULE}" in with_none


def test_a_blank_remedy_renders_no_remedy_at_all() -> None:
    """Premise: a blank ``system_install=`` leaves the refusal with no next step."""
    message = _refusal_for(system_install="")
    assert message.strip() == f"'{_ABSENT_MODULE}' is required for a probe"
    assert "pip install" not in message
    assert "Install with:" not in message


def test_a_non_string_remedy_replaces_the_documented_importerror() -> None:
    """Premise: a non-string remedy raises ``TypeError`` from ``str.join``.

    ``require_optional`` documents ``ImportError``, and a caller writing
    ``except ImportError`` around an optional import misses this entirely.
    """
    from strands_robots import utils

    overrides: dict[str, Any] = {"system_install": 0}
    with pytest.raises(TypeError, match="expected str instance"):
        utils.require_optional(_ABSENT_MODULE, purpose="a probe", **overrides)


def test_a_real_remedy_replaces_the_pip_block() -> None:
    """Premise: the accepted form does what the keyword promises."""
    message = _refusal_for(system_install="Source a distro first.")
    assert message.endswith("Source a distro first.")
    assert "pip install" not in message


def test_the_rclpy_refusals_name_the_step_that_supplies_it() -> None:
    """Both rclpy refusals must point at sourcing a distro, not at installing it.

    Asserted on the messages the two production sites really raise, with the
    import forced to fail so the check holds whether or not the interpreter
    running the suite happens to have a ROS 2 distro sourced.
    """
    from strands_robots import utils

    class _BlockRclpy:
        """Meta-path finder that makes ``import rclpy`` fail."""

        def find_spec(self, name: str, path: object = None, target: object = None) -> None:
            """Refuse ``rclpy`` and defer every other name to the real finders."""
            if name == "rclpy" or name.startswith("rclpy."):
                raise ImportError("rclpy blocked for this test")
            return None

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "meta_path", [_BlockRclpy(), *sys.meta_path])
        # A fresh cache, restored on exit: require_optional short-circuits on a
        # module it has already resolved.
        patch.setattr(utils, "_lazy_modules", {})

        from strands_robots.hardware_robot import Robot
        from strands_robots.ros_telemetry import RosTelemetryBridge

        messages = {}
        with pytest.raises(ImportError) as bridge_error:
            RosTelemetryBridge(domain_id=0)
        messages["RosTelemetryBridge()"] = str(bridge_error.value)
        with pytest.raises(ImportError) as robot_error:
            Robot._check_ros2_bridge_deps(ros2_transport="rclpy")
        messages["Robot(ros2_bridge=True)"] = str(robot_error.value)

    for label, message in messages.items():
        assert "source /opt/ros/" in message, f"{label} names no way to obtain rclpy:\n{message}"
        assert "pip install rclpy" not in message, (
            f"{label} tells the caller to install a distribution that does not exist:\n{message}"
        )
        assert "Install with:\n  pip install 'strands-robots[ros2]'" not in message, (
            f"{label} leads with an extra that installs the cyclonedds binding and leaves rclpy missing:\n{message}"
        )


# ---------------------------------------------------------------------------
# [sim-newton]: one MuJoCo series for one newton minor (#3012)
# ---------------------------------------------------------------------------
# newton enforces its MuJoCo requirement at RUNTIME, from a declaration the
# RESOLVER never sees: it lives under newton's own `[sim]` extra, which
# `[sim-newton]` does not install, and
# `solver_mujoco._warn_if_mujoco_versions_mismatch` reads newton's METADATA and
# matches `^mujoco(?=[<>=!~])([^;]+)`, truncating at the `;` and discarding the
# `extra == "sim"` marker. So the pins that bind at import are invisible to uv,
# and the only place they can be stated is `[sim-newton]` itself.
#
# What that requirement is per release cannot be read offline - it is newton's
# published metadata, not anything in this tree - so these tests grade the
# SHAPE that has to hold for any newton the extra admits, rather than copying
# newton's table here where it would rot. The requirement is always a single
# `~=X.Y.0` series, which is what makes the shape decisive:
#
#   * both MuJoCo distributions carry ONE shared series - capping only
#     `mujoco-warp` leaves the `mujoco` half of the warning standing;
#   * the `newton` range admits ONE minor - the series is per minor, and the
#     three the old `>=1.3.0,<2.0.0` admitted disagree (1.3.0 wants `~=3.8.0`,
#     1.4.0 `~=3.10.0`, 1.5.x `~=3.11.0`), so no single MuJoCo pin can satisfy
#     a multi-minor range;
#   * the narrowed `mujoco` stays inside `[sim-mujoco]`'s own range, so an
#     extra never refuses a sibling a version that sibling declares.
_SIM_NEWTON_EXTRA = "sim-newton"
_SIM_MUJOCO_EXTRA = "sim-mujoco"
# Inputs to `SpecifierSet.contains`, not a claim about which releases exist, so
# no published wheel can make this sweep stale.
_VERSION_SWEEP = tuple(f"{major}.{minor}.{patch}" for major in range(7) for minor in range(41) for patch in (0, 1))


def _declared_requirement(extra: str, distribution: str) -> Requirement | None:
    """The requirement *extra* declares for *distribution*, if it declares one.

    Args:
        extra: Name of the extra in ``[project.optional-dependencies]``.
        distribution: Canonical distribution name to look for.

    Returns:
        The parsed :class:`Requirement`, or ``None`` if *extra* names no such
        distribution directly. A distribution reached only through a
        ``strands-robots[...]`` self-reference is deliberately not found: what
        these tests grade is what this extra itself narrows.
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    for raw in data["project"]["optional-dependencies"][extra]:
        requirement = Requirement(raw)
        if requirement.name == distribution:
            return requirement
    return None


def _admitted_series(requirement: Requirement) -> set[tuple[int, int]]:
    """The ``(major, minor)`` series *requirement* admits, over ``_VERSION_SWEEP``."""
    admitted: set[tuple[int, int]] = set()
    for candidate in _VERSION_SWEEP:
        if requirement.specifier.contains(candidate):
            version = Version(candidate)
            admitted.add((version.major, version.minor))
    return admitted


def test_the_series_reader_separates_a_one_series_pin_from_a_wider_one() -> None:
    """The shape rule must be able to both accept and reject.

    The three rejected specifiers are the ones ``[sim-newton]`` carried before
    #3012, so this also records that the rule fails on pre-fix pins rather than
    being satisfied by them.
    """
    assert _admitted_series(Requirement("mujoco>=3.11.0,<3.12.0")) == {(3, 11)}
    assert len(_admitted_series(Requirement("mujoco-warp>=3.8.0"))) > 1
    assert len(_admitted_series(Requirement("newton>=1.3.0,<2.0.0"))) > 1
    assert len(_admitted_series(Requirement("mujoco>=3.5.0,<4.0.0"))) > 1


def test_sim_newton_declares_both_mujoco_distributions() -> None:
    """``[sim-newton]`` must name ``mujoco`` itself, not inherit it.

    Guards the three tests below against passing vacuously, and pins the
    substance of the ``mujoco`` half: inheriting ``[sim-mujoco]``'s
    ``>=3.5.0,<4.0.0`` is what let the resolver take mujoco to 3.12.0 while
    newton required the 3.11 series.
    """
    missing = [
        distribution
        for distribution in ("mujoco", "mujoco-warp")
        if _declared_requirement(_SIM_NEWTON_EXTRA, distribution) is None
    ]
    assert not missing, (
        f"[sim-newton] does not declare {missing}; newton enforces a MuJoCo series at "
        f"import that the resolver cannot see, so an inherited range does not bound it"
    )


def test_sim_newton_pins_both_mujoco_distributions_to_one_shared_series() -> None:
    """The two MuJoCo pins must be the same single series.

    newton requires one ``~=X.Y.0`` for both distributions and warns once per
    distribution that misses it, so a pin on one of them alone closes half the
    finding.
    """
    mujoco = _declared_requirement(_SIM_NEWTON_EXTRA, "mujoco")
    mujoco_warp = _declared_requirement(_SIM_NEWTON_EXTRA, "mujoco-warp")
    assert mujoco is not None and mujoco_warp is not None, (
        "[sim-newton] must declare both mujoco and mujoco-warp directly; "
        "test_sim_newton_declares_both_mujoco_distributions names why"
    )
    mujoco_series = _admitted_series(mujoco)
    mujoco_warp_series = _admitted_series(mujoco_warp)
    assert mujoco_series == mujoco_warp_series, (
        f"[sim-newton] admits mujoco {sorted(mujoco_series)} but mujoco-warp "
        f"{sorted(mujoco_warp_series)}; newton requires one series for both, so the "
        f"distribution outside the agreed series still warns"
    )
    assert len(mujoco_series) == 1, (
        f"[sim-newton] admits more than one MuJoCo series {sorted(mujoco_series)}; "
        f"newton's requirement is a single ~=X.Y.0, so a wider range admits a resolve it refuses"
    )


def test_sim_newton_admits_a_single_newton_minor() -> None:
    """The ``newton`` range must admit one minor, because the MuJoCo series follows it.

    A cap at the next major would admit a newton minor free to require a
    different MuJoCo series, reopening #3012 with no edit in this repository to
    attribute it to. Capping the minor instead makes such a bump an unresolved
    requirement somebody has to act on. This is a deliberate exception to the
    cap-the-major convention, and it is scoped to this extra.
    """
    newton = _declared_requirement(_SIM_NEWTON_EXTRA, "newton")
    assert newton is not None, "[sim-newton] declares no newton"
    series = _admitted_series(newton)
    assert len(series) == 1, (
        f"[sim-newton] admits newton minors {sorted(series)}; each chooses its own MuJoCo "
        f"series, so the pins here can only be correct for one of them"
    )


def test_the_sim_newton_mujoco_pin_stays_inside_the_sim_mujoco_range() -> None:
    """Narrowing a sibling's pin may not refuse a version that sibling declares.

    ``[sim-newton]`` depends on ``[sim-mujoco]``, and uv resolves one mujoco for
    the environment, so this pin is what a joint install gets. Keeping it a
    subset means the narrowing only moves a ``[sim-mujoco]`` user inside the
    range their own extra already allows, and that the intersection is
    non-empty rather than an unresolvable pair.
    """
    narrowed = _declared_requirement(_SIM_NEWTON_EXTRA, "mujoco")
    inherited = _declared_requirement(_SIM_MUJOCO_EXTRA, "mujoco")
    assert narrowed is not None and inherited is not None, (
        "both [sim-newton] and [sim-mujoco] must declare mujoco for the subset relation between them to be checkable"
    )
    admitted = [c for c in _VERSION_SWEEP if narrowed.specifier.contains(c)]
    assert admitted, f"[sim-newton]'s mujoco pin {narrowed.specifier} admits no release"
    outside = [c for c in admitted if not inherited.specifier.contains(c)]
    assert not outside, (
        f"[sim-newton] admits mujoco {outside[:5]} which [sim-mujoco]'s {inherited.specifier} "
        f"does not; the two extras would resolve to an empty intersection"
    )
